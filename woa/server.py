import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .core import Store, model_checkpoint_path
from .hardware import detect_hardware
from .training import ModelRuntime


WEB_ROOT = Path(__file__).parent / "web"
MAX_BODY_SIZE = 1024 * 1024


def handler_for(store, runtime=None):
    runtime = runtime or ModelRuntime(
        model_checkpoint_path(store.path.parent / "model.pt")
    )

    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "WoaAweOo/0.2"

        def log_message(self, format_string, *args):
            print(f"[woa] {self.address_string()} {format_string % args}")

        def send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Invalid Content-Length.") from error
            if length <= 0 or length > MAX_BODY_SIZE:
                raise ValueError("Request body must be between 1 byte and 1 MiB.")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Request body must be valid JSON.") from error
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def is_local_origin(self, origin):
            try:
                parsed_origin = urlparse(origin)
                return (
                    parsed_origin.scheme == "http"
                    and parsed_origin.hostname
                    in ("127.0.0.1", "localhost")
                    and parsed_origin.port == self.server.server_port
                    and not parsed_origin.username
                    and not parsed_origin.password
                    and not parsed_origin.path
                    and not parsed_origin.query
                    and not parsed_origin.fragment
                )
            except ValueError:
                return False

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/health":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "model_available": runtime.available,
                        "model_kind": "project_trained_transformer",
                        "hardware": detect_hardware(),
                    },
                )
                return
            if path == "/api/training/events":
                self.send_json(
                    200,
                    {"events": store.list_training_events()},
                )
                return
            if path == "/api/conversations":
                self.send_json(200, {"conversations": store.list_conversations()})
                return
            if path.startswith("/api/conversations/") and path.endswith("/messages"):
                conversation_id = unquote(
                    path[len("/api/conversations/") : -len("/messages")].rstrip("/")
                )
                if not store.get_conversation(conversation_id):
                    self.send_json(404, {"error": "Conversation not found."})
                    return
                self.send_json(
                    200, {"messages": store.list_messages(conversation_id)}
                )
                return
            if path == "/" or path == "/index.html":
                self.send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
                return
            self.send_json(404, {"error": "Not found."})

        def send_file(self, path, content_type):
            try:
                body = path.read_bytes()
            except OSError:
                self.send_json(500, {"error": "The web app could not be loaded."})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = urlparse(self.path).path
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self.send_json(415, {"error": "Content-Type must be application/json."})
                return
            origin = self.headers.get("Origin")
            if origin and not self.is_local_origin(origin):
                self.send_json(403, {"error": "Cross-origin requests are not allowed."})
                return
            try:
                payload = self.read_json()
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            if path == "/api/conversations":
                title = payload.get("title", "New conversation")
                if not isinstance(title, str) or not title.strip():
                    self.send_json(400, {"error": "Conversation title is required."})
                    return
                self.send_json(201, {"conversation": store.create_conversation(title)})
                return
            prefix = "/api/conversations/"
            suffix = "/messages"
            if path.startswith(prefix) and path.endswith(suffix):
                conversation_id = unquote(path[len(prefix) : -len(suffix)].rstrip("/"))
                if not store.get_conversation(conversation_id):
                    self.send_json(404, {"error": "Conversation not found."})
                    return
                content = payload.get("content")
                if not isinstance(content, str) or not content.strip():
                    self.send_json(400, {"error": "Message content is required."})
                    return
                user_message = store.add_message(
                    conversation_id, "user", content.strip()
                )
                try:
                    history = store.list_messages(conversation_id)[-40:]
                    reply = runtime.generate(history)
                except (RuntimeError, ValueError) as error:
                    self.send_json(
                        503,
                        {"error": str(error), "message": user_message},
                    )
                    return
                assistant_message = store.add_message(
                    conversation_id, "assistant", reply
                )
                self.send_json(
                    201,
                    {"messages": [user_message, assistant_message]},
                )
                return
            self.send_json(404, {"error": "Not found."})

    return RequestHandler


def create_server(host=None, port=None, store=None, runtime=None):
    host = host or os.environ.get("WOA_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            "The local service supports only 127.0.0.1 or localhost binding."
        )
    port = int(os.environ.get("WOA_PORT", "8765") if port is None else port)
    return ThreadingHTTPServer(
        (host, port), handler_for(store or Store(), runtime=runtime)
    )


def start_server_in_thread(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
