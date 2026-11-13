import argparse
import os
import threading
import webbrowser
from pathlib import Path

from .client import api_url, request_json
from .core import Store, model_checkpoint_path
from .desktop import run as run_desktop
from .hardware import detect_hardware, recommended_model_settings
from .model import ModelConfig
from .server import create_server
from .sources import MAX_SOURCE_BYTES, fetch_public_text
from .training import train


def print_message(message):
    label = "you" if message["role"] == "user" else "Woa-Awe-oo"
    print(f"\n[{label}]\n{message['content']}\n", flush=True)


def interactive_chat():
    try:
        health = request_json("/api/health")
        print(
            "Project-trained model is available."
            if health["model_available"]
            else "No project-trained model yet; train one with `python -m woa train --data PATH`."
        )
        conversations = request_json("/api/conversations")["conversations"]
        if conversations:
            active_id = conversations[0]["id"]
        else:
            active_id = request_json(
                "/api/conversations", {"title": "New conversation"}
            )["conversation"]["id"]
    except RuntimeError as error:
        print(error)
        return 1

    state_lock = threading.Lock()
    state = {"conversation_id": active_id, "seen_ids": set()}

    def display_history(conversation_id):
        try:
            messages = request_json(
                f"/api/conversations/{conversation_id}/messages"
            )["messages"]
        except RuntimeError as error:
            print(f"\n[error] {error}", flush=True)
            return
        with state_lock:
            if state["conversation_id"] != conversation_id:
                return
            state["seen_ids"] = {item["id"] for item in messages}
        if messages:
            print(f"\n--- {len(messages)} messages ---")
            for item in messages:
                print_message(item)

    def sync_loop():
        last_error = None
        while not stop_sync.wait(1):
            with state_lock:
                conversation_id = state["conversation_id"]
                seen_ids = set(state["seen_ids"])
            try:
                messages = request_json(
                    f"/api/conversations/{conversation_id}/messages"
                )["messages"]
            except RuntimeError:
                if last_error is None:
                    print("\n[error] Lost connection to the local service.", flush=True)
                last_error = "disconnected"
                continue
            if last_error is not None:
                print("\n[info] Reconnected to the local service.", flush=True)
                last_error = None
            for item in messages:
                if item["id"] not in seen_ids:
                    print_message(item)
                    seen_ids.add(item["id"])
            with state_lock:
                if state["conversation_id"] == conversation_id:
                    state["seen_ids"] = seen_ids

    display_history(active_id)
    stop_sync = threading.Event()
    threading.Thread(target=sync_loop, daemon=True).start()
    print(
        f"Connected to {api_url()}. Commands: /new, /list, /use NUMBER, "
        "/status, /events, /quit"
    )
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/list":
                try:
                    items = request_json("/api/conversations")["conversations"]
                    for index, item in enumerate(items, start=1):
                        marker = "*" if item["id"] == state["conversation_id"] else " "
                        print(f"{marker} {index}. {item['title']} ({item['id']})")
                except RuntimeError as error:
                    print(error)
                continue
            if text == "/status":
                try:
                    health = request_json("/api/health")
                    hardware = health["hardware"]
                    accelerator = (
                        f" ({hardware['accelerator']})"
                        if hardware["accelerator"]
                        else ""
                    )
                    print(
                        f"Model checkpoint: {'found' if health['model_available'] else 'not trained'}; "
                        f"compute: {hardware['device']}{accelerator}"
                    )
                except RuntimeError as error:
                    print(error)
                continue
            if text == "/events":
                try:
                    events = request_json("/api/training/events")["events"]
                    for event in events[:10]:
                        print(f"{event['created_at']} {event['event_type']} {event['payload']}")
                except RuntimeError as error:
                    print(error)
                continue
            if text == "/new":
                try:
                    active_id = request_json(
                        "/api/conversations", {"title": "New conversation"}
                    )["conversation"]["id"]
                    with state_lock:
                        state["conversation_id"] = active_id
                        state["seen_ids"] = set()
                except RuntimeError as error:
                    print(error)
                continue
            if text.startswith("/use "):
                try:
                    index = int(text[5:]) - 1
                    items = request_json("/api/conversations")["conversations"]
                    active_id = items[index]["id"]
                except (ValueError, IndexError):
                    print("Use `/list` to see valid conversation numbers.")
                    continue
                except RuntimeError as error:
                    print(error)
                    continue
                with state_lock:
                    state["conversation_id"] = active_id
                    state["seen_ids"] = set()
                display_history(active_id)
                continue
            with state_lock:
                conversation_id = state["conversation_id"]
            try:
                request_json(
                    f"/api/conversations/{conversation_id}/messages",
                    {"content": text},
                )
            except RuntimeError as error:
                print(f"[error] {error}")
    except KeyboardInterrupt:
        print()
    finally:
        stop_sync.set()
    return 0


def main():
    parser = argparse.ArgumentParser(prog="woa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="start the shared local service")
    serve_parser.add_argument("--host", default=os.environ.get("WOA_HOST", "127.0.0.1"))
    serve_parser.add_argument("--port", type=int, default=int(os.environ.get("WOA_PORT", "8765")))
    subparsers.add_parser("web", help="open the browser client")
    subparsers.add_parser("desktop", help="open the desktop client")
    subparsers.add_parser("chat", help="start the interactive CLI")
    train_parser = subparsers.add_parser(
        "train", help="train the project-owned decoder-only model from local text"
    )
    train_parser.add_argument("--data", nargs="+", required=True, metavar="PATH")
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--context-length", type=int)
    train_parser.add_argument("--width", type=int)
    train_parser.add_argument("--layers", type=int)
    train_parser.add_argument("--heads", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=50)
    train_parser.add_argument("--checkpoint-interval", type=int, default=100)
    train_parser.add_argument("--output")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue training from the checkpoint at --output (or the default model path)",
    )
    ingest_parser = subparsers.add_parser(
        "ingest", help="fetch one allowlisted HTTPS text/JSON/CSV dataset endpoint"
    )
    ingest_parser.add_argument("--url", required=True)
    ingest_parser.add_argument("--allow-host", nargs="+", required=True)
    ingest_parser.add_argument("--output", required=True)
    ingest_parser.add_argument("--max-bytes", type=int, default=MAX_SOURCE_BYTES)
    args = parser.parse_args()

    if args.command == "serve":
        try:
            server = create_server(host=args.host, port=args.port)
        except ValueError as error:
            print(f"Cannot start service: {error}")
            return 1
        print(f"Woa-Awe-oo service listening at http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping service.")
        finally:
            server.server_close()
    elif args.command == "web":
        url = api_url()
        print(f"Opening {url}")
        webbrowser.open(url)
    elif args.command == "desktop":
        run_desktop()
    elif args.command == "chat":
        return interactive_chat()
    elif args.command == "train":
        store = Store()
        profile = detect_hardware()
        recommendation = recommended_model_settings(profile)
        output_path = Path(args.output) if args.output else model_checkpoint_path()
        try:
            if args.resume and any(
                value is not None
                for value in (
                    args.context_length,
                    args.width,
                    args.layers,
                    args.heads,
                )
            ):
                raise ValueError(
                    "Do not override architecture dimensions when resuming."
                )
            config = None if args.resume else ModelConfig(
                context_length=args.context_length or recommendation["context_length"],
                d_model=args.width or recommendation["d_model"],
                n_heads=args.heads or recommendation["n_heads"],
                n_layers=args.layers or recommendation["n_layers"],
            )
            accelerator = (
                f" ({profile['accelerator']})" if profile["accelerator"] else ""
            )
            if args.resume:
                print(
                    f"Detected {profile['device']}{accelerator}; resuming {output_path}.",
                    flush=True,
                )
            else:
                print(
                    f"Detected {profile['device']}{accelerator}; "
                    f"starting with {config.d_model} width, {config.n_layers} layers, "
                    f"context {config.context_length}.",
                    flush=True,
                )
            store.record_training_event(
                "training_started",
                {
                    "data_paths": args.data,
                    "steps": args.steps,
                    "hardware": profile,
                    "config": config.to_dict() if config else None,
                    "resume": args.resume,
                },
            )

            def report(event):
                print(
                    f"step {event['step']}/{event['total_steps']} "
                    f"loss={event['loss']:.4f} lr={event['learning_rate']:.3g} "
                    f"device={event['device']}",
                    flush=True,
                )
                if event["step"] % 10 == 0 or event["step"] == event["total_steps"]:
                    store.record_training_event("training_step", event)
                if event["checkpoint_saved"]:
                    store.record_training_event(
                        "checkpoint_saved",
                        {"step": event["step"], "path": str(output_path)},
                    )

            output = train(
                args.data,
                output_path,
                steps=args.steps,
                config=config,
                batch_size=args.batch_size,
                learning_rate_value=args.learning_rate,
                warmup_steps=args.warmup_steps,
                on_step=report,
                resume_from=output_path if args.resume else None,
                checkpoint_interval=args.checkpoint_interval,
            )
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as error:
            store.record_training_event(
                "training_failed",
                {"error": str(error)},
            )
            print(f"Training failed: {error}")
            return 1
        print(f"Saved project-trained model to {output}")
    elif args.command == "ingest":
        try:
            output = Path(args.output)
            if output.suffix.lower() not in {".txt", ".md"}:
                raise ValueError("Ingestion output must end in .txt or .md.")
            text, source_info = fetch_public_text(
                args.url,
                args.allow_host,
                max_bytes=args.max_bytes,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as destination:
                destination.write(text)
        except (RuntimeError, ValueError, OSError) as error:
            print(f"Ingestion failed: {error}")
            return 1
        Store().record_training_event(
            "source_ingested",
            {**source_info, "output": str(output)},
        )
        print(
            f"Saved {source_info['bytes']} bytes from {source_info['host']} "
            f"to {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
