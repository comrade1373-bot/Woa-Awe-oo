import argparse
import importlib.util
import os
import sqlite3
import sys
import threading
import time
import webbrowser
from importlib.resources import files as package_files
from pathlib import Path

from .alignment import train_dpo
from .bundled_knowledge import BUNDLED_SOURCES, seed_bundled_knowledge
from .client import api_url, request_json
from .core import Store, data_directory, model_checkpoint_path
from .corpus import download_starter_corpus
from .datasets import (
    build_preference_examples,
    build_sft_examples,
    load_chat_records,
    load_preference_records,
)
from .desktop import run as run_desktop
from .hardware import detect_hardware, recommended_model_settings
from .knowledge import KnowledgeIndex
from .model import ModelConfig
from .server import create_server, start_server_in_thread
from .sources import MAX_SOURCE_BYTES, fetch_public_text
from .training import evaluate_checkpoint, train


def print_message(message):
    label = "you" if message["role"] == "user" else "Woa"
    print(f"\n[{label}]\n{message['content']}\n", flush=True)


def interactive_chat():
    try:
        health = request_json("/api/health")
        print(
            "Experimental project model checkpoint found."
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
                with state_lock:
                    if state["conversation_id"] != conversation_id:
                        continue
                if last_error is None:
                    print("\n[error] Lost connection to the local service.", flush=True)
                last_error = "disconnected"
                continue
            with state_lock:
                if state["conversation_id"] != conversation_id:
                    continue
            if last_error is not None:
                print("\n[info] Reconnected to the local service.", flush=True)
                last_error = None
            with state_lock:
                if state["conversation_id"] == conversation_id:
                    for item in messages:
                        if item["id"] not in seen_ids:
                            print_message(item)
                            seen_ids.add(item["id"])
                    state["seen_ids"] = seen_ids

    display_history(active_id)
    stop_sync = threading.Event()
    threading.Thread(target=sync_loop, daemon=True).start()
    print(
        f"Connected to {api_url()}. Commands: /new, /list, /use NUMBER, "
        "/status, /events, /tools, /calc EXPR, /solve EQUATION, /entropy TEXT, "
        "/search QUERY, /digest QUERY, /about, /cues TEXT, /quit. "
        "You can also ask to calculate, search your files, or summarize a topic."
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


def run_diagnostics():
    print(f"Python: {sys.version.split()[0]}")
    try:
        local_data = data_directory()
        checkpoint = model_checkpoint_path()
        missing_bundled = [
            source["filename"]
            for source in BUNDLED_SOURCES
            if not (
                package_files("woa")
                .joinpath("data")
                .joinpath("knowledge")
                .joinpath(source["filename"])
                .is_file()
            )
        ]
        store = Store()
        documents = KnowledgeIndex(store).list_documents()
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
        print(f"Local data store: unavailable ({error})")
        print("Check write access to WOA_DATA_DIR and local SQLite/FTS5 support.")
        return 1

    print(f"Local data directory: {local_data}")
    print(
        f"Local SQLite/FTS5 store: ready ({len(documents)} indexed documents)"
    )
    print(
        f"Bundled knowledge: {'ready' if not missing_bundled else 'incomplete'} "
        f"({len(BUNDLED_SOURCES) - len(missing_bundled)}/{len(BUNDLED_SOURCES)} files)"
    )
    if missing_bundled:
        print("Missing bundled files: " + ", ".join(missing_bundled))
    print(
        f"PyTorch: {'available' if importlib.util.find_spec('torch') else 'not installed'}"
    )
    print(
        f"Model checkpoint: {'found' if checkpoint.is_file() else 'not found'} "
        f"({checkpoint})"
    )

    try:
        health = request_json("/api/health")
        tools = request_json("/api/tools")["tools"]
        service_documents = request_json("/api/knowledge/documents")["documents"]
    except (RuntimeError, ValueError) as error:
        print(f"Local service: unavailable ({error})")
        print("Start it with `python -m woa start`.")
        return 1
    if health.get("model_kind") != "project_trained_transformer":
        print("The service responded, but it is not a Woa-Awe-oo service.")
        return 1
    hardware = health["hardware"]
    print(f"Local service: online at {api_url()}")
    print(
        f"Model checkpoint: {'file found' if health['model_available'] else 'not trained'}; "
        f"compute: {hardware['device']}"
    )
    print("Local tools: " + ", ".join(tool["name"] for tool in tools))
    print(f"Indexed knowledge documents: {len(service_documents)}")
    print("Service features: " + ", ".join(health.get("features", [])))
    print("Clients: web, desktop, CLI (shared service and conversation database)")
    return 0


def launch_application(client, host, port):
    if not 1 <= port <= 65535:
        print("The local service port must be between 1 and 65535.")
        return 1
    os.environ["WOA_PORT"] = str(port)
    server = None
    try:
        try:
            health = request_json("/api/health")
        except RuntimeError as error:
            if not str(error).startswith("Cannot connect to "):
                raise
            health = None

        if health is not None:
            if (
                health.get("ok") is not True
                or health.get("model_kind") != "project_trained_transformer"
            ):
                raise RuntimeError(
                    f"A non-Woa service is responding at {api_url()}."
                )
            required_features = {
                "shared_conversations",
                "tool_bus",
                "knowledge_index",
                "serialized_turns",
                "mewtronic_environment",
            }
            missing_features = required_features - set(health.get("features", []))
            if missing_features:
                raise RuntimeError(
                    "The running service is outdated; restart it to enable: "
                    + ", ".join(sorted(missing_features))
                )
            print(f"Using the existing local service at {api_url()}.")
        else:
            server = create_server(host=host, port=port)
            start_server_in_thread(server)
            print(f"Started the local service at {api_url()}.")

        if client == "web":
            url = api_url()
            try:
                opened = webbrowser.open(url)
            except webbrowser.Error:
                opened = False
            if opened:
                print("Opened the web client.")
            else:
                print(f"Open the web client at {url}.")
            if server is None:
                return 0
            print("Keep this terminal open; press Ctrl+C to stop the service.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping the local service.")
            return 0
        if client == "desktop":
            run_desktop()
            return 0
        return interactive_chat()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Could not launch the local assistant: {error}")
        return 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


def main():
    parser = argparse.ArgumentParser(prog="woa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="start the shared local service")
    serve_parser.add_argument("--host", default=os.environ.get("WOA_HOST", "127.0.0.1"))
    serve_parser.add_argument("--port", type=int, default=int(os.environ.get("WOA_PORT", "8765")))
    start_parser = subparsers.add_parser(
        "start", help="start the shared service and launch a client"
    )
    start_parser.add_argument(
        "--client", choices=("web", "desktop", "chat"), default="web"
    )
    start_parser.add_argument("--host", default=os.environ.get("WOA_HOST", "127.0.0.1"))
    start_parser.add_argument("--port", type=int, default=int(os.environ.get("WOA_PORT", "8765")))
    subparsers.add_parser("doctor", help="check service, model, tools, and local index")
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
    train_parser.add_argument(
        "--exact-million",
        action="store_true",
        help="use the learned-position architecture with exactly 1,000,000 trainable parameters",
    )
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
    eval_parser = subparsers.add_parser(
        "eval", help="measure next-token loss and perplexity on local text"
    )
    eval_parser.add_argument("--data", nargs="+", required=True, metavar="PATH")
    eval_parser.add_argument("--model", default=str(model_checkpoint_path()))
    corpus_parser = subparsers.add_parser(
        "download-starter",
        help="download bounded, curated public-domain and open-data text sources",
    )
    corpus_parser.add_argument(
        "--category",
        action="append",
        choices=(
            "scripture",
            "esoteric",
            "economics",
            "aerospace",
            "forensics",
            "philosophy",
            "science",
            "art",
            "technology",
            "communication",
            "language",
        ),
        help="category to fetch; repeat for multiple categories (default: all)",
    )
    corpus_parser.add_argument(
        "--output-dir",
        default=str(data_directory() / "corpora"),
    )
    sft_parser = subparsers.add_parser(
        "sft", help="supervised fine-tune a project checkpoint on chat JSONL"
    )
    sft_parser.add_argument("--data", nargs="+", required=True, metavar="PATH")
    sft_parser.add_argument(
        "--init-from",
        help="base project model checkpoint (defaults to the current model)",
    )
    sft_parser.add_argument("--output")
    sft_parser.add_argument("--steps", type=int, default=500)
    sft_parser.add_argument("--batch-size", type=int)
    sft_parser.add_argument("--learning-rate", type=float, default=5e-5)
    sft_parser.add_argument("--warmup-steps", type=int, default=20)
    sft_parser.add_argument("--checkpoint-interval", type=int, default=100)
    sft_parser.add_argument("--resume", action="store_true")
    dpo_parser = subparsers.add_parser(
        "dpo", help="preference-tune a project model on chosen/rejected JSONL pairs"
    )
    dpo_parser.add_argument("--data", nargs="+", required=True, metavar="PATH")
    dpo_parser.add_argument(
        "--reference",
        help="frozen reference project checkpoint (defaults to current model)",
    )
    dpo_parser.add_argument("--output")
    dpo_parser.add_argument("--steps", type=int, default=300)
    dpo_parser.add_argument("--batch-size", type=int)
    dpo_parser.add_argument("--learning-rate", type=float, default=1e-5)
    dpo_parser.add_argument("--beta", type=float, default=0.1)
    dpo_parser.add_argument("--checkpoint-interval", type=int, default=100)
    dpo_parser.add_argument("--resume", action="store_true")
    ingest_parser = subparsers.add_parser(
        "ingest", help="fetch one allowlisted HTTPS text/JSON/CSV dataset endpoint"
    )
    ingest_parser.add_argument("--url", required=True)
    ingest_parser.add_argument("--allow-host", nargs="+", required=True)
    ingest_parser.add_argument("--output", required=True)
    ingest_parser.add_argument("--max-bytes", type=int, default=MAX_SOURCE_BYTES)
    digest_parser = subparsers.add_parser(
        "digest", help="index local text, Markdown, and code for cited search"
    )
    digest_parser.add_argument("paths", nargs="+", metavar="PATH")
    search_parser = subparsers.add_parser(
        "search", help="search the local knowledge index"
    )
    search_parser.add_argument("query", nargs="+")
    args = parser.parse_args()

    if args.command == "start":
        try:
            seed_bundled_knowledge(KnowledgeIndex(Store()))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not load bundled local knowledge: {error}")
            return 1
        return launch_application(args.client, args.host, args.port)
    elif args.command == "doctor":
        return run_diagnostics()
    elif args.command == "digest":
        try:
            results = KnowledgeIndex(Store()).ingest_paths(args.paths)
        except (ValueError, OSError) as error:
            print(f"Digest failed: {error}")
            return 1
        total_chunks = sum(result["chunks"] for result in results)
        total_bytes = sum(result["bytes"] for result in results)
        print(
            f"Indexed {len(results)} documents ({total_chunks} chunks, "
            f"{total_bytes} bytes) in the local knowledge store."
        )
    elif args.command == "search":
        try:
            results = KnowledgeIndex(Store()).search(" ".join(args.query))
        except ValueError as error:
            print(f"Search failed: {error}")
            return 1
        if not results:
            print("No matching excerpts found in the local knowledge index.")
        for result in results:
            print(
                f"\n[{result['source']} - chunk {result['chunk']} "
                f"- score {result['score']}]\n{result['content']}"
            )
    elif args.command == "serve":
        try:
            seed_bundled_knowledge(KnowledgeIndex(Store()))
            server = create_server(host=args.host, port=args.port)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Cannot start service: {error}")
            return 1
        print(f"Woa service listening at http://{args.host}:{args.port}")
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
            explicit_dimensions = (
                args.context_length,
                args.width,
                args.layers,
                args.heads,
            )
            if args.exact_million and args.resume:
                raise ValueError("--exact-million cannot be combined with --resume.")
            if args.exact_million and any(
                value is not None for value in explicit_dimensions
            ):
                raise ValueError(
                    "Do not combine --exact-million with individual architecture dimensions."
                )
            if args.resume and any(
                value is not None
                for value in explicit_dimensions
            ):
                raise ValueError(
                    "Do not override architecture dimensions when resuming."
                )
            if args.resume:
                config = None
            elif args.exact_million:
                config = ModelConfig(
                    context_length=204,
                    d_model=100,
                    n_heads=5,
                    n_layers=7,
                    learned_position_embeddings=True,
                )
            else:
                config = ModelConfig(
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
                progress_interval = max(1, event["total_steps"] // 100)
                if event["step"] % progress_interval == 0 or event["checkpoint_saved"]:
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
    elif args.command == "sft":
        store = Store()
        base_path = Path(args.init_from) if args.init_from else model_checkpoint_path()
        output_path = (
            Path(args.output)
            if args.output
            else base_path.with_name(f"{base_path.stem}-sft{base_path.suffix}")
        )
        try:
            if args.resume and args.init_from:
                raise ValueError("Use either --resume or --init-from, not both.")
            if args.resume and args.output is None:
                raise ValueError("--resume requires an explicit --output checkpoint.")
            records = load_chat_records(args.data)
            if args.resume:
                config = None
                initialize_from = None
                resume_from = output_path
                import torch

                resume = torch.load(output_path, map_location="cpu", weights_only=True)
                config = ModelConfig(**resume["config"])
            else:
                if not base_path.is_file():
                    raise ValueError(
                        f"Base model checkpoint not found: {base_path}. "
                        "Run `python -m woa train --data ...` first."
                    )
                import torch

                base = torch.load(base_path, map_location="cpu", weights_only=True)
                config = ModelConfig(**base["config"])
                initialize_from = base_path
                resume_from = None
            examples = build_sft_examples(records, config.context_length)
            store.record_training_event(
                "training_started",
                {
                    "stage": "sft",
                    "data_paths": args.data,
                    "examples": len(examples),
                    "steps": args.steps,
                    "base_checkpoint": str(base_path),
                    "output": str(output_path),
                },
            )

            def report_sft(event):
                progress_interval = max(1, event["total_steps"] // 100)
                if event["step"] % progress_interval == 0 or event["checkpoint_saved"]:
                    print(
                        f"SFT step {event['step']}/{event['total_steps']} "
                        f"loss={event['loss']:.4f} device={event['device']}",
                        flush=True,
                    )
                if event["step"] % 10 == 0 or event["checkpoint_saved"]:
                    store.record_training_event("training_step", event)
                if event["checkpoint_saved"]:
                    store.record_training_event(
                        "checkpoint_saved",
                        {"stage": "sft", "step": event["step"], "path": str(output_path)},
                    )

            output = train(
                [],
                output_path,
                steps=args.steps,
                config=config,
                batch_size=args.batch_size,
                learning_rate_value=args.learning_rate,
                warmup_steps=args.warmup_steps,
                checkpoint_interval=args.checkpoint_interval,
                examples=examples,
                initialize_from=initialize_from,
                resume_from=resume_from,
                on_step=report_sft,
            )
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as error:
            store.record_training_event(
                "training_failed",
                {"stage": "sft", "error": str(error)},
            )
            print(f"SFT failed: {error}")
            return 1
        print(f"Saved instruction-tuned checkpoint to {output}")
    elif args.command == "dpo":
        store = Store()
        reference_path = (
            Path(args.reference) if args.reference else model_checkpoint_path()
        )
        output_path = (
            Path(args.output)
            if args.output
            else reference_path.with_name(
                f"{reference_path.stem}-aligned{reference_path.suffix}"
            )
        )
        try:
            if not reference_path.is_file():
                raise ValueError(f"Reference checkpoint not found: {reference_path}")
            if args.resume and args.output is None:
                raise ValueError("--resume requires an explicit --output checkpoint.")
            if args.resume and args.reference is None:
                raise ValueError(
                    "Specify --reference when resuming so the frozen reference is stable."
                )
            records = load_preference_records(args.data)
            import torch

            reference = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
            config = ModelConfig(**reference["config"])
            examples = build_preference_examples(records, config.context_length)
            store.record_training_event(
                "training_started",
                {
                    "stage": "dpo",
                    "data_paths": args.data,
                    "examples": len(examples),
                    "steps": args.steps,
                    "reference": str(reference_path),
                    "output": str(output_path),
                },
            )

            def report_dpo(event):
                progress_interval = max(1, event["total_steps"] // 100)
                if event["step"] % progress_interval == 0 or event["checkpoint_saved"]:
                    print(
                        f"DPO step {event['step']}/{event['total_steps']} "
                        f"loss={event['loss']:.4f} device={event['device']}",
                        flush=True,
                    )
                if event["step"] % 10 == 0 or event["checkpoint_saved"]:
                    store.record_training_event("training_step", event)
                if event["checkpoint_saved"]:
                    store.record_training_event(
                        "checkpoint_saved",
                        {
                            "stage": "dpo",
                            "step": event["step"],
                            "path": str(output_path),
                        },
                    )

            output = train_dpo(
                examples,
                reference_path,
                output_path,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                beta=args.beta,
                checkpoint_interval=args.checkpoint_interval,
                resume_from=output_path if args.resume else None,
                on_step=report_dpo,
            )
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as error:
            store.record_training_event(
                "training_failed",
                {"stage": "dpo", "error": str(error)},
            )
            print(f"DPO failed: {error}")
            return 1
        print(f"Saved preference-aligned checkpoint to {output}")
    elif args.command == "eval":
        try:
            metrics = evaluate_checkpoint(args.model, args.data)
        except (RuntimeError, ValueError, OSError) as error:
            print(f"Evaluation failed: {error}")
            return 1
        print(
            f"tokens={metrics['tokens']} "
            f"cross_entropy={metrics['mean_cross_entropy']:.4f} "
            f"perplexity={metrics['perplexity']:.3f} "
            f"device={metrics['device']}"
        )
    elif args.command == "download-starter":
        def report_download(record):
            print(
                f"Downloaded {record['name']}: {record['bytes']} bytes -> "
                f"{record['path']}"
            )

        try:
            records = download_starter_corpus(
                args.output_dir,
                categories=args.category,
                on_download=report_download,
            )
        except (RuntimeError, ValueError, OSError) as error:
            print(f"Dataset download failed: {error}")
            return 1
        print(
            f"Downloaded {len(records)} source files. Provenance and rights notes: "
            f"{Path(args.output_dir) / 'sources.json'}"
        )
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
