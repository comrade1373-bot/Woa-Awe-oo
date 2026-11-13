import contextlib
from concurrent.futures import ThreadPoolExecutor
import gzip
import io
import importlib.util
import json
import os
import re
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from woa.agent import LocalAgent, ToolInvocationError
from woa.bundled_knowledge import BUNDLED_SOURCES, seed_bundled_knowledge
from woa.communication import communication_cues
from woa.client import request_json
from woa.cli import (
    launch_application,
    main as cli_main,
    run_diagnostics,
)
from woa.core import Store
from woa.corpus import _nasa_abstract_text, _world_bank_text, download_starter_corpus
from woa.desktop import DesktopClient
from woa.datasets import (
    build_preference_examples,
    build_sft_examples,
    create_synthetic_sft_dataset,
    load_chat_records,
    load_preference_records,
)
from woa.hardware import recommended_batch_size, recommended_model_settings
from woa.knowledge import KnowledgeIndex
from woa.model import ModelConfig
from woa.server import create_server
from woa.sources import fetch_public_text, validate_source_url
from woa.tokenizer import ByteTokenizer
from woa.training import (
    ModelRuntime,
    evaluate_checkpoint,
    learning_rate,
    load_text_corpus,
    train,
)
from woa.alignment import train_dpo
from woa.tools import (
    ToolBus,
    build_tool_bus,
    calculate,
    information_entropy,
    solve_polynomial,
)


class StoreTests(unittest.TestCase):
    def test_user_message_updates_shared_conversation_title(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "woa.db")
            conversation = store.create_conversation()

            store.add_message(conversation["id"], "user", "First question")

            self.assertEqual(
                store.get_conversation(conversation["id"])["title"],
                "First question",
            )
            self.assertEqual(
                [item["content"] for item in store.list_messages(conversation["id"])],
                ["First question"],
            )

    def test_list_messages_limit_returns_latest_messages_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "woa.db")
            conversation = store.create_conversation()
            for content in ("first", "second", "third"):
                store.add_message(conversation["id"], "user", content)

            messages = store.list_messages(conversation["id"], limit=2)

            self.assertEqual(
                [message["content"] for message in messages],
                ["second", "third"],
            )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                store.list_messages(conversation["id"], limit=0)

    def test_training_events_are_shared_and_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "woa.db")
            store.record_training_event("step", {"loss": 1.25, "step": 4})

            events = store.list_training_events()

            self.assertEqual(events[0]["event_type"], "step")
            self.assertEqual(events[0]["payload"], {"loss": 1.25, "step": 4})


class KnowledgeTests(unittest.TestCase):
    def test_ingest_search_and_replace_preserve_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            index = KnowledgeIndex(Store(Path(directory) / "woa.db"))
            index.ingest_text(
                "notes.md",
                "Aerospace logistics depend on reliable supply chains.",
            )

            results = index.search("aerospace supply")

            self.assertEqual(results[0]["source"], "notes.md")
            self.assertEqual(results[0]["chunk"], 1)
            self.assertIn("reliable supply chains", results[0]["content"])
            index.ingest_text("notes.md", "Marine biology studies ocean life.")
            self.assertEqual(index.search("aerospace"), [])
            self.assertEqual(len(index.list_documents()), 1)

    def test_digest_paths_reject_unsupported_and_invalid_utf8_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supported = root / "guide.md"
            supported.write_text("A local document.", encoding="utf-8")
            index = KnowledgeIndex(Store(root / "woa.db"))
            self.assertEqual(len(index.ingest_paths([supported])), 1)

            unsupported = root / "image.png"
            unsupported.write_bytes(b"\x89PNG")
            with self.assertRaisesRegex(ValueError, "Unsupported file type"):
                index.ingest_paths([unsupported])

            invalid_text = root / "invalid.txt"
            invalid_text.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                index.ingest_paths([invalid_text])

    def test_bundled_knowledge_is_searchable_and_seed_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            index = KnowledgeIndex(Store(Path(directory) / "woa.db"))

            first = seed_bundled_knowledge(index)
            second = seed_bundled_knowledge(index)

            self.assertEqual(len(first), len(BUNDLED_SOURCES))
            self.assertEqual(second, [])
            self.assertTrue(
                any(
                    hit["source"] == "bundled/plato-republic-gutenberg-1497.txt"
                    for hit in index.search("Socrates justice")
                )
            )
            self.assertTrue(
                any(
                    hit["source"] == "bundled/nasa-forensics-public-abstracts.txt"
                    for hit in index.search("forensic science evidence")
                )
            )


class ToolTests(unittest.TestCase):
    def test_communication_cues_are_explicit_and_non_persistent(self):
        cues = communication_cues(
            "Please help ASAP; this is the best ever, like a rocket! 😂"
        )

        self.assertEqual(cues["intent"], "request")
        self.assertIn("urgency_words", cues["tone_cues"])
        self.assertIn("playful_marker", cues["tone_cues"])
        self.assertIn("possible_hyperbole", cues["rhetorical_cues"])
        self.assertFalse(cues["persistent_profile"])
        self.assertEqual(cues["certainty"], "heuristic")

    def test_calculator_supports_arithmetic_and_rejects_code(self):
        self.assertEqual(calculate("2 + 3 * 4"), 14)
        self.assertEqual(calculate("-(8 / 2)"), -4.0)
        for expression in (
            "__import__('os').system('whoami')",
            "1 / 0",
            "2 ** 13",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculate(expression)

    def test_quadratic_solver_handles_linear_complex_and_invalid_inputs(self):
        self.assertEqual(
            solve_polynomial("x**2 - 5*x + 6 = 0")["roots"],
            [2.0, 3.0],
        )
        self.assertEqual(solve_polynomial("2*x + 4 = 0")["roots"], [-2.0])
        self.assertEqual(
            solve_polynomial("x**2 + 1 = 0")["roots"],
            [
                {"real": 0.0, "imaginary": -1.0},
                {"real": 0.0, "imaginary": 1.0},
            ],
        )
        self.assertEqual(solve_polynomial("x = x")["roots"], "all real numbers")
        self.assertEqual(solve_polynomial("x = x + 1")["roots"], [])
        for equation in (
            "x**3 + 1 = 0",
            "1/x = 2",
            "__import__('os')",
            "x = x = 1",
        ):
            with self.subTest(equation=equation):
                with self.assertRaises(ValueError):
                    solve_polynomial(equation)

    def test_shannon_entropy_is_local_and_bounded(self):
        metrics = information_entropy("a b")

        self.assertAlmostEqual(
            metrics["character_entropy_bits_per_symbol"], 1.584962500721156
        )
        self.assertEqual(metrics["token_entropy_bits_per_token"], 1.0)
        self.assertIn("not a measure of truth", metrics["note"])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            information_entropy("  ")

    def test_tool_bus_validates_arguments_and_agent_routes_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            index = KnowledgeIndex(Store(Path(directory) / "woa.db"))
            index.ingest_text("facts.txt", "Saturn has prominent rings.")
            bus = build_tool_bus(index)

            self.assertEqual(
                bus.invoke("calculator", {"expression": "6 * 7"})["result"],
                42,
            )
            self.assertEqual(
                bus.invoke(
                    "algebraic_solve",
                    {"equation": "x**2 - 5*x + 6 = 0"},
                )["roots"],
                [2.0, 3.0],
            )
            self.assertEqual(
                bus.invoke("information_entropy", {"text": "a b"})[
                    "token_entropy_bits_per_token"
                ],
                1.0,
            )
            self.assertEqual(
                bus.invoke(
                    "communication_cues",
                    {"text": "Please help ASAP."},
                )["tone_cues"],
                ["urgency_words", "polite_marker"],
            )
            with self.assertRaisesRegex(ValueError, "Missing tool arguments"):
                bus.invoke("calculator", {})
            with self.assertRaisesRegex(ValueError, "Unknown tool"):
                bus.invoke("unknown", {})

            class FakeRuntime:
                calls = 0

                def generate(self, _messages):
                    self.calls += 1
                    return "model response"

            runtime = FakeRuntime()
            agent = LocalAgent(runtime, bus)
            self.assertEqual(
                agent.reply([{"role": "user", "content": "/calc 6 * 7"}]),
                "42",
            )
            self.assertEqual(
                agent.reply([{"role": "user", "content": "/calc 2 plus 2"}]),
                "4",
            )
            self.assertEqual(
                agent.reply(
                    [{"role": "user", "content": "Please calculate 12 divided by 3 plus 1."}]
                ),
                "5.0",
            )
            self.assertEqual(
                agent.reply([{"role": "user", "content": "What is 2 plus 2?"}]),
                "4",
            )
            labels = agent.reply(
                [{"role": "user", "content": "/cues I'm not sure; maybe later."}]
            )
            self.assertIn('"intent": "statement"', labels)
            self.assertIn("uncertainty_words", labels)
            with self.assertRaisesRegex(ToolInvocationError, "Add text to label"):
                agent.reply([{"role": "user", "content": "/cues"}])
            help_reply = agent.reply(
                [
                    {
                        "role": "user",
                        "content": "What can you help me with locally?",
                    }
                ]
            )
            self.assertIn("You can ask naturally", help_reply)
            self.assertIn("/cues TEXT", help_reply)
            self.assertEqual(runtime.calls, 0)

            class ToolCallingRuntime:
                calls = []

                def generate(self, messages):
                    self.calls.append(messages)
                    return '<tool>calculator {"expression":"6 * 7"}</tool>'

            tool_runtime = ToolCallingRuntime()
            tool_agent = LocalAgent(tool_runtime, bus)
            self.assertEqual(
                tool_agent.reply(
                    [{"role": "user", "content": "I need an arithmetic result."}]
                ),
                "42",
            )
            self.assertEqual(len(tool_runtime.calls), 1)
            self.assertEqual(tool_runtime.calls[0][-2]["role"], "system")
            self.assertIn("<tool>", tool_runtime.calls[0][-2]["content"])
            self.assertIn("dependency order", tool_runtime.calls[0][-2]["content"])
            self.assertIn("completion checks", tool_runtime.calls[0][-2]["content"])
            self.assertIn(
                "not private internal reasoning",
                tool_runtime.calls[0][-2]["content"],
            )

            class InvalidToolCallingRuntime:
                def generate(self, _messages):
                    return '<tool>run_shell {"command":"id"}</tool>'

            with self.assertRaisesRegex(ToolInvocationError, "unsupported local tool"):
                LocalAgent(InvalidToolCallingRuntime(), bus).reply(
                    [{"role": "user", "content": "Do something local."}]
                )
            self.assertIn(
                '"result": 42',
                agent.reply(
                    [
                        {
                            "role": "user",
                            "content": '/tool calculator {"expression":"6 * 7"}',
                        }
                    ]
                ),
            )
            self.assertIn(
                "[facts.txt - chunk 1]",
                agent.reply([{"role": "user", "content": "/search Saturn rings"}]),
            )
            self.assertIn(
                "[facts.txt - chunk 1]",
                agent.reply(
                    [
                        {
                            "role": "user",
                            "content": "Search my local notes for Saturn",
                        }
                    ]
                ),
            )
            self.assertIn(
                "[facts.txt - chunk 1]",
                agent.reply(
                    [
                        {
                            "role": "user",
                            "content": "What do my files say about Saturn?",
                        }
                    ]
                ),
            )
            self.assertIn(
                "Saturn has prominent rings.",
                agent.reply([{"role": "user", "content": "/digest Saturn rings"}]),
            )
            self.assertIn(
                "Saturn has prominent rings.",
                agent.reply(
                    [
                        {
                            "role": "user",
                            "content": "Summarize my documents about Saturn",
                        }
                    ]
                ),
            )
            self.assertEqual(
                agent.reply(
                    [{"role": "user", "content": "Solve x**2 - 5*x + 6 = 0"}]
                ),
                "Roots of x**2 - 5*x + 6 = 0: x = 2.0, x = 3.0",
            )
            self.assertIn(
                "1.0000 bits per token",
                agent.reply(
                    [{"role": "user", "content": "Shannon entropy of a b"}]
                ),
            )
            self.assertIn(
                "x = -2.0",
                agent.reply(
                    [{"role": "user", "content": "/solve 2*x + 4 = 0"}]
                ),
            )
            self.assertIn(
                "1.0000 bits per token",
                agent.reply([{"role": "user", "content": "/entropy a b"}]),
            )
            self.assertIn(
                "calculator",
                agent.reply([{"role": "user", "content": "What tools do you have?"}]),
            )
            self.assertIn(
                "offline",
                agent.reply(
                    [{"role": "user", "content": "Search the web for Saturn"}]
                ),
            )
            about = agent.reply(
                [{"role": "user", "content": "What is Mewtronic?"}]
            )
            self.assertIn("not conscious", about.lower())
            self.assertIn("Mewtronic", about)
            self.assertIn("facts.txt", about)
            self.assertIn(
                "I am Woa",
                agent.reply([{"role": "user", "content": "What's your name?"}]),
            )
            self.assertEqual(
                agent.reply([{"role": "user", "content": "Hello"}]),
                "model response",
            )
            self.assertEqual(runtime.calls, 1)
            with self.assertRaises(ToolInvocationError):
                agent.reply([{"role": "user", "content": "/calc 1 / 0"}])


class CliKnowledgeTests(unittest.TestCase):
    def test_make_synthetic_cli_creates_reviewable_sft_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "woa-synthetic.jsonl"
            with patch.object(
                sys,
                "argv",
                [
                    "woa",
                    "make-synthetic",
                    "--output",
                    str(output_path),
                    "--arithmetic-examples",
                    "2",
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli_main(), 0)

            records = load_chat_records([output_path])
            self.assertEqual(len(records), 12)
            self.assertIn("original SFT examples", output.getvalue())

    def test_start_seeds_bundled_sources_before_launching_client(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"WOA_DATA_DIR": directory}):
                with patch.object(sys, "argv", ["woa", "start"]):
                    with patch("woa.cli.seed_bundled_knowledge") as seed:
                        with patch(
                            "woa.cli.launch_application", return_value=0
                        ) as launch:
                            self.assertEqual(cli_main(), 0)

            self.assertIsInstance(seed.call_args.args[0], KnowledgeIndex)
            launch.assert_called_once()

    def test_digest_and_search_cli_share_the_local_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "app-data"
            document = root / "reference.md"
            document.write_text(
                "Local agents can use cited knowledge snippets.",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"WOA_DATA_DIR": str(data_dir)}):
                with patch.object(sys, "argv", ["woa", "digest", str(document)]):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(cli_main(), 0)
                    self.assertIn("Indexed 1 documents", output.getvalue())

                with patch.object(
                    sys, "argv", ["woa", "search", "cited", "knowledge"]
                ):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(cli_main(), 0)
                    self.assertIn(str(document), output.getvalue())
                    self.assertIn("cited knowledge snippets", output.getvalue())

    def test_doctor_reports_local_readiness_when_service_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"WOA_DATA_DIR": directory}):
                with patch(
                    "woa.cli.request_json",
                    side_effect=RuntimeError("Cannot connect to local service."),
                ):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        result = run_diagnostics()

        self.assertEqual(result, 1)
        report = output.getvalue()
        self.assertIn("Local SQLite/FTS5 store: ready", report)
        self.assertIn("Bundled knowledge: ready (8/8 files)", report)
        self.assertIn("Model readiness: No local model was found", report)
        self.assertIn("Local service: unavailable", report)
        self.assertIn("python -m woa start", report)


class LaunchTests(unittest.TestCase):
    def test_client_reports_slow_local_service_without_raw_timeout_traceback(self):
        with patch(
            "woa.client.urllib.request.urlopen", side_effect=TimeoutError
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "timed out after 15 seconds"):
                request_json("/api/health")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 15)

    def test_start_creates_service_and_opens_web_client(self):
        server = Mock()
        with patch.dict(os.environ, {"WOA_PORT": "8765"}):
            with patch(
                "woa.cli.request_json",
                side_effect=RuntimeError("Cannot connect to http://127.0.0.1:8765."),
            ):
                with patch("woa.cli.create_server", return_value=server) as create:
                    with patch("woa.cli.start_server_in_thread") as start:
                        with patch("woa.cli.webbrowser.open", return_value=True) as open_web:
                            with patch(
                                "woa.cli.time.sleep", side_effect=KeyboardInterrupt
                            ):
                                with contextlib.redirect_stdout(io.StringIO()):
                                    result = launch_application(
                                        "web", "127.0.0.1", 8765
                                    )

        self.assertEqual(result, 0)
        create.assert_called_once_with(host="127.0.0.1", port=8765)
        start.assert_called_once_with(server)
        open_web.assert_called_once_with("http://127.0.0.1:8765")
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()

    def test_start_reuses_existing_service_and_doctor_reports_integrations(self):
        health = {
            "ok": True,
            "model_kind": "project_trained_transformer",
            "model_available": True,
            "hardware": {"device": "cpu"},
            "features": [
                "shared_conversations",
                "tool_bus",
                "knowledge_index",
                "serialized_turns",
                "mewtronic_environment",
            ],
        }

        def request(path):
            if path == "/api/health":
                return health
            if path == "/api/tools":
                return {"tools": [{"name": "calculator"}, {"name": "knowledge_search"}]}
            if path == "/api/knowledge/documents":
                return {"documents": [{"source": "notes.md"}]}
            raise AssertionError(f"Unexpected endpoint: {path}")

        with patch.dict(os.environ, {"WOA_PORT": "8765"}):
            with patch("woa.cli.request_json", side_effect=request):
                with patch("woa.cli.create_server") as create:
                    with patch("woa.cli.webbrowser.open", return_value=True):
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(
                                launch_application("web", "127.0.0.1", 8765),
                                0,
                            )
                        create.assert_not_called()
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(run_diagnostics(), 0)
                    self.assertIn("calculator, knowledge_search", output.getvalue())
                    self.assertIn("Indexed knowledge documents: 1", output.getvalue())


class DesktopTests(unittest.TestCase):
    def test_health_message_is_shown_when_model_is_unavailable(self):
        client = DesktopClient.__new__(DesktopClient)
        client.model_status = Mock()
        with patch(
            "woa.desktop.request_json",
            return_value={
                "model_available": False,
                "model_message": "Transformers is required.",
                "hardware": {"accelerator": None, "device": "cpu"},
            },
        ):
            client.refresh_health()

        client.model_status.config.assert_called_once_with(
            text="Transformers is required."
        )

    def test_health_message_falls_back_when_model_message_is_missing(self):
        client = DesktopClient.__new__(DesktopClient)
        client.model_status = Mock()
        with patch(
            "woa.desktop.request_json",
            return_value={
                "model_available": False,
                "hardware": {"accelerator": None, "device": "cpu"},
            },
        ):
            client.refresh_health()

        client.model_status.config.assert_called_once_with(
            text="No local model found; general chat needs a model."
        )


class FoundationTests(unittest.TestCase):
    def test_world_bank_records_are_rendered_as_attributed_text(self):
        response = json.dumps(
            [
                {"lastupdated": "2026-01-01"},
                [
                    {
                        "country": {"value": "Exampleland"},
                        "countryiso3code": "EXM",
                        "date": "2025",
                        "value": 42.5,
                    },
                    {"country": {"value": "No data"}, "value": None},
                ],
            ]
        )
        with patch(
            "woa.corpus.fetch_public_text",
            return_value=(response, {"host": "api.worldbank.org"}),
        ):
            text, _metadata = _world_bank_text("NE.TRD.GNFS.ZS", "Trade (% of GDP)")

        self.assertIn("Exampleland (EXM), 2025", text)
        self.assertNotIn("No data", text)
        self.assertIn("World Bank Indicators API", text)

    def test_nasa_abstract_import_filters_non_public_records(self):
        response = json.dumps(
            {
                "results": [
                    {
                        "id": 123,
                        "title": "Public report",
                        "abstract": "Public abstract.",
                        "distribution": "PUBLIC",
                        "copyright": {
                            "determinationType": "GOV_PUBLIC_USE_PERMITTED",
                            "containsThirdPartyMaterial": False,
                        },
                    },
                    {
                        "id": 456,
                        "title": "Restricted report",
                        "abstract": "Do not include.",
                        "distribution": "LIMITED",
                        "copyright": {
                            "determinationType": "GOV_PUBLIC_USE_PERMITTED",
                            "containsThirdPartyMaterial": False,
                        },
                    },
                ]
            }
        )
        with patch(
            "woa.corpus.fetch_public_text",
            return_value=(response, {"host": "ntrs.nasa.gov"}),
        ):
            text, metadata = _nasa_abstract_text("aerospace")

        self.assertIn("Public abstract.", text)
        self.assertNotIn("Do not include.", text)
        self.assertEqual(metadata["included_records"], 1)

    def test_curated_corpus_download_writes_rights_and_hash_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "woa.corpus.fetch_public_text",
                return_value=("A public-domain text.", {"host": "www.gutenberg.org"}),
            ):
                records = download_starter_corpus(
                    directory,
                    categories=["scripture"],
                )
            target = Path(directory) / "bible-kjv-gutenberg-10.txt"
            manifest = json.loads(
                (Path(directory) / "sources.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(records), 1)
            self.assertTrue(target.is_file())
            self.assertIn("Project Gutenberg", manifest["sources"][0]["rights"])

    def test_curated_philosophy_corpus_records_historical_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "woa.corpus.fetch_public_text",
                return_value=("A public-domain philosophical text.", {"host": "www.gutenberg.org"}),
            ) as fetch:
                records = download_starter_corpus(
                    directory,
                    categories=["philosophy"],
                )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["category"], "philosophy")
            self.assertIn("historical", records[0]["scope_note"])
            self.assertEqual(fetch.call_args.args[1], ["www.gutenberg.org"])
            self.assertTrue(Path(records[0]["path"]).is_file())

    def test_byte_tokenizer_round_trips_utf8_and_roles(self):
        tokenizer = ByteTokenizer()
        text = "Hello, 世界 🌍"

        encoded = tokenizer.encode(text)

        self.assertEqual(encoded[0], tokenizer.BOS)
        self.assertEqual(encoded[-1], tokenizer.EOS)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertEqual(
            tokenizer.encode_message("assistant", text)[0], tokenizer.ASSISTANT
        )

    def test_model_config_rejects_invalid_attention_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(d_model=15, n_heads=4)
        with self.assertRaisesRegex(ValueError, "n_heads"):
            ModelConfig(n_heads=0)

    def test_inference_prompt_keeps_recent_turns_and_fits_model_context(self):
        from woa.training import _build_inference_prompt

        tokenizer = ByteTokenizer()
        prompt = _build_inference_prompt(
            [
                {"role": "user", "content": "old question " * 12},
                {"role": "assistant", "content": "old answer " * 12},
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
                {"role": "user", "content": "current request"},
            ],
            context_length=128,
        )
        decoded = tokenizer.decode(prompt)

        self.assertLessEqual(len(prompt), 128)
        self.assertIn("You are Woa", decoded)
        self.assertIn("recent answer", decoded)
        self.assertIn("current request", decoded)
        self.assertNotIn("old question", decoded)

    def test_inference_prompt_guides_planning_tasks_and_completion_checks(self):
        from woa.training import _build_inference_prompt

        prompt = _build_inference_prompt(
            [{"role": "user", "content": "Plan a multi-step project."}],
            context_length=512,
        )

        decoded = ByteTokenizer().decode(prompt)
        self.assertIn("dependencies", decoded)
        self.assertIn("checkpoints", decoded)
        self.assertIn("success checks", decoded)

    def test_inference_prompt_truncates_long_unicode_user_input_safely(self):
        from woa.training import _build_inference_prompt

        tokenizer = ByteTokenizer()
        prompt = _build_inference_prompt(
            [{"role": "user", "content": "🌍" * 100}],
            context_length=64,
        )

        self.assertLessEqual(len(prompt), 64)
        self.assertNotIn("\ufffd", tokenizer.decode(prompt))
        self.assertGreater(tokenizer.decode(prompt).count("🌍"), 0)

    def test_batch_size_and_cosine_schedule_are_bounded(self):
        self.assertEqual(
            recommended_batch_size({"device": "cuda", "total_memory_bytes": 8 * 2**30}),
            4,
        )
        self.assertEqual(
            recommended_batch_size({"device": "cpu"}, requested=3),
            3,
        )
        self.assertAlmostEqual(learning_rate(0, 10, 1.0, 2), 0.5)
        self.assertEqual(learning_rate(10, 10, 1.0, 2), 0.0)
        self.assertEqual(learning_rate(9, 10, 1.0, 2), 0.0)

    def test_hardware_recommendations_use_detected_vram(self):
        settings = recommended_model_settings(
            {"device": "cuda", "total_memory_bytes": 24 * 1024**3}
        )
        self.assertEqual(settings["d_model"], 256)
        self.assertEqual(settings["n_layers"], 8)

    def test_corpus_loader_reads_local_text_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            first = source / "first.txt"
            duplicate = source / "duplicate.md"
            first.write_text("A small training corpus.", encoding="utf-8")
            duplicate.write_text("A small training corpus.", encoding="utf-8")

            self.assertEqual(load_text_corpus([source]), "A small training corpus.")

    def test_corpus_loader_reports_non_utf8_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_bytes(b"\xff\xfe")

            with self.assertRaisesRegex(ValueError, "UTF-8"):
                load_text_corpus([corpus])

    def test_sft_dataset_masks_prompt_and_trains_assistant_answer(self):
        tokenizer = ByteTokenizer()
        examples = build_sft_examples(
            [
                [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            ],
            context_length=64,
            tokenizer=tokenizer,
        )

        example = examples[0]
        self.assertTrue(all(target == -100 for target in example["targets"][:-7]))
        self.assertEqual(
            example["targets"][-7:],
            [*b"Answer", tokenizer.EOS],
        )

    def test_chat_jsonl_loader_rejects_malformed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.jsonl"
            data.write_text('{"messages": []}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "messages array"):
                load_chat_records([data])

    def test_preference_jsonl_builds_response_only_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "preferences.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "Question"}],
                        "chosen": "Helpful",
                        "rejected": "Unhelpful",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_preference_records([data])
            examples = build_preference_examples(records, context_length=64)

            self.assertEqual(len(examples), 1)
            self.assertTrue(any(examples[0]["chosen"]["response_mask"]))
            self.assertTrue(any(examples[0]["rejected"]["response_mask"]))

    def test_synthetic_sft_data_is_reproducible_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first_result = create_synthetic_sft_dataset(
                first,
                arithmetic_examples=6,
                seed=23,
            )
            second_result = create_synthetic_sft_dataset(
                second,
                arithmetic_examples=6,
                seed=23,
            )

            self.assertEqual(first_result["examples"], 16)
            self.assertEqual(second_result["examples"], 16)
            self.assertEqual(
                first.read_text(encoding="utf-8"),
                second.read_text(encoding="utf-8"),
            )
            self.assertTrue(
                any(
                    message["content"].startswith("<tool>calculator ")
                    for conversation in load_chat_records([first])
                    for message in conversation
                    if message["role"] == "assistant"
                )
            )
            with self.assertRaises(FileExistsError):
                create_synthetic_sft_dataset(first, arithmetic_examples=0)

    def test_remote_dataset_url_requires_exact_allowlist_and_public_dns(self):
        public_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        ]
        with patch("woa.sources.socket.getaddrinfo", return_value=public_result):
            parsed, hostname = validate_source_url(
                "https://data.example.org/corpus.jsonl",
                ["data.example.org"],
            )
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(hostname, "data.example.org")

        with self.assertRaisesRegex(ValueError, "allowlist"):
            validate_source_url(
                "https://other.example.org/data",
                ["data.example.org"],
            )
        private_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ]
        with patch("woa.sources.socket.getaddrinfo", return_value=private_result):
            with self.assertRaisesRegex(ValueError, "public IP"):
                validate_source_url(
                    "https://data.example.org/corpus",
                    ["data.example.org"],
                )

    def test_dataset_fetch_rejects_html_pages(self):
        headers = Message()
        headers["Content-Type"] = "text/html"
        response = MagicMock()
        response.headers = headers
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        public_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        ]
        with (
            patch("woa.sources.socket.getaddrinfo", return_value=public_result),
            patch("woa.sources.urllib.request.build_opener", return_value=opener),
        ):
            with self.assertRaisesRegex(ValueError, "content type"):
                fetch_public_text(
                    "https://data.example.org/page",
                    ["data.example.org"],
                )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch training extra not installed")
class TransformerTests(unittest.TestCase):
    def test_local_pretrained_runtime_uses_offline_chat_template(self):
        import types
        import torch

        from woa.training import ModelRuntime

        class FakeBatchEncoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids

            def __getitem__(self, key):
                return getattr(self, key)

        class FakeTokenizer:
            chat_template = "test-template"
            pad_token_id = 0
            eos_token_id = 1
            load_options = None
            messages = None
            template_options = None

            @classmethod
            def from_pretrained(cls, _path, **options):
                cls.load_options = options
                return cls()

            def apply_chat_template(self, messages, **options):
                type(self).messages = messages
                type(self).template_options = options
                return FakeBatchEncoding(torch.tensor([[2, 3, 4]]))

            def decode(self, _tokens, **_options):
                return "Hello! How can I help?"

        class FakeModel:
            config = type("Config", (), {"max_position_embeddings": 1024})()
            load_options = None
            generation_options = None

            @classmethod
            def from_pretrained(cls, _path, **options):
                cls.load_options = options
                return cls()

            def to(self, _device):
                return self

            def eval(self):
                return self

            def generate(self, input_ids, **options):
                type(self).generation_options = options
                new_tokens = torch.tensor([[8, 9]], device=input_ids.device)
                return torch.cat((input_ids, new_tokens), dim=1)

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = FakeTokenizer
        transformers.AutoModelForCausalLM = FakeModel

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}", encoding="utf-8")
            Path(directory, "model.safetensors").write_bytes(b"test weights")
            Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")
            runtime = ModelRuntime(directory)
            with patch.dict(sys.modules, {"transformers": transformers}):
                self.assertTrue(runtime.available)
                self.assertEqual(runtime.backend, "local_pretrained")
                reply = runtime.generate(
                    [
                        {"role": "system", "content": "Be clear."},
                        {"role": "user", "content": "Hello"},
                    ]
                )

        self.assertEqual(reply, "Hello! How can I help?")
        self.assertEqual(FakeTokenizer.load_options["local_files_only"], True)
        self.assertEqual(FakeModel.load_options["local_files_only"], True)
        self.assertEqual(FakeTokenizer.messages[-1]["content"], "Hello")
        self.assertTrue(FakeTokenizer().template_options["add_generation_prompt"])
        self.assertTrue(FakeModel().generation_options["do_sample"])
        self.assertEqual(FakeModel.generation_options["max_new_tokens"], 256)

    def test_local_pretrained_runtime_reports_missing_model_artifacts(self):
        from woa.training import ModelRuntime

        with tempfile.TemporaryDirectory() as directory:
            runtime = ModelRuntime(directory)

            self.assertFalse(runtime.available)
            self.assertIn("config.json", runtime.readiness_message)
            self.assertIn("model weights", runtime.readiness_message)
            self.assertIn("tokenizer files", runtime.readiness_message)

    def test_local_pretrained_runtime_reports_missing_optional_dependency(self):
        from woa.training import ModelRuntime

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}", encoding="utf-8")
            Path(directory, "model.safetensors").write_bytes(b"test weights")
            Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")
            runtime = ModelRuntime(directory)

            def available_torch_only(module_name):
                return object() if module_name == "torch" else None

            with (
                patch.dict(sys.modules, {"transformers": None}),
                patch(
                    "woa.training.importlib.util.find_spec",
                    side_effect=available_torch_only,
                ),
            ):
                self.assertFalse(runtime.available)
                self.assertIn("Transformers is required", runtime.readiness_message)

    def test_model_runtime_reports_missing_torch_dependency(self):
        from woa.training import ModelRuntime

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "model.pt")
            checkpoint.write_bytes(b"test checkpoint")
            runtime = ModelRuntime(checkpoint)

            with (
                patch.dict(sys.modules, {"torch": None}),
                patch("woa.training.importlib.util.find_spec", return_value=None),
            ):
                self.assertFalse(runtime.available)
                self.assertIn("PyTorch is required", runtime.readiness_message)

    def test_exact_million_parameter_architecture_uses_learned_positions(self):
        import torch

        from woa.model import build_model

        config = ModelConfig(
            context_length=204,
            d_model=100,
            n_heads=5,
            n_layers=7,
            learned_position_embeddings=True,
        )
        model = build_model(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(parameter_count, 1_000_000)
        logits = model(torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(logits.shape), (1, 3, config.vocab_size))

    def test_causal_forward_backward_and_generation(self):
        import torch

        from woa.model import build_model, register_architecture

        torch.manual_seed(1)
        model = build_model(
            ModelConfig(context_length=16, d_model=16, n_heads=2, n_layers=1)
        )
        model.eval()
        first = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 3, 9]])
        original_logits = model(first)
        future_logits = model(changed_future)
        self.assertTrue(torch.allclose(original_logits[:, :3], future_logits[:, :3]))

        _, loss = model(first, torch.tensor([[2, 3, 4, 5]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.generate(first, max_new_tokens=2).shape[1], 6)

    def test_custom_architecture_has_a_registration_extension_point(self):
        from woa.model import build_model, register_architecture

        config = ModelConfig(architecture="test_architecture")
        register_architecture("test_architecture", lambda supplied: supplied)

        self.assertIs(build_model(config), config)

    def test_training_writes_loadable_project_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            checkpoint = Path(directory) / "model.pt"
            corpus.write_text(
                "A tiny corpus for validating a model trained from scratch. " * 8,
                encoding="utf-8",
            )
            train(
                [corpus],
                checkpoint,
                steps=1,
                batch_size=1,
                warmup_steps=1,
                config=ModelConfig(
                    context_length=16, d_model=16, n_heads=2, n_layers=1
                ),
            )
            runtime = ModelRuntime(checkpoint)
            self.assertTrue(runtime.available)
            self.assertIsInstance(runtime.generate([], max_new_tokens=2), str)
            first_mtime = runtime._loaded_mtime
            train(
                [corpus],
                checkpoint,
                steps=1,
                batch_size=1,
                warmup_steps=1,
                resume_from=checkpoint,
            )

            self.assertIsInstance(runtime.generate([], max_new_tokens=2), str)
            self.assertNotEqual(runtime._loaded_mtime, first_mtime)
            import torch

            payload = torch.load(checkpoint, weights_only=True)
            self.assertEqual(payload["training"]["completed_steps"], 2)
            self.assertNotIn("optimizer", payload)
            self.assertTrue(
                checkpoint.with_name(
                    f"{checkpoint.name}.{payload['checkpoint_id']}.optimizer.pt"
                ).is_file()
            )

    def test_sft_finetunes_base_checkpoint_and_generates(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text(
                "A tiny corpus for validating a model trained from scratch. " * 8,
                encoding="utf-8",
            )
            config = ModelConfig(
                context_length=32, d_model=16, n_heads=2, n_layers=1
            )
            base = Path(directory) / "base.pt"
            fine_tuned = Path(directory) / "sft.pt"
            train(
                [corpus],
                base,
                steps=1,
                batch_size=1,
                warmup_steps=1,
                config=config,
            )
            metrics = evaluate_checkpoint(base, [corpus])
            self.assertEqual(metrics["tokens"], len(corpus.read_bytes()) + 1)
            self.assertGreater(metrics["perplexity"], 0)
            examples = build_sft_examples(
                [
                    [
                        {"role": "user", "content": "What is 2 + 2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                ],
                config.context_length,
            )
            train(
                [],
                fine_tuned,
                steps=1,
                batch_size=1,
                warmup_steps=1,
                config=config,
                examples=examples,
                initialize_from=base,
            )

            runtime = ModelRuntime(fine_tuned)
            self.assertTrue(runtime.generate([], max_new_tokens=2))

    def test_dpo_trains_from_a_frozen_reference_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text(
                "A tiny corpus for validating a model trained from scratch. " * 8,
                encoding="utf-8",
            )
            config = ModelConfig(
                context_length=32, d_model=16, n_heads=2, n_layers=1
            )
            reference = Path(directory) / "reference.pt"
            aligned = Path(directory) / "aligned.pt"
            train(
                [corpus],
                reference,
                steps=1,
                batch_size=1,
                warmup_steps=1,
                config=config,
            )
            records = [
                {
                    "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                    "chosen": "4",
                    "rejected": "5",
                }
            ]
            examples = build_preference_examples(records, config.context_length)
            train_dpo(
                examples,
                reference,
                aligned,
                steps=1,
                batch_size=1,
                checkpoint_interval=1,
            )
            train_dpo(
                examples,
                reference,
                aligned,
                steps=1,
                batch_size=1,
                checkpoint_interval=1,
                resume_from=aligned,
            )

            self.assertTrue(ModelRuntime(aligned).generate([], max_new_tokens=2))


class ApiTests(unittest.TestCase):
    def test_chat_api_serializes_concurrent_turns_per_conversation(self):
        first_started = threading.Event()
        second_user_saved = threading.Event()
        release_first = threading.Event()

        class FakeRuntime:
            available = True

            def generate(self, messages):
                content = messages[-1]["content"]
                if content == "first":
                    first_started.set()
                    if not release_first.wait(5):
                        raise RuntimeError("Timed out waiting to finish the first turn.")
                return f"reply: {content}"

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "woa.db")
            add_message = store.add_message

            def observe_second_user(conversation_id, role, content):
                if role == "user" and content == "second":
                    second_user_saved.set()
                return add_message(conversation_id, role, content)

            store.add_message = observe_second_user
            server = create_server(
                host="127.0.0.1",
                port=0,
                store=store,
                runtime=FakeRuntime(),
            )
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            conversation = store.create_conversation("Concurrent turns")

            def send(content):
                request = urllib.request.Request(
                    f"{base_url}/api/conversations/{conversation['id']}/messages",
                    data=json.dumps({"content": content}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    return json.load(response)

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(send, "first")
                    self.assertTrue(first_started.wait(2))
                    second = executor.submit(send, "second")
                    try:
                        self.assertFalse(second_user_saved.wait(0.2))
                    finally:
                        release_first.set()
                    first.result(timeout=5)
                    second.result(timeout=5)
                messages = store.list_messages(conversation["id"])
                self.assertEqual(
                    [message["content"] for message in messages],
                    ["first", "reply: first", "second", "reply: second"],
                )
                with urllib.request.urlopen(
                    f"{base_url}/api/conversations/{conversation['id']}/messages"
                ) as response:
                    synced_messages = json.load(response)["messages"]
                self.assertEqual(synced_messages, messages)
            finally:
                release_first.set()
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2)

    def test_local_knowledge_and_tools_work_through_chat_api(self):
        class FakeRuntime:
            available = True

            def generate(self, _messages):
                return "A locally generated answer."

        with tempfile.TemporaryDirectory() as directory:
            server = create_server(
                host="127.0.0.1",
                port=0,
                store=Store(Path(directory) / "woa.db"),
                runtime=FakeRuntime(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"

            def post(path, payload):
                request = urllib.request.Request(
                    base_url + path,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    return json.load(response)

            try:
                with urllib.request.urlopen(base_url + "/api/tools") as response:
                    tools = json.load(response)["tools"]
                self.assertEqual(
                    {tool["name"] for tool in tools},
                    {
                        "calculator",
                        "algebraic_solve",
                        "information_entropy",
                        "knowledge_search",
                        "knowledge_digest",
                        "local_environment",
                        "communication_cues",
                    },
                )

                uploaded = post(
                    "/api/knowledge/documents",
                    {
                        "name": "facts.md",
                        "content": "The local library documents orbital mechanics.",
                    },
                )
                self.assertEqual(uploaded["document"]["chunks"], 1)
                search = post(
                    "/api/tools/knowledge_search",
                    {"arguments": {"query": "orbital mechanics"}},
                )
                self.assertEqual(
                    search["result"]["results"][0]["source"],
                    "upload/facts.md",
                )

                conversation = post(
                    "/api/conversations", {"title": "Tools"}
                )["conversation"]
                result = post(
                    f"/api/conversations/{conversation['id']}/messages",
                    {"content": "/calc 6 * 7"},
                )
                self.assertEqual(result["messages"][1]["content"], "42")
                result = post(
                    f"/api/conversations/{conversation['id']}/messages",
                    {"content": "Please calculate 6 times 7."},
                )
                self.assertEqual(result["messages"][1]["content"], "42")
                result = post(
                    f"/api/conversations/{conversation['id']}/messages",
                    {"content": "Search my files for orbital mechanics"},
                )
                self.assertIn(
                    "[upload/facts.md - chunk 1]",
                    result["messages"][1]["content"],
                )
                result = post(
                    f"/api/conversations/{conversation['id']}/messages",
                    {"content": "Summarize my documents about orbital mechanics"},
                )
                self.assertIn(
                    "orbital mechanics",
                    result["messages"][1]["content"],
                )

                invalid_tool_request = urllib.request.Request(
                    f"{base_url}/api/conversations/{conversation['id']}/messages",
                    data=json.dumps({"content": "/calc 1 / 0"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(invalid_tool_request)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_chat_api_saves_project_runtime_reply(self):
        class FakeRuntime:
            available = True
            calls = 0

            def generate(self, _messages):
                self.calls += 1
                return "A locally generated answer."

        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            server = create_server(
                host="127.0.0.1",
                port=0,
                store=Store(Path(directory) / "woa.db"),
                runtime=runtime,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                request = urllib.request.Request(
                    base_url + "/api/conversations",
                    data=json.dumps({"title": "Test"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    conversation_id = json.load(response)["conversation"]["id"]
                request = urllib.request.Request(
                    f"{base_url}/api/conversations/{conversation_id}/messages",
                    data=json.dumps({"content": "Hello"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    messages = json.load(response)["messages"]
                self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
                self.assertEqual(messages[1]["content"], "A locally generated answer.")
                request = urllib.request.Request(
                    f"{base_url}/api/conversations/{conversation_id}/messages",
                    data=json.dumps({"content": "What is Mewtronic?"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    messages = json.load(response)["messages"]
                self.assertIn("not conscious", messages[1]["content"].lower())
                self.assertEqual(runtime.calls, 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_web_health_and_offline_chat_status(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "woa.db")
            store.record_training_event("training_step", {"step": 2, "loss": 0.5})
            with self.assertRaisesRegex(ValueError, "127.0.0.1"):
                create_server(host="0.0.0.0", port=0, store=store)
            with self.assertRaisesRegex(ValueError, "127.0.0.1"):
                create_server(host="::1", port=0, store=store)
            server = create_server(
                host="127.0.0.1",
                port=0,
                store=store,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base_url) as response:
                    page = response.read()
                    self.assertTrue(
                        response.headers["Content-Type"].startswith("text/html")
                    )
                    self.assertIn(b"<title>Woa</title>", page)
                    self.assertIn(b"empty-state", page)
                    self.assertIn(b"Enter to send", page)
                    self.assertIn(b"Shift+Enter", page)
                    self.assertIn(b'<body class="no-effects large-chat-text">', page)
                    self.assertIn(b"effects: false", page)
                    self.assertIn(b"largeText: true", page)
                    for control in (
                        b'id="new-chat"',
                        b'id="knowledge-files"',
                        b'id="effects-toggle"',
                        b'id="compact-toggle"',
                        b'id="large-text-toggle"',
                        b'data-command="/calc "',
                        b'data-command="/search "',
                        b'data-command="/digest "',
                        b'data-command="/help"',
                        b'id="send" type="submit"',
                    ):
                        with self.subTest(control=control):
                            self.assertIn(control, page)
                    self.assertIn(
                        b"General chat needs a local model.",
                        page,
                    )
                    self.assertIn(b"LOCAL TOOLS ONLY", page)
                    self.assertIn(b"No system commands", page)
                    self.assertEqual(
                        re.findall(rb'data-command="([^"]*)"', page),
                        [b"/calc ", b"/search ", b"/digest ", b"/help"],
                    )
                compressed_request = urllib.request.Request(
                    base_url, headers={"Accept-Encoding": "gzip"}
                )
                with urllib.request.urlopen(compressed_request) as response:
                    self.assertEqual(response.headers["Content-Encoding"], "gzip")
                    self.assertIn(b"<title>Woa</title>", gzip.decompress(response.read()))
                with urllib.request.urlopen(base_url + "/api/health") as response:
                    self.assertTrue(
                        response.headers["Content-Type"].startswith(
                            "application/json"
                        )
                    )
                    health = json.load(response)
                self.assertTrue(health["ok"])
                self.assertFalse(health["model_available"])
                self.assertIn("Set WOA_MODEL_PATH", health["model_message"])
                self.assertEqual(health["model_backend"], "project_owned")
                self.assertEqual(health["model_kind"], "project_trained_transformer")
                self.assertIn("device", health["hardware"])
                with urllib.request.urlopen(base_url + "/api/training/events") as response:
                    events = json.load(response)["events"]
                self.assertEqual(events[0]["event_type"], "training_step")

                cross_origin_request = urllib.request.Request(
                    base_url + "/api/conversations",
                    data=json.dumps({"title": "Blocked"}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://attacker.example",
                    },
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(cross_origin_request)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()

                create_request = urllib.request.Request(
                    base_url + "/api/conversations",
                    data=json.dumps({"title": "New conversation"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(create_request) as response:
                    conversation_id = json.load(response)["conversation"]["id"]

                send_request = urllib.request.Request(
                    f"{base_url}/api/conversations/{conversation_id}/messages",
                    data=json.dumps({"content": "Hello"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(send_request)
                self.assertEqual(error.exception.code, 503)
                error_body = error.exception.read()
                self.assertIn(b"local instruction/chat model directory", error_body)
                self.assertIn(b"train one", error_body)
                error.exception.close()

                with urllib.request.urlopen(
                    f"{base_url}/api/conversations/{conversation_id}/messages"
                ) as response:
                    messages = json.load(response)["messages"]
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0]["content"], "Hello")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
