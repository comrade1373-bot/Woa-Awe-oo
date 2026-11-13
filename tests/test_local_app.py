import importlib.util
import json
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from woa.core import Store
from woa.hardware import recommended_batch_size, recommended_model_settings
from woa.model import ModelConfig
from woa.server import create_server
from woa.sources import fetch_public_text, validate_source_url
from woa.tokenizer import ByteTokenizer
from woa.training import ModelRuntime, learning_rate, load_text_corpus, train


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

    def test_training_events_are_shared_and_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "woa.db")
            store.record_training_event("step", {"loss": 1.25, "step": 4})

            events = store.list_training_events()

            self.assertEqual(events[0]["event_type"], "step")
            self.assertEqual(events[0]["payload"], {"loss": 1.25, "step": 4})


class FoundationTests(unittest.TestCase):
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


class ApiTests(unittest.TestCase):
    def test_chat_api_saves_project_runtime_reply(self):
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
                    self.assertIn(b"<title>Woa-Awe-oo</title>", response.read())
                with urllib.request.urlopen(base_url + "/api/health") as response:
                    health = json.load(response)
                self.assertTrue(health["ok"])
                self.assertFalse(health["model_available"])
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
                self.assertIn(b"Train one", error.exception.read())
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
