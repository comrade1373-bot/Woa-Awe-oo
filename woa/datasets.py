import json
import random
from pathlib import Path

from .tokenizer import ByteTokenizer


def _jsonl_files(paths):
    files = []
    for path in paths:
        source = Path(path)
        if source.is_dir():
            files.extend(
                item for item in sorted(source.rglob("*.jsonl")) if item.is_file()
            )
        elif source.suffix.lower() == ".jsonl":
            files.append(source)
        else:
            raise ValueError(f"Instruction datasets must be .jsonl files: {source}")
    if not files:
        raise ValueError("No .jsonl instruction data files were found.")
    return files


def load_chat_records(paths):
    records = []
    for path in _jsonl_files(paths):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} line {line_number}."
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path} line {line_number} must be a JSON object.")
                messages = record.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"{path} line {line_number} needs a non-empty messages array."
                    )
                normalized = []
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError("Each message must be a JSON object.")
                    role = message.get("role")
                    content = message.get("content")
                    if role not in ("system", "user", "assistant"):
                        raise ValueError(f"Unsupported conversation role: {role!r}.")
                    if not isinstance(content, str) or not content:
                        raise ValueError("Message content must be a non-empty string.")
                    normalized.append({"role": role, "content": content})
                if not any(item["role"] == "assistant" for item in normalized):
                    raise ValueError(f"{path} line {line_number} has no assistant answer.")
                records.append(normalized)
    if not records:
        raise ValueError("Instruction dataset contains no conversation records.")
    return records


def create_synthetic_sft_dataset(output_path, *, arithmetic_examples=32, seed=1337):
    if (
        not isinstance(arithmetic_examples, int)
        or isinstance(arithmetic_examples, bool)
        or not 0 <= arithmetic_examples <= 10_000
    ):
        raise ValueError("arithmetic_examples must be an integer from 0 to 10000.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    def record(user, assistant):
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }

    records = [
        record(
            "What's your name?",
            "I'm Woa, a local assistant. I can help explain ideas, work through "
            "supported calculations, and search documents you choose to index.",
        ),
        record(
            "Can you browse the web or inspect my computer?",
            "No. I can only use the local tools and documents made available to "
            "this app. Web access is not part of chat, and I do not scan your files.",
        ),
        record(
            "Are you conscious?",
            "No. I'm software running as a local assistant; I don't have "
            "consciousness or personal experiences.",
        ),
        record(
            "I don't understand this yet. Can you help?",
            "Of course. Tell me which part is unclear, and we can break it into "
            "smaller steps. I can explain another way if the first one doesn't help.",
        ),
        record(
            "I think that answer may be wrong.",
            "Thanks for flagging it. I'll reconsider the claim, separate what I "
            "can verify from what I'm unsure about, and correct it if needed.",
        ),
        record(
            "What should I learn next?",
            "Let's choose a useful next step. What are you trying to accomplish, "
            "what do you already know, and how much time can you spend?",
        ),
        record(
            "Can you summarize my documents about logistics?",
            "<tool>knowledge_digest "
            '{"query":"logistics"}</tool>',
        ),
        record(
            "Find information about orbital mechanics in my local notes.",
            "<tool>knowledge_search "
            '{"query":"orbital mechanics"}</tool>',
        ),
        record(
            "Solve x**2 - 5*x + 6 = 0.",
            "<tool>algebraic_solve "
            '{"equation":"x**2 - 5*x + 6 = 0"}</tool>',
        ),
        record(
            "Measure the Shannon entropy of the text hello world.",
            "<tool>information_entropy "
            '{"text":"hello world"}</tool>',
        ),
    ]

    rng = random.Random(seed)
    operators = {"+": "plus", "-": "minus", "*": "times"}
    question_templates = (
        "What is {left} {word} {right}?",
        "Please calculate {expression}.",
        "Could you work out {expression} for me?",
    )
    operator_symbols = tuple(operators)
    for index in range(arithmetic_examples):
        symbol = operator_symbols[index % len(operator_symbols)]
        left = rng.randint(1, 25)
        right = rng.randint(1, 25)
        word = operators[symbol]
        expression = f"{left} {symbol} {right}"
        question = rng.choice(question_templates).format(
            left=left,
            word=word,
            right=right,
            expression=expression,
        )
        arguments = json.dumps(
            {"expression": expression},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        records.append(
            record(question, f"<tool>calculator {arguments}</tool>")
        )

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as destination:
        for item in records:
            destination.write(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return {"path": str(output_path), "examples": len(records)}


def build_sft_examples(records, context_length, tokenizer=None):
    if context_length < 2:
        raise ValueError("context_length must be at least two.")
    tokenizer = tokenizer or ByteTokenizer()
    examples = []
    for record_index, messages in enumerate(records):
        prefix = [tokenizer.BOS]
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                prefix.extend(
                    tokenizer.encode_message(message["role"], message["content"])
                )
                continue

            answer = message["content"].encode("utf-8")
            suffix = [tokenizer.ASSISTANT, *answer, tokenizer.EOS]
            max_sequence_length = context_length + 1
            if len(suffix) > max_sequence_length:
                raise ValueError(
                    f"Assistant answer {record_index + 1}:{message_index + 1} "
                    "is longer than the model context."
                )
            available_prefix = max_sequence_length - len(suffix)
            context = prefix[-available_prefix:] if available_prefix else []
            sequence = [*context, *suffix]
            inputs = sequence[:-1]
            targets = [-100] * len(inputs)
            answer_start = len(context) + 1
            answer_end = len(sequence) - 1
            for position in range(answer_start - 1, answer_end):
                targets[position] = sequence[position + 1]
            examples.append({"input_ids": inputs, "targets": targets})

            prefix.extend(
                tokenizer.encode_message("assistant", message["content"])
            )
    if not examples:
        raise ValueError("Instruction dataset produced no assistant training examples.")
    return examples


def load_preference_records(paths):
    records = []
    for path in _jsonl_files(paths):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} line {line_number}."
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path} line {line_number} must be a JSON object.")
                prompt = record.get("prompt")
                chosen = record.get("chosen")
                rejected = record.get("rejected")
                if not isinstance(prompt, list) or not prompt:
                    raise ValueError(
                        f"{path} line {line_number} needs a non-empty prompt array."
                    )
                if not isinstance(chosen, str) or not chosen:
                    raise ValueError("chosen must be a non-empty assistant response.")
                if not isinstance(rejected, str) or not rejected:
                    raise ValueError("rejected must be a non-empty assistant response.")
                normalized_prompt = []
                for message in prompt:
                    if not isinstance(message, dict):
                        raise ValueError("Each prompt message must be a JSON object.")
                    role = message.get("role")
                    content = message.get("content")
                    if role not in ("system", "user", "assistant"):
                        raise ValueError(f"Unsupported conversation role: {role!r}.")
                    if not isinstance(content, str) or not content:
                        raise ValueError("Prompt message content must be non-empty.")
                    normalized_prompt.append({"role": role, "content": content})
                records.append(
                    {
                        "prompt": normalized_prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
    if not records:
        raise ValueError("Preference dataset contains no records.")
    return records


def build_preference_examples(records, context_length, tokenizer=None):
    if context_length < 2:
        raise ValueError("context_length must be at least two.")
    tokenizer = tokenizer or ByteTokenizer()
    examples = []
    for record_index, record in enumerate(records, start=1):
        prompt = [tokenizer.BOS]
        for message in record["prompt"]:
            prompt.extend(
                tokenizer.encode_message(message["role"], message["content"])
            )
        responses = {}
        max_sequence_length = context_length + 1
        for label in ("chosen", "rejected"):
            completion = record[label].encode("utf-8")
            suffix = [tokenizer.ASSISTANT, *completion, tokenizer.EOS]
            if len(suffix) > max_sequence_length:
                raise ValueError(
                    f"{label} response in preference record {record_index} "
                    "is longer than the model context."
                )
            available_prefix = max_sequence_length - len(suffix)
            context = prompt[-available_prefix:] if available_prefix else []
            sequence = [*context, *suffix]
            targets = sequence[1:]
            response_mask = [False] * len(targets)
            first_response_target = len(context)
            for position in range(first_response_target, len(targets)):
                response_mask[position] = True
            responses[label] = {
                "input_ids": sequence[:-1],
                "targets": targets,
                "response_mask": response_mask,
            }
        examples.append(
            {
                "chosen": responses["chosen"],
                "rejected": responses["rejected"],
            }
        )
    return examples
