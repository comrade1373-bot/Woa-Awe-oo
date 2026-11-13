import json
import re


class ToolInvocationError(ValueError):
    pass


_DOCUMENTS = r"(?:my\s+)?(?:local\s+)?(?:documents|files|notes|library|knowledge)"
_SEARCH_PATTERNS = (
    re.compile(
        rf"^(?:please\s+)?(?:search|find|look\s+up)\s+(?:the\s+)?{_DOCUMENTS}"
        r"(?:\s+(?:for|about|on))?\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:please\s+)?(?:find|search|look\s+up)\s+(.+?)\s+in\s+{_DOCUMENTS}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:search|look\s+up)\s+(?:for\s+)?(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:what|which|where)\s+(?:do|does)\s+{_DOCUMENTS}\s+"
        r"(?:say|contain)\s+(?:anything\s+)?(?:about\s+|on\s+)?(.+)$",
        re.IGNORECASE,
    ),
)
_DIGEST_PATTERNS = (
    re.compile(
        rf"^(?:please\s+)?(?:digest|summari[sz]e)\s+{_DOCUMENTS}"
        r"(?:\s+(?:about|on|for)\s+(.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:please\s+)?(?:digest|summari[sz]e)\s+(.+?)\s+"
        rf"(?:from|in)\s+{_DOCUMENTS}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:please\s+)?(?:summari[sz]e)\s+(?:what\s+)?{_DOCUMENTS}"
        r"\s+(?:say|contain)\s+(?:about\s+)?(.+)$",
        re.IGNORECASE,
    ),
)
_MATH_PREFIX = re.compile(
    r"^(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?"
    r"(?:calc(?:ulate)?|compute|solve|evaluate|work\s+out|what\s+is|"
    r"what's|how\s+much\s+is)\s+",
    re.IGNORECASE,
)
_MATH_OPERATORS = (
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.IGNORECASE), "**"),
    (re.compile(r"\bmultiplied\s+by\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bover\b", re.IGNORECASE), "/"),
    (re.compile(r"\bmod(?:ulo)?\b", re.IGNORECASE), "%"),
)
_PURE_MATH = re.compile(r"[\d\s.+\-*/%()]+\Z")
_HELP = re.compile(
    r"^(?:/?(?:help|commands|tools)|what can (?:you|this assistant) "
    r"(?:do(?: for me)?|help(?: me)?(?: with(?: .*)?)?)"
    r"|what tools do you have|show (?:me )?(?:the )?tools|list (?:the )?commands)[?.!]*$",
    re.IGNORECASE,
)
_SELF_STATUS = re.compile(
    r"^(?:who are you|what are you|what is mewtronic|what does mewtronic mean|"
    r"are you (?:self[- ]aware|conscious)|what do you know about your environment|"
    r"what resources are available locally|what can you do)[?.!]*$",
    re.IGNORECASE,
)
_SOLVE_POLYNOMIAL = re.compile(
    r"^(?:please\s+)?(?:solve|find\s+the\s+roots\s+of)\s+(.+)$",
    re.IGNORECASE,
)
_ENTROPY_QUERY = re.compile(
    r"^(?:please\s+)?(?:(?:calculate|compute|measure)\s+)?(?:the\s+)?"
    r"(?:shannon|information)\s+entropy(?:\s+of)?\s+(.+)$",
    re.IGNORECASE,
)
_WEB_SEARCH = re.compile(
    r"^(?:please\s+)?(?:search|look\s+up)\s+(?:the\s+)?(?:web|internet|online)\b",
    re.IGNORECASE,
)
_MODEL_TOOL_CALL = re.compile(
    r"\A\s*<tool>([a-z_]{1,64})\s+([\s\S]*?)</tool>\s*\Z"
)
_MODEL_TOOL_ALLOWLIST = frozenset(
    {
        "algebraic_solve",
        "calculator",
        "communication_cues",
        "information_entropy",
        "knowledge_digest",
        "knowledge_search",
        "local_environment",
    }
)
_MODEL_TOOL_GUIDANCE = (
    'Use at most one needed local tool: <tool>name {"key":"value"}</tool>. '
    "calculator=expression; knowledge_search/knowledge_digest=query; "
    "algebraic_solve=equation; information_entropy/communication_cues=text; "
    "local_environment={}. No shell."
)


def _clean_query(query):
    return query.strip().strip("\"' \t\r\n.,!?")


def _search_query(text):
    for pattern in _SEARCH_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return _clean_query(match.group(1))
    return None


def _digest_query(text):
    for pattern in _DIGEST_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return _clean_query(match.group(1) or "")
    return None


def _calculation_expression(text):
    expression = text.strip().rstrip("?.!")
    prefix_match = _MATH_PREFIX.match(expression)
    if prefix_match:
        expression = expression[prefix_match.end() :]
    for pattern, replacement in _MATH_OPERATORS:
        expression = pattern.sub(replacement, expression)
    expression = re.sub(
        r"\s+(?:please|thanks|thank you)$", "", expression, flags=re.IGNORECASE
    )
    if not re.search(r"\d", expression) or not re.search(r"[+\-*/%]", expression):
        return None
    if not prefix_match and not _PURE_MATH.fullmatch(expression.strip()):
        return None
    return expression.strip()


class LocalAgent:
    def __init__(self, runtime, tool_bus):
        self.runtime = runtime
        self.tool_bus = tool_bus

    def _help(self):
        tools = "\n".join(
            f"{tool['name']} - {tool['description']}"
            for tool in self.tool_bus.describe()
        )
        return (
            "You can ask naturally, or use /calc EXPR, /search QUERY, "
            "/digest QUERY, /solve EQUATION, /entropy TEXT, /about, "
            "/cues TEXT, or /tool NAME JSON. Local tools:\n"
            + tools
        )

    def _run_tool(self, name, arguments):
        result = self.tool_bus.invoke(name, arguments)
        if name == "calculator":
            return str(result["result"])
        if name == "knowledge_search":
            if not result["results"]:
                return "No matching excerpts found in the local knowledge index."
            return "\n\n".join(
                f"[{hit['source']} - chunk {hit['chunk']}]\n{hit['content']}"
                for hit in result["results"]
            )
        if name == "knowledge_digest":
            if not result["sentences"]:
                return "No matching sentences found in the local knowledge index."
            return "\n".join(
                f"[{item['source']} - chunk {item['chunk']}] {item['content']}"
                for item in result["sentences"]
            )
        if name == "algebraic_solve":
            roots = result["roots"]
            if roots == "all real numbers":
                return f"{result['equation']} is true for every real x."
            if not roots:
                return f"{result['equation']} has no roots."
            formatted = []
            for root in roots:
                if isinstance(root, dict):
                    formatted.append(
                        f"{root['real']} {'+' if root['imaginary'] >= 0 else '-'} "
                        f"{abs(root['imaginary'])}i"
                    )
                else:
                    formatted.append(str(root))
            return f"Roots of {result['equation']}: " + ", ".join(
                f"x = {root}" for root in formatted
            )
        if name == "information_entropy":
            return (
                f"Empirical Shannon entropy: "
                f"{result['character_entropy_bits_per_symbol']:.4f} bits per "
                f"character ({result['characters']} characters); "
                f"{result['token_entropy_bits_per_token']:.4f} bits per token "
                f"({result['tokens']} tokens). This describes the sample, "
                "not intelligence or consciousness."
            )
        if name == "local_environment":
            checkpoint = result["model_checkpoint_available"]
            checkpoint_status = (
                "available"
                if checkpoint is True
                else "not found"
                if checkpoint is False
                else "not reported"
            )
            sources = ", ".join(result["indexed_sources"]) or "none"
            additional = result["additional_sources_not_listed"]
            if additional:
                sources += f", and {additional} more"
            return (
                "I am Woa, a local AI assistant. I am not conscious or "
                "self-aware. “Mewtronic” is this project's name for composing "
                "the local model, bundled and explicitly indexed knowledge, "
                "and registered local tools; it does not mean I inspect the "
                "whole computer.\n"
                f"Model checkpoint: {checkpoint_status}. "
                f"Compute: {result['compute_device']} on {result['platform']}.\n"
                f"Indexed documents ({result['indexed_document_count']}): "
                f"{sources}.\n"
                f"Local tools: {', '.join(result['local_tools']) or 'none'}."
            )
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    def _run_slash_command(self, command, argument):
        if command in {"/tools", "/help", "/commands"}:
            return self._help()
        if command == "/about":
            return self._run_tool("local_environment", {})
        if command == "/cues":
            if not argument.strip():
                raise ValueError("Add text to label, for example: `/cues I'm not sure.`")
            return self._run_tool("communication_cues", {"text": argument})
        if command in {"/solve", "/algebra"}:
            return self._run_tool(
                "algebraic_solve", {"equation": argument.strip()}
            )
        if command == "/entropy":
            return self._run_tool("information_entropy", {"text": argument})
        if command == "/tool":
            parts = argument.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError('Use `/tool NAME {"argument":"value"}`.')
            try:
                arguments = json.loads(parts[1])
            except json.JSONDecodeError as error:
                raise ValueError("Tool arguments must be valid JSON.") from error
            result = self.tool_bus.invoke(parts[0], arguments)
            return json.dumps(result, ensure_ascii=False, sort_keys=True)

        aliases = {
            "/calc": "calculator",
            "/calculate": "calculator",
            "/calculator": "calculator",
            "/search": "knowledge_search",
            "/knowledge_search": "knowledge_search",
            "/digest": "knowledge_digest",
            "/knowledge_digest": "knowledge_digest",
        }
        name = aliases.get(command)
        if name is None:
            return "Unknown command. Use /help to list supported commands."
        if name == "calculator":
            expression = _calculation_expression(argument) or argument.strip()
            return self._run_tool(name, {"expression": expression})
        query = argument.strip()
        if name == "knowledge_search":
            query = re.sub(r"^for\s+", "", query, flags=re.IGNORECASE)
        if name == "knowledge_digest":
            query = re.sub(r"^(?:about|on)\s+", "", query, flags=re.IGNORECASE)
        return self._run_tool(name, {"query": query})

    def _run_natural_tool(self, content):
        if _SELF_STATUS.fullmatch(content):
            return self._run_tool("local_environment", {})
        if _HELP.fullmatch(content):
            return self._help()
        if _WEB_SEARCH.match(content):
            return (
                "This assistant is offline and can search only documents you "
                "have added to its local knowledge library."
            )

        match = _SOLVE_POLYNOMIAL.fullmatch(content)
        if match:
            return self._run_tool(
                "algebraic_solve", {"equation": match.group(1).strip()}
            )
        match = _ENTROPY_QUERY.fullmatch(content)
        if match:
            return self._run_tool(
                "information_entropy", {"text": match.group(1).strip()}
            )

        query = _digest_query(content)
        if query is not None:
            if not query:
                raise ToolInvocationError(
                    "Add a topic, for example: summarize my documents about logistics."
                )
            return self._run_tool("knowledge_digest", {"query": query})

        query = _search_query(content)
        if query is not None:
            if not query:
                raise ToolInvocationError(
                    "Add search terms, for example: search my files for logistics."
                )
            return self._run_tool("knowledge_search", {"query": query})

        expression = _calculation_expression(content)
        if expression is not None:
            return self._run_tool("calculator", {"expression": expression})
        return None

    def reply(self, messages):
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("The latest message must be from the user.")
        content = messages[-1].get("content", "").strip()
        if content.startswith("/"):
            parts = content.split(maxsplit=1)
            command = parts[0].lower()
            argument = parts[1] if len(parts) == 2 else ""
            try:
                return self._run_slash_command(command, argument)
            except ValueError as error:
                raise ToolInvocationError(str(error)) from error

        try:
            tool_reply = self._run_natural_tool(content)
        except ValueError as error:
            raise ToolInvocationError(str(error)) from error
        if tool_reply is not None:
            return tool_reply
        model_messages = [
            *messages[:-1],
            {"role": "system", "content": _MODEL_TOOL_GUIDANCE},
            messages[-1],
        ]
        response = self.runtime.generate(model_messages)
        tool_call = _MODEL_TOOL_CALL.fullmatch(response)
        if tool_call is None:
            return response
        tool_name, raw_arguments = tool_call.groups()
        if tool_name not in _MODEL_TOOL_ALLOWLIST:
            raise ToolInvocationError(
                f"The model requested unsupported local tool {tool_name!r}."
            )
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object.")
            return self._run_tool(tool_name, arguments)
        except (json.JSONDecodeError, ValueError) as error:
            raise ToolInvocationError(
                f"The model's {tool_name} request was rejected: {error}"
            ) from error
