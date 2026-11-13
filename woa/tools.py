import ast
import cmath
import math
import operator
import platform
import re
from pathlib import Path

from .hardware import detect_hardware

_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ENTROPY_TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_MAX_ALGEBRA_LENGTH = 256
_MAX_ALGEBRA_NODES = 64
_MAX_ENTROPY_LENGTH = 100_000


class ToolBus:
    def __init__(self):
        self._tools = {}

    def register(self, name, description, schema, handler):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Tool names must be lowercase identifiers.")
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Tool descriptions are required.")
        if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
            raise ValueError("Tool schema must declare its properties.")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(key, str) or key not in schema["properties"]
            for key in required
        ):
            raise ValueError("Tool schema has invalid required arguments.")
        supported_types = {
            "array",
            "boolean",
            "integer",
            "null",
            "number",
            "object",
            "string",
        }
        for key, definition in schema["properties"].items():
            if (
                not isinstance(key, str)
                or not isinstance(definition, dict)
                or definition.get("type") not in supported_types
            ):
                raise ValueError("Tool schema has an invalid property definition.")
            if "maxLength" in definition and (
                definition["type"] != "string"
                or not isinstance(definition["maxLength"], int)
                or isinstance(definition["maxLength"], bool)
                or definition["maxLength"] < 0
            ):
                raise ValueError("maxLength is only valid for string arguments.")
        if not callable(handler):
            raise TypeError("Tool handler must be callable.")
        self._tools[name] = {
            "name": name,
            "description": description,
            "schema": schema,
            "handler": handler,
        }

    def describe(self):
        return [
            {
                "name": name,
                "description": tool["description"],
                "schema": tool["schema"],
            }
            for name, tool in sorted(self._tools.items())
        ]

    def invoke(self, name, arguments):
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        tool = self._tools[name]
        schema = tool["schema"]
        properties = schema["properties"]
        required = schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        extra = sorted(set(arguments) - set(properties))
        if missing:
            raise ValueError(f"Missing tool arguments: {', '.join(missing)}.")
        if extra:
            raise ValueError(f"Unsupported tool arguments: {', '.join(extra)}.")
        for key, value in arguments.items():
            expected = properties[key]["type"]
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"Tool argument {key!r} must be a string.")
            if expected == "number" and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"Tool argument {key!r} must be a number.")
            if expected == "number":
                outside_range = (
                    abs(value) > 10**100
                    if isinstance(value, int)
                    else not math.isfinite(value) or abs(value) > 1e100
                )
                if outside_range:
                    raise ValueError(
                        f"Tool argument {key!r} is outside the supported range."
                    )
            if expected == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"Tool argument {key!r} must be an integer.")
            if expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Tool argument {key!r} must be a boolean.")
            if expected == "object" and not isinstance(value, dict):
                raise ValueError(f"Tool argument {key!r} must be an object.")
            if expected == "array" and not isinstance(value, list):
                raise ValueError(f"Tool argument {key!r} must be an array.")
            if expected == "null" and value is not None:
                raise ValueError(f"Tool argument {key!r} must be null.")
            maximum = properties[key].get("maxLength")
            if maximum is not None and len(value) > maximum:
                raise ValueError(
                    f"Tool argument {key!r} exceeds the {maximum}-character limit."
                )
        return tool["handler"](**arguments)


def _bounded_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Calculator only accepts numeric expressions.")
    if isinstance(value, int):
        if abs(value) > 10**100:
            raise ValueError("Calculation result is outside the supported numeric range.")
        return value
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("Calculation result is outside the supported numeric range.")
    return value


def calculate(expression):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A calculator expression is required.")
    if len(expression) > 256:
        raise ValueError("Calculator expressions may not exceed 256 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid arithmetic expression.") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Calculator expression is too complex.")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Calculator only accepts numeric expressions.")
            return _bounded_number(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _bounded_number(_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (
                abs(right) > 12 or abs(left) > 1e12
            ):
                raise ValueError("Exponentiation is limited to powers up to 12.")
            try:
                return _bounded_number(
                    _BINARY_OPERATORS[type(node.op)](left, right)
                )
            except (ArithmeticError, OverflowError) as error:
                raise ValueError(f"Cannot calculate expression: {error}") from error
        raise ValueError("Only basic arithmetic operators and numeric literals are allowed.")

    return evaluate(tree)


def _poly_add(left, right, scale=1.0):
    return [
        (left[index] if index < len(left) else 0.0)
        + scale * (right[index] if index < len(right) else 0.0)
        for index in range(max(len(left), len(right)))
    ]


def _poly_multiply(left, right):
    result = [0.0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    if len(result) > 3 and any(abs(value) > 1e-12 for value in result[3:]):
        raise ValueError("Only polynomial equations up to degree two are supported.")
    return result[:3]


def _polynomial(node):
    if isinstance(node, ast.Expression):
        return _polynomial(node.body)
    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
            or not math.isfinite(node.value)
            or abs(node.value) > 1e100
        ):
            raise ValueError("Algebra accepts only finite, bounded numeric constants.")
        return [float(node.value)]
    if isinstance(node, ast.Name) and node.id == "x":
        return [0.0, 1.0]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        polynomial = _polynomial(node.operand)
        if isinstance(node.op, ast.USub):
            return [-coefficient for coefficient in polynomial]
        return polynomial
    if isinstance(node, ast.BinOp):
        left = _polynomial(node.left)
        right = _polynomial(node.right)
        if isinstance(node.op, ast.Add):
            return _poly_add(left, right)
        if isinstance(node.op, ast.Sub):
            return _poly_add(left, right, scale=-1.0)
        if isinstance(node.op, ast.Mult):
            return _poly_multiply(left, right)
        if isinstance(node.op, ast.Div):
            if len(right) != 1 or right[0] == 0:
                raise ValueError("Polynomial division requires a nonzero constant divisor.")
            return [coefficient / right[0] for coefficient in left]
        if isinstance(node.op, ast.Pow):
            if len(right) != 1 or not right[0].is_integer or not 0 <= right[0] <= 2:
                raise ValueError("Polynomial powers must be integers from zero to two.")
            result = [1.0]
            for _ in range(int(right[0])):
                result = _poly_multiply(result, left)
            return result
    raise ValueError(
        "Use only x, numeric constants, parentheses, and +, -, *, /, or powers 0-2."
    )


def _clean_root(value):
    real = 0.0 if abs(value.real) < 1e-12 else value.real
    imaginary = 0.0 if abs(value.imag) < 1e-12 else value.imag
    if not math.isfinite(real) or not math.isfinite(imaginary):
        raise ValueError("Equation roots exceed the supported numeric range.")
    if abs(real) > 1e100 or abs(imaginary) > 1e100:
        raise ValueError("Equation roots exceed the supported numeric range.")
    if imaginary == 0:
        return real
    return {"real": real, "imaginary": imaginary}


def solve_polynomial(equation):
    if not isinstance(equation, str) or not equation.strip():
        raise ValueError("Provide a polynomial equation in x.")
    if len(equation) > _MAX_ALGEBRA_LENGTH:
        raise ValueError(
            f"Polynomial equations may not exceed {_MAX_ALGEBRA_LENGTH} characters."
        )
    try:
        sides = equation.split("=")
        if len(sides) > 2:
            raise ValueError("Use at most one equals sign in a polynomial equation.")
        left = ast.parse(sides[0].strip(), mode="eval")
        right = ast.parse(sides[1].strip(), mode="eval") if len(sides) == 2 else None
    except SyntaxError as error:
        raise ValueError(
            "Invalid equation syntax. Use ** for powers, for example x**2."
        ) from error
    node_count = sum(1 for _ in ast.walk(left))
    if right is not None:
        node_count += sum(1 for _ in ast.walk(right))
    if node_count > _MAX_ALGEBRA_NODES:
        raise ValueError("Polynomial equation is too complex.")

    coefficients = _polynomial(left)
    if right is not None:
        coefficients = _poly_add(coefficients, _polynomial(right), scale=-1.0)
    coefficients.extend([0.0] * (3 - len(coefficients)))
    constant, linear, quadratic = coefficients
    if abs(quadratic) > 1e-12:
        discriminant = linear * linear - 4.0 * quadratic * constant
        radical = cmath.sqrt(discriminant)
        roots = [
            _clean_root((-linear - radical) / (2.0 * quadratic)),
            _clean_root((-linear + radical) / (2.0 * quadratic)),
        ]
        roots.sort(
            key=lambda root: (
                root["real"] if isinstance(root, dict) else root,
                root.get("imaginary", 0.0) if isinstance(root, dict) else 0.0,
            )
        )
        return {"degree": 2, "roots": roots}
    if abs(linear) > 1e-12:
        return {"degree": 1, "roots": [_clean_root(complex(-constant / linear))]}
    if abs(constant) <= 1e-12:
        return {"degree": 0, "identity": True, "roots": "all real numbers"}
    return {"degree": 0, "identity": False, "roots": []}


def information_entropy(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide non-empty text to measure its entropy.")
    if len(text) > _MAX_ENTROPY_LENGTH:
        raise ValueError(
            f"Entropy input may not exceed {_MAX_ENTROPY_LENGTH} characters."
        )

    def entropy(items):
        if not items:
            return 0.0
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        total = len(items)
        return -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )

    characters = list(text)
    tokens = _ENTROPY_TOKENS.findall(text.casefold())
    return {
        "characters": len(characters),
        "character_entropy_bits_per_symbol": entropy(characters),
        "tokens": len(tokens),
        "token_entropy_bits_per_token": entropy(tokens),
        "note": (
            "Empirical Shannon entropy of this sample, not a measure of truth, "
            "intelligence, or the model's internal state."
        ),
    }


def build_tool_bus(knowledge, runtime=None):
    bus = ToolBus()
    bus.register(
        "calculator",
        "Evaluate a bounded basic arithmetic expression without executing code.",
        {
            "type": "object",
            "properties": {"expression": {"type": "string", "maxLength": 256}},
            "required": ["expression"],
        },
        lambda expression: {
            "expression": expression,
            "result": calculate(expression),
        },
    )
    bus.register(
        "algebraic_solve",
        "Solve a bounded real-coefficient polynomial equation of degree at most two in x.",
        {
            "type": "object",
            "properties": {"equation": {"type": "string", "maxLength": 256}},
            "required": ["equation"],
        },
        lambda equation: {
            "equation": equation,
            **solve_polynomial(equation),
        },
    )
    bus.register(
        "information_entropy",
        "Measure empirical Shannon entropy of Unicode characters and word/punctuation tokens.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": _MAX_ENTROPY_LENGTH}
            },
            "required": ["text"],
        },
        lambda text: information_entropy(text),
    )
    bus.register(
        "knowledge_search",
        "Search locally indexed text and return ranked excerpts with source citations.",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": 1000}},
            "required": ["query"],
        },
        lambda query: {"results": knowledge.search(query)},
    )
    bus.register(
        "knowledge_digest",
        "Return a short extractive digest of relevant local sentences with citations.",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": 1000}},
            "required": ["query"],
        },
        lambda query: {"sentences": knowledge.digest(query)},
    )

    def local_environment():
        documents = knowledge.list_documents()
        hardware = detect_hardware()
        sources = [Path(document["source"]).name for document in documents[:50]]
        return {
            "assistant": "Woa-Awe-oo",
            "mewtronic_definition": (
                "Composition of this local model, bundled knowledge, explicitly "
                "indexed documents, and registered local tools."
            ),
            "is_conscious": False,
            "model_checkpoint_available": (
                getattr(runtime, "available", None)
                if runtime is not None
                else None
            ),
            "platform": platform.platform(),
            "compute_device": hardware["device"],
            "indexed_document_count": len(documents),
            "indexed_sources": sources,
            "additional_sources_not_listed": max(0, len(documents) - len(sources)),
            "local_tools": [
                tool["name"]
                for tool in bus.describe()
                if tool["name"] != "local_environment"
            ],
        }

    bus.register(
        "local_environment",
        "Report the local assistant, model, indexed knowledge, and tools without scanning files.",
        {"type": "object", "properties": {}, "required": []},
        lambda: local_environment(),
    )
    return bus
