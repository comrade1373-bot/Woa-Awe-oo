import re


_INTENTS = (
    ("greeting", re.compile(r"^(?:hi|hello|hey|good morning|good evening)\b", re.I)),
    ("appreciation", re.compile(r"\b(?:thanks|thank you|much appreciated)\b", re.I)),
    ("correction", re.compile(r"\b(?:actually|correction|i meant|that's wrong)\b", re.I)),
    ("question", re.compile(r"\?\s*$")),
    ("request", re.compile(r"^(?:please\b|can you\b|could you\b|would you\b)", re.I)),
)
_TONE_PATTERNS = {
    "urgency_words": re.compile(r"\b(?:urgent|asap|immediately|right now)\b", re.I),
    "frustration_words": re.compile(
        r"\b(?:frustrated|annoyed|upset|angry|fed up)\b", re.I
    ),
    "positive_words": re.compile(
        r"\b(?:glad|happy|excited|wonderful|great|love it)\b", re.I
    ),
    "uncertainty_words": re.compile(
        r"\b(?:maybe|perhaps|possibly|not sure|i think)\b", re.I
    ),
}
_RHETORIC_PATTERNS = {
    "possible_hyperbole": re.compile(
        r"\b(?:always|never|everyone|nobody|the best ever|a million times)\b", re.I
    ),
    "possible_figuration": re.compile(
        r"\b(?:metaphor|simile|like a|as if|as though|figuratively)\b", re.I
    ),
    "wordplay_markers": re.compile(
        r"\b(?:pun|wordplay|play on words|double meaning|homophone|riddle)\b", re.I
    ),
}


def communication_cues(text):
    """Return ephemeral lexical cues, not a persistent or definitive user profile."""
    content = text.strip()
    lowered = content.casefold()
    intent = next(
        (name for name, pattern in _INTENTS if pattern.search(content)),
        "statement",
    )
    tone_cues = [
        name for name, pattern in _TONE_PATTERNS.items() if pattern.search(content)
    ]
    rhetorical_cues = [
        name for name, pattern in _RHETORIC_PATTERNS.items() if pattern.search(content)
    ]
    if any(token in lowered for token in ("lol", "lmao", "😂", "😄", "😉")):
        tone_cues.append("playful_marker")
    if any(token in lowered for token in ("please", "would you", "could you")):
        tone_cues.append("polite_marker")
    if any(token in lowered for token in ("gonna", "wanna", "kinda", "y'all")):
        register = "casual"
    elif any(token in lowered for token in ("therefore", "regarding", "furthermore")):
        register = "formal"
    else:
        register = "neutral"
    return {
        "intent": intent,
        "register": register,
        "tone_cues": tone_cues,
        "rhetorical_cues": rhetorical_cues,
        "certainty": "heuristic",
        "persistent_profile": False,
    }


def prompt_guidance(cues):
    tone = ", ".join(cues["tone_cues"]) or "none"
    rhetoric = ", ".join(cues["rhetorical_cues"]) or "none"
    return (
        f"Possible cues in the latest message: intent={cues['intent']}, "
        f"register={cues['register']}, tone={tone}, rhetoric={rhetoric}. "
        "These are heuristic observations about wording, "
        "not facts about the person. Adapt only the response style when useful; "
        "do not diagnose, label, or assume a lasting emotional state."
    )
