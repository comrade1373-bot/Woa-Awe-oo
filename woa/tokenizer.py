class ByteTokenizer:
    """A deterministic byte tokenizer with reserved conversation boundaries."""

    SYSTEM = 256
    USER = 257
    ASSISTANT = 258
    EOS = 259
    BOS = 260
    VOCAB_SIZE = 261

    def encode(self, text):
        return [self.BOS, *text.encode("utf-8"), self.EOS]

    def encode_message(self, role, content):
        role_tokens = {
            "system": self.SYSTEM,
            "user": self.USER,
            "assistant": self.ASSISTANT,
        }
        try:
            role_token = role_tokens[role]
        except KeyError as error:
            raise ValueError("role must be 'system', 'user', or 'assistant'.") from error
        return [role_token, *content.encode("utf-8"), self.EOS]

    def decode(self, token_ids):
        byte_values = [token for token in token_ids if 0 <= token <= 255]
        return bytes(byte_values).decode("utf-8", errors="replace")
