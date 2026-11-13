import math
from dataclasses import asdict, dataclass


class ModelDependencyError(RuntimeError):
    pass


_ARCHITECTURE_BUILDERS = {}


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "decoder_only_transformer"
    vocab_size: int = 261
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    learned_position_embeddings: bool = False

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if self.d_model < 1:
            raise ValueError("d_model must be positive.")
        if self.n_heads < 1:
            raise ValueError("n_heads must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("Each attention head must have an even dimension for RoPE.")
        if self.context_length < 2:
            raise ValueError("context_length must be at least two.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least one.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

    def to_dict(self):
        return asdict(self)


def register_architecture(name, builder):
    if not name or name == "decoder_only_transformer":
        raise ValueError("A custom architecture must have a unique non-empty name.")
    if name in _ARCHITECTURE_BUILDERS:
        raise ValueError(f"Architecture {name!r} is already registered.")
    if not callable(builder):
        raise TypeError("Architecture builder must be callable.")
    _ARCHITECTURE_BUILDERS[name] = builder


def _torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as error:
        raise ModelDependencyError(
            "PyTorch is required for training and model inference. "
            'Install it with `pip install -e ".[train]"`.'
        ) from error
    return torch, nn, functional


def build_model(config=None):
    torch, nn, functional = _torch()
    config = config or ModelConfig()
    if config.architecture != "decoder_only_transformer":
        builder = _ARCHITECTURE_BUILDERS.get(config.architecture)
        if builder is None:
            raise ValueError(f"Unsupported model architecture: {config.architecture}")
        return builder(config)

    class RMSNorm(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(width))

        def forward(self, value):
            normalized = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)
            return normalized * self.scale

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = config.d_model // config.n_heads
            self.n_heads = config.n_heads
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.output = nn.Linear(config.d_model, config.d_model, bias=False)
            self.dropout = config.dropout
            half_dim = self.head_dim // 2
            inv_freq = 1.0 / (
                10000
                ** (
                    torch.arange(half_dim, dtype=torch.float32)
                    / half_dim
                )
            )
            positions = torch.arange(config.context_length, dtype=torch.float32)
            angles = torch.outer(positions, inv_freq)
            self.register_buffer(
                "rope_cos", angles.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "rope_sin", angles.sin()[None, None, :, :], persistent=False
            )

        def apply_rope(self, value):
            length = value.shape[-2]
            cos = self.rope_cos[:, :, :length].to(dtype=value.dtype)
            sin = self.rope_sin[:, :, :length].to(dtype=value.dtype)
            first, second = value.chunk(2, dim=-1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

        def forward(self, value):
            batch, length, width = value.shape
            query, key, content = self.qkv(value).chunk(3, dim=-1)
            shape = (batch, length, self.n_heads, self.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            content = content.view(shape).transpose(1, 2)
            query = self.apply_rope(query)
            key = self.apply_rope(key)
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                content,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
            return self.output(attended)

    class SwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = (8 * config.d_model) // 3
            hidden = ((hidden + 63) // 64) * 64
            self.gate = nn.Linear(config.d_model, hidden, bias=False)
            self.value = nn.Linear(config.d_model, hidden, bias=False)
            self.output = nn.Linear(hidden, config.d_model, bias=False)

        def forward(self, value):
            return self.output(functional.silu(self.gate(value)) * self.value(value))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_norm = RMSNorm(config.d_model)
            self.attention = CausalSelfAttention()
            self.ffn_norm = RMSNorm(config.d_model)
            self.ffn = SwiGLU()

        def forward(self, value):
            value = value + self.attention(self.attention_norm(value))
            return value + self.ffn(self.ffn_norm(value))

    class DecoderOnlyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.position_embedding = (
                nn.Embedding(config.context_length, config.d_model)
                if config.learned_position_embeddings
                else None
            )
            self.blocks = nn.ModuleList(Block() for _ in range(config.n_layers))
            self.final_norm = RMSNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, token_ids, targets=None):
            if token_ids.shape[1] > config.context_length:
                raise ValueError("Input exceeds the model context length.")
            value = self.token_embedding(token_ids)
            if self.position_embedding is not None:
                positions = torch.arange(token_ids.shape[1], device=token_ids.device)
                value = value + self.position_embedding(positions).unsqueeze(0)
            for block in self.blocks:
                value = block(value)
            logits = self.lm_head(self.final_norm(value))
            if targets is None:
                return logits
            loss = functional.cross_entropy(
                logits.reshape(-1, config.vocab_size), targets.reshape(-1)
            )
            return logits, loss

        @torch.no_grad()
        def generate(
            self,
            token_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40,
            eos_token_id=None,
            min_new_tokens=0,
            forbidden_token_ids=None,
        ):
            if temperature <= 0 or not math.isfinite(temperature):
                raise ValueError("temperature must be a positive finite number.")
            if max_new_tokens < 0:
                raise ValueError("max_new_tokens cannot be negative.")
            if top_k < 0:
                raise ValueError("top_k cannot be negative.")
            if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
                raise ValueError("min_new_tokens must be between zero and max_new_tokens.")
            if eos_token_id is not None and not 0 <= eos_token_id < config.vocab_size:
                raise ValueError("eos_token_id is outside the vocabulary.")
            forbidden_token_ids = forbidden_token_ids or []
            if any(
                token < 0 or token >= config.vocab_size
                for token in forbidden_token_ids
            ):
                raise ValueError("forbidden token ID is outside the vocabulary.")
            self.eval()
            for index in range(max_new_tokens):
                context = token_ids[:, -config.context_length :]
                logits = self(context)[:, -1, :] / temperature
                if eos_token_id is not None and index < min_new_tokens:
                    logits[:, eos_token_id] = float("-inf")
                if forbidden_token_ids:
                    logits[:, forbidden_token_ids] = float("-inf")
                if top_k:
                    cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probabilities = functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                token_ids = torch.cat((token_ids, next_token), dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
            return token_ids

    return DecoderOnlyTransformer()
