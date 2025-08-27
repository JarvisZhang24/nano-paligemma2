import torch
from torch import nn
from typing import Optional, Tuple, List
import math
from src.text.gemma2_config import Gemma2Config
from src.kv_cache import KVCache
from src.attention.attention import Gemma2GQA



################################### Gemma 2 Model ###################################



class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # dim is the hidden size of the model
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.type_as(x)


class MLP(nn.Module):

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # (batch_size, sequence_length, hidden_size)
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )



class Gemma2Layer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Gemma2GQA(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # (batch_size, sequence_length, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # (batch_size, sequence_length, hidden_size)
        return hidden_states


class Gemma2Model(nn.Module):

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Gemma2Layer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        output = inputs_embeds
        # (batch_size, sequence_length, hidden_size)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=output.dtype)
        output = output * normalizer

        for layer in self.layers:
            output = layer(
                output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # (batch_size, sequence_length, hidden_size)
        output = self.norm(output)

        return output


class Gemma2(nn.Module):

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.model = Gemma2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config.final_logit_softcapping = config.final_logit_softcapping

    def get_input_embeddings(self):
        return self.model.embed_tokens

    # We reuse the embeddings of the model in the last linear layer
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # inputs_embeds: (batch_size, sequence_length, hidden_size)
        # outputs = (batch_size, sequence_length, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(outputs).float()

        # Apply softcapping to the logits if specified
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        # We return the logits and the key-value cache if it is used.
        return return_data
