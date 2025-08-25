"""
This module implements the Gemma model, which is a causal language model that uses the Gemma model.
It is used to generate text from a given input text.

"""
import torch
from torch import nn
from typing import Optional, Tuple
import math


from config import GemmaConfig
from kv_cache import KVCache
from rotary_embedding import apply_rotary_pos_emb, GemmaRotaryEmbedding

class GemmaRMSNorm(nn.Module):
    '''
    This class implements the RMSNorm layer.

    Args:
        dim (int): The dimension of the layer.
        eps (float): The epsilon value for numerical stability.

    Attributes:
        weight (nn.Parameter): The weight of the layer.
        eps (float): The epsilon value for numerical stability.

    Methods:
        forward(self, x: torch.Tensor) -> torch.Tensor: Forward pass of the RMSNorm layer.

    Example:
        >>> config = GemmaConfig()
        >>> rms_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        >>> hidden_states = torch.randn(1, 10, 768)
        >>> output = rms_norm(hidden_states)
        >>> print(output.shape) # [1, 10, 768]

        input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaRMSNorm # [Batch_Size, Seq_Len, Hidden_Size]
        Outputs:
        -> hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
    '''
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # a small constant to avoid division by zero
        self.eps = eps
        # a learnable parameter
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # [Batch_Size, Seq_Len, Hidden_Size]
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function repeats the key and value states along the sequence length dimension.

    Args:
        hidden_states (torch.Tensor): The hidden states to repeat.
        n_rep (int): The number of times to repeat the hidden states.

    Returns:
        torch.Tensor: The repeated hidden states.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # If we don't need to repeat the hidden states, return the hidden states
    if n_rep == 1:
        return hidden_states
    # Otherwise, repeat the hidden states along the sequence length dimension
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Reshape the hidden states to the desired shape
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    '''
    This class implements the GemmaAttention layer.
    It contains the self-attention mechanism.

    Args:
        config (GemmaConfig): The configuration of the Gemma model.
        layer_idx (int): The index of the layer.

    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        layer_idx (int): The index of the layer.
        attention_dropout (float): The dropout rate of the attention.
        hidden_size (int): The hidden size of the model.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        num_key_value_heads (int): The number of key-value heads.
        num_key_value_groups (int): The number of key-value groups.
        max_position_embeddings (int): The maximum position embeddings.
        rope_theta (float): The theta value for the RoPE.
        is_causal (bool): Whether the attention is causal.

    Methods:
        forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, kv_cache: Optional[KVCache] = None) -> torch.Tensor: Forward pass of the GemmaAttention layer.

    Example:
        >>> config = GemmaConfig()
        >>> attention = GemmaAttention(config, layer_idx=0)
        >>> hidden_states = torch.randn(1, 10, 768)
        >>> attention_mask = torch.randn(1, 10, 10)
        >>> position_ids = torch.arange(10)
        >>> kv_cache = KVCache(config)
        >>> output = attention(hidden_states, attention_mask, position_ids, kv_cache)
        >>> print(output.shape) # [1, 10, 768]

        Inputs:
        -> hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        -> attention_mask: [Batch_Size, Seq_Len_Q, Seq_Len_KV]
        -> position_ids: [Batch_Size, Seq_Len]
        -> kv_cache: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        Outputs:
        -> hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        -> attn_weights: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
    '''

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # Initialize the configuration
        self.config = config
        # Initialize the layer index corresponding to the GemmaDecoderLayer
        self.layer_idx = layer_idx
        # Initialize the attention dropout
        self.attention_dropout = config.attention_dropout
        # Initialize the hidden size
        self.hidden_size = config.hidden_size
        # Initialize the number of attention heads
        self.num_heads = config.num_attention_heads
        # Initialize the dimension of each attention head
        self.head_dim = config.head_dim
        # Initialize the number of key-value heads
        self.num_key_value_heads = config.num_key_value_heads
        # use for group query attention heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Initialize the maximum position embeddings
        self.max_position_embeddings = config.max_position_embeddings
        # Initialize the theta value for the RoPE
        self.rope_theta = config.rope_theta
        # Initialize the causal flag
        self.is_causal = True

        # Check if the hidden size is divisible by the number of attention heads
        assert self.hidden_size % self.num_heads == 0 

        """
        hidden_size = 1024
        num_heads = 8
        head_dim = 128
        
        Wq: [1024, 8 * 128] = [1024, 1024]
        Wk: [1024, 1 * 128] = [1024, 128]
        Wv: [1024, 1 * 128] = [1024, 128]
        
        
        """           

        # Initialize the query projection layer (hidden_size -> num_heads * head_dim)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

        # Instead of using multi-head attention, we use group query attention
        # Initialize the key projection layer (hidden_size -> num_key_value_heads * head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Initialize the value projection layer (hidden_size -> num_key_value_heads * head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Initialize the output projection layer (num_heads * head_dim -> hidden_size)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        
        # Initialize the rotary embedding layer (head_dim -> max_position_embeddings)
        # We use the rotary embedding to project the hidden states to the hidden size
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # Input: [Batch_Size, Seq_Len, Hidden_Size]
        bsz, q_len, _ = hidden_states.size() 
        # Step 1: Project the hidden states to the query, key, and value states
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)

        # Step 2: Split the query, key, and value states into groups
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Step 3: Apply the rotary embedding
        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        # Step 4: Apply the rotary embedding to the query and key states
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Step 5: Update the key and value states in the kv_cache
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Step 6: We don't have cuda_graph support for group query attention, 
        # so we need to repeat the key and value states
        # [Batch_Size, Num_Heads_KV , Seq_Len, Head_Dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Step 7: Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)    
        
        # Step 8: Apply the attention mask
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Step 9: Apply the softmax
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Step 10: Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Step 11: Multiply by the values. [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim] -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Check if the output shape is correct
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        
        # Step 12: Make sure the sequence length is the second dimension. 
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Step 13: Concatenate all the heads together. 
        # [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        
        # Step 14: Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        # Step 15: Return the output states and the attention weights
        return attn_output, attn_weights
        

class GemmaMLP(nn.Module):
    '''
    This class implements the GemmaMLP layer.
    It contains the gate, up, and down projection layers.
    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        hidden_size (int): The hidden size of the model.
        intermediate_size (int): The intermediate size of the model.
        gate_proj (nn.Linear): The gate projection layer.
        up_proj (nn.Linear): The up projection layer.
        down_proj (nn.Linear): The down projection layer.

    Methods:
        forward(self, x: torch.Tensor) -> torch.Tensor: Forward pass of the GemmaMLP layer.

    Example:
        >>> config = GemmaConfig()
        >>> mlp = GemmaMLP(config)
        >>> hidden_states = torch.randn(1, 10, 768)
        >>> output = mlp(hidden_states)
        >>> print(output.shape) # [1, 10, 768]

        Inputs: hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        -> gate_proj: [Batch_Size, Seq_Len, Intermediate_Size]
        -> up_proj: [Batch_Size, Seq_Len, Intermediate_Size]
        -> down_proj: [Batch_Size, Seq_Len, Hidden_Size]
        Outputs:
        -> hidden_states: [Batch_Size, Seq_Len, Hidden_Size]

    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Hidden size of the model
        self.hidden_size = config.hidden_size
        # Intermediate size of the model
        self.intermediate_size = config.intermediate_size
        # Initialize the gate projection layer
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # Initialize the up projection layer
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # Initialize the down projection layer
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        """
        Forward pass of the GemmaMLP layer.
        """
        # Step 1: Apply the gate projection layer
        y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        y = nn.functional.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # Step 3: Apply the up projection layer
        j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # Step 4: Apply the down projection layer
        z = y * j # [Batch_Size, Seq_Len, Intermediate_Size] 
        z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        # [Batch_Size, Seq_Len, Hidden_Size]
        return z


class GemmaDecoderLayer(nn.Module):
    '''
    This class implements the GemmaDecoderLayer.
    It contains the self-attention and the MLP layers.
    
    Args:
        config (GemmaConfig): The configuration of the Gemma model.
        layer_idx (int): The index of the layer.

    Attributes:
        hidden_size (int): The hidden size of the model.
        self_attn (GemmaAttention): The self-attention layer.
        mlp (GemmaMLP): The MLP layer.
        input_layernorm (GemmaRMSNorm): The input layer normalization.
        post_attention_layernorm (GemmaRMSNorm): The post-attention layer normalization.

    Methods:
        forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, kv_cache: Optional[KVCache] = None) -> torch.Tensor: Forward pass of the GemmaDecoderLayer.

    Example:
        >>> config = GemmaConfig()
        >>> layer = GemmaDecoderLayer(config, layer_idx=0)
        >>> hidden_states = torch.randn(1, 10, 768)
        >>> attention_mask = torch.randn(1, 10, 10)
        >>> position_ids = torch.arange(10)
        >>> kv_cache = KVCache(config)
        >>> output = layer(hidden_states, attention_mask, position_ids, kv_cache)
        >>> print(output.shape) # [1, 10, 768]

        input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaRMSNorm # [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaAttention # [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaRMSNorm # [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaMLP # [Batch_Size, Seq_Len, Hidden_Size]
        Outputs:
        -> hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
    '''
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        
        # Hidden size of the model
        self.hidden_size = config.hidden_size

        # Initialize the self-attention layer
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        # Initialize the MLP layer
        self.mlp = GemmaMLP(config)

        # Initialize the input layer normalization
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize the post-attention layer normalization
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the GemmaDecoderLayer.
        """
        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        # Normalize the hidden states
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        # Pass the hidden states through the self-attention layer
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        # Add the residual connection
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        # Normalize the hidden states
        residual = hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        # Normalize the hidden states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        # Pass the hidden states through the MLP layer
        hidden_states = self.mlp(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        # Add the residual connection
        hidden_states = residual + hidden_states

        return hidden_states
    

class GemmaModel(nn.Module):
    '''
    This class implements the GemmaModel
    Given the input embeddings, it passes them through the decoder layers and returns the hidden states.

    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        padding_idx (int): The padding index of the model.
        vocab_size (int): The vocabulary size of the model.
        embed_tokens (nn.Embedding): The embedding layer to project the input ids to the hidden size.
        layers (nn.ModuleList): The list of decoder layers of the model.
        norm (GemmaRMSNorm): The normalization layer of the model.

    Methods:
        get_input_embeds(self) -> torch.Tensor: Get the input embeddings for the input ids.
        forward(self, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, kv_cache: Optional[KVCache] = None) -> torch.FloatTensor: Forward pass of the GemmaModel.

    Example:
        >>> config = GemmaConfig()
        >>> model = GemmaModel(config)
        >>> inputs_embeds = model.get_input_embeds() # [Batch_Size, Seq_Len, Hidden_Size]
        >>> outputs = model(inputs_embeds=inputs_embeds) # [Batch_Size, Seq_Len, Hidden_Size]
        >>> print(outputs.shape) # [Batch_Size, Seq_Len, Hidden_Size]

        input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaDecoderLayer * num_layers # [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaRMSNorm # [Batch_Size, Seq_Len, Hidden_Size]
        Outputs: 
        -> hidden_states  #[Batch_Size, Seq_Len, Hidden_Size]
    
    '''

    def __init__(self, config: GemmaConfig):
        """
        Initialize the GemmaModel.
        """
        super().__init__()
        # Initialize the configuration
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeds(self) -> torch.Tensor:
        """
        Get the input embeddings for the input ids.
        """
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass of the GemmaModel.

        Args:
            attention_mask (Optional[torch.Tensor]): The attention mask of the input embeddings.
            position_ids (Optional[torch.LongTensor]): The position ids of the input embeddings.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings of the model.
            kv_cache (Optional[KVCache]): The key-value cache of the model.

        Returns:
            hidden_states: [Batch_Size, Seq_Len, Hidden_Size]

        """
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        # Normalize the hidden states
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # Pass the hidden states through the decoder layer
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # Return the hidden states
        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states



class GemmaForCausalLM(nn.Module):
    """
    This class implements the GemmaForCausalLM model, which is a causal language model that uses the Gemma model.
    It is used to generate text from a given input text.
    
    Args:
        config (GemmaConfig): The configuration of the Gemma model.
    
    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        model (GemmaModel): The Gemma model.
        vocab_size (int): The vocabulary size of the model.
        lm_head (nn.Linear): The linear layer to project the hidden states to the vocabulary size.

    Example:
        >>> config = GemmaConfig()
        >>> model = GemmaForCausalLM(config)
        >>> inputs_embeds = model.get_input_embeds() # [Batch_Size, Seq_Len, Hidden_Size]
        >>> outputs = model(inputs_embeds=inputs_embeds) # [Batch_Size, Seq_Len, Hidden_Size]
        >>> print(outputs.shape) # [Batch_Size, Seq_Len, Hidden_Size]
        >>> print(outputs["logits"].shape) # [Batch_Size, Seq_Len, Vocab_Size]
        >>> print(outputs["kv_cache"].shape) # [Batch_Size, Seq_Len, Hidden_Size]

        input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaModel # [Batch_Size, Seq_Len, Hidden_Size]
        -> lm_head # [Batch_Size, Seq_Len, Vocab_Size]
        Outputs:
        -> logits: [Batch_Size, Seq_Len, Vocab_Size]
        -> kv_cache: [Batch_Size, Seq_Len, Hidden_Size] (if kv_cache is not None)
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        # Initialize the configuration
        self.config = config
        # Initialize the Gemma model
        self.model = GemmaModel(config)
        # Initialize the vocabulary size
        self.vocab_size = config.vocab_size
        # Initialize the linear layer to project the hidden states to the vocabulary size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self):
        """
        Tie the weights of the linear layer to the weights of the model.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeds(self):
        """
        Get the input embeddings for the input ids.
        """
        return self.model.embed_tokens
    
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Forward pass of the GemmaForCausalLM model.
        """
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        # Get the hidden states from the outputs
        hidden_states = outputs
        # Get the logits from the linear layer
        logits = self.lm_head(hidden_states)
        # Convert the logits to float
        logits = logits.float()
        
        # Return the logits and the kv_cache if it is not None
        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data