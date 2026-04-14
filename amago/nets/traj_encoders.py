"""
Seq2Seq models for long-term memory.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple

import torch
from torch import nn
import numpy as np
import gin

from amago.nets import ff, transformer, utils
from amago.utils import amago_warning


##############################
## TrajEncoder Registration ##
##############################

_TRAJ_ENCODER_REGISTRY: dict[str, type] = {}


def register_traj_encoder(name: str):
    """Decorator to register a TrajEncoder class under a shortcut name.

    Args:
        name: The shortcut name to register the encoder under (e.g., "transformer", "ff").

    Example:
        @gin.configurable
        @register_traj_encoder("my_encoder")
        class MyCustomTrajEncoder(TrajEncoder):
            ...
    """

    def decorator(cls):
        if name in _TRAJ_ENCODER_REGISTRY:
            raise ValueError(
                f"TrajEncoder '{name}' is already registered to {_TRAJ_ENCODER_REGISTRY[name]}. "
                f"Cannot re-register to {cls}."
            )
        _TRAJ_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_traj_encoder_cls(name: str) -> type:
    """Look up a registered TrajEncoder class by its shortcut name."""
    if name not in _TRAJ_ENCODER_REGISTRY:
        available = list(_TRAJ_ENCODER_REGISTRY.keys())
        raise KeyError(
            f"TrajEncoder '{name}' is not registered. Available: {available}"
        )
    return _TRAJ_ENCODER_REGISTRY[name]


def list_registered_traj_encoders() -> list[str]:
    """Return a list of all registered TrajEncoder shortcut names."""
    return list(_TRAJ_ENCODER_REGISTRY.keys())


class TrajEncoder(nn.Module, ABC):
    """Abstract base class for trajectory encoders.

    An agent's "TrajEncoder" is the sequence model in charge of mapping the output
    of the "TstepEncoder" for each timestep of the trajectory to the latent
    dimension where actor-critic learning takes place. Because the actor and
    critic are feed-forward networks, this is the place to add long-term memory
    over previous timesteps.

    Note:
        It would *not* make sense for the sequence model defined here to be
        bi-directional or non-causal.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. Any inputs will have
            been trimmed to this length before reaching the TrajEncoder.
    """

    def __init__(self, tstep_dim: int, max_seq_len: int, **kwargs):
        super().__init__()
        self.tstep_dim = tstep_dim
        self.max_seq_len = max_seq_len

    @property
    @abstractmethod
    def emb_dim(self) -> int:
        """Defines the expected output dim of this model.

        Used to infer the input dim of actor/critics.

        Returns:
            int: The embedding dimension.
        """
        pass

    def init_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        """Hook to create an architecture-specific hidden state.

        Return value is passed as `TrajEncoder.forward(..., hidden_state=self.init_hidden_state(...))`
        when the agent begins to interact with the environment.

        Args:
            batch_size: Number of parallel environments.
            device: Device to store hidden state tensors (if applicable).

        Returns:
            Optional[Any]: Some hidden state object, or None if not applicable.
                Defaults to None.
        """
        return None

    def reset_hidden_state(
        self, hidden_state: Optional[Any], dones: np.ndarray
    ) -> Optional[Any]:
        """Hook to implement architecture-specific hidden state reset.

        Args:
            hidden_state: We only expect to see hidden states that were created
                by `self.init_hidden_state()`.
            dones: A bool array of shape (num_parallel_envs,) where True
                indicates the agent loop has finished this episode and expects
                the hidden state for this batch index to be reset.

        Returns:
            Optional[Any]: Architecture-specific hidden state. Defaults to a
                no-op: `new_hidden_state = hidden_state`.
        """
        return hidden_state

    @abstractmethod
    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[Any] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Sequence model forward pass.

        Args:
            seq: [Batch, Num Timesteps, TstepDim]. TstepDim is defined by the
                output of the TstepEncoder.
            time_idxs: [Batch, Num Timesteps, 1]. A sequence of ints tying the
                input seq to the number of steps that have passed since the start
                of the episode. Can be used to compute position embeddings or
                other temporal features.
            hidden_state: Architecture-specific hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[Any]]: A tuple containing:
                - output_seq: [Batch, Timestep, self.emb_dim]. Output of our
                    seq2seq model.
                - new_hidden_state: Architecture-specific hidden state. Expected to
                    be `None` if input `hidden_state` is `None`. Otherwise, we
                    assume we are at inference time and that this `forward`
                    method has handled any updates to the hidden state that were
                    needed.
        """
        pass


@gin.configurable
@register_traj_encoder("ff")
class FFTrajEncoder(TrajEncoder):
    """Feed-forward (memory-free) trajectory encoder.

    A useful tool for applying AMAGO to standard MDPs and benchmarking general
    RL details/hyperparamters on common benchmarks. The feed-forward architecture
    is designed to be close to an attention-less Transformer (residual blocks,
    norm, dropout, etc.). This makes it easy to create perfect 1:1 ablations of
    "memory vs. no memory" by only changing the TrajEncoder and without touching
    the `max_seq_len`, which would have the side-effect of changing the effective
    batch size of actor-critic learning.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. Any inputs will have
            been trimmed to this length before reaching the TrajEncoder.

    Keyword Args:
        d_model: Dimension of the main residual stream and output. 1:1 with how
            this would be defined in a Transformer. Defaults to 256.
        d_ff: Hidden dim of the feed-forward network along each residual block.
            1:1 with how this would be defined in a Transformer. Defaults to
            `4 * d_model`.
        n_layers: Number of residual feed-forward blocks. 1:1 with how this would
            be defined in a Transformer. Defaults to 1.
        dropout: Dropout rate. Equivalent to the dropout paramter of feed-forward
            blocks in a Transformer, but is also applied to the first and last
            linear layers (inp --> d_model and d_model --> out). Defaults to 0.0.
        activation: Activation function. Defaults to "leaky_relu".
        norm: Normalization for hidden layers within FFBlocks. Defaults to None.
        out_norm: Normalization for the final output. Defaults to "layer".
        layer_type: Type of linear layer to use. Defaults to nn.Linear.
    """

    def __init__(
        self,
        tstep_dim,
        max_seq_len,
        d_model: int = 256,
        d_ff: Optional[int] = None,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        norm: Optional[str] = None,
        out_norm: str = "layer",
        layer_type: type[nn.Module] = nn.Linear,
    ):
        super().__init__(tstep_dim, max_seq_len)
        d_ff = d_ff or d_model * 4
        self.traj_emb = layer_type(tstep_dim, d_model)
        self.traj_blocks = nn.ModuleList(
            [
                ff.FFBlock(
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    layer_type=layer_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.traj_last = layer_type(d_model, d_model)
        self.out_norm = ff.Normalization(out_norm, d_model)
        self.activation = utils.activation_switch(activation)
        self.dropout = nn.Dropout(dropout)
        self._emb_dim = d_model

    @torch.compile
    def _traj_blocks_forward(self, seq: torch.Tensor) -> torch.Tensor:
        traj_emb = self.dropout(self.activation(self.traj_emb(seq)))
        for traj_block in self.traj_blocks:
            traj_emb = traj_block(traj_emb)
        traj_emb = self.traj_last(traj_emb)
        traj_emb = self.dropout(self.out_norm(traj_emb))
        return traj_emb

    def forward(
        self, seq, time_idxs=None, hidden_state=None, log_dict: Optional[dict] = None
    ):
        return self._traj_blocks_forward(seq), hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
@register_traj_encoder("rnn")
class GRUTrajEncoder(TrajEncoder):
    """RNN (GRU) Trajectory Encoder.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. Any inputs will have
            been trimmed to this length before reaching the TrajEncoder.

    Keyword Args:
        d_hidden: Dimension of the hidden state of the GRU. Defaults to 256.
        n_layers: Number of layers in the GRU. Defaults to 2.
        d_output: Dimension of the output linear layer after the GRU. Defaults to
            256.
        norm: Normalization applied after the final linear layer. Defaults to
            "layer" (LayerNorm).
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        d_output: int = 256,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.rnn = nn.GRU(
            input_size=tstep_dim,
            hidden_size=d_hidden,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.out = nn.Linear(d_hidden, d_output)
        self.out_norm = ff.Normalization(norm, d_output)
        self._emb_dim = d_output

    def reset_hidden_state(self, hidden_state, dones):
        assert hidden_state is not None
        hidden_state[:, dones] = 0.0
        return hidden_state

    def forward(
        self, seq, time_idxs=None, hidden_state=None, log_dict: Optional[dict] = None
    ):
        output_seq, new_hidden_state = self.rnn(seq, hidden_state)
        out = self.out_norm(self.out(output_seq))
        return out, new_hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
@register_traj_encoder("transformer")
class TformerTrajEncoder(TrajEncoder):
    r"""Transformer Trajectory Encoder.

    A pre-norm Transformer decoder-only model that processes sequences of timestep
    embeddings.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. The max context length
            of the model during training.

    Keyword Args:
        d_model: Dimension of the main residual stream and output. Defaults to
            256.
        n_heads: Number of self-attention heads. Each head has dimension
            d_model/n_heads. Defaults to 8.
        d_ff: Dimension of feed-forward network in residual blocks. Defaults to
            4*d_model.
        n_layers: Number of Transformer layers. Defaults to 3.
        dropout_ff: Dropout rate for linear layers within Transformer. Defaults to
            0.05.
        dropout_emb: Dropout rate for input embedding (combined input sequence and
            position embeddings passed to Transformer). Defaults to 0.05.
        dropout_attn: Dropout rate for attention matrix. Defaults to 0.00.
        dropout_qkv: Dropout rate for query/key/value projections. Defaults to
            0.00.
        activation: Activation function. Defaults to "leaky_relu".
        norm: Normalization function. Defaults to "layer" (LayerNorm).
        pos_emb: Position embedding type. "fixed" (default) uses sinusoidal
            embeddings, "learned" uses trainable embeddings per timestep.
        causal: Whether to use causal attention mask. Defaults to True.
        sigma_reparam: Whether to use :math:`\sigma`-reparam feed-forward layers
            from https://arxiv.org/abs/2303.06296. Defaults to True.
        normformer_norms: Whether to use extra norm layers from NormFormer
            (https://arxiv.org/abs/2110.09456). Always uses pre-norm Transformer.
        head_scaling: Whether to use head scaling from NormFormer. Defaults to
            True.
        attention_type: Attention layer type. Defaults to
            transformer.FlashAttention. transformer.VanillaAttention provided as
            backup. New types can inherit from transformer.SelfAttention.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 3,
        dropout_ff: float = 0.05,
        dropout_emb: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        activation: str = "leaky_relu",
        norm: str = "layer",
        pos_emb: str = "fixed",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
        head_scaling: bool = True,
        attention_type: type[transformer.SelfAttention] = transformer.FlashAttention,#TODO rollback to FlashAttention after debugging
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.d_model = d_model

        def make_layer():
            return transformer.TransformerLayer(
                attention_layer=transformer.AttentionLayer(
                    self_attention=attention_type(causal=True, dropout=dropout_attn),
                    d_model=self.d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                    head_scaling=head_scaling,
                    sigma_reparam=sigma_reparam,
                ),
                d_model=self.d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
                sigma_reparam=sigma_reparam,
                normformer_norms=normformer_norms,
            )

        layers = [make_layer() for _ in range(self.n_layers)]
        self.tformer = transformer.Transformer(
            inp_dim=tstep_dim,
            d_model=self.d_model,
            layers=layers,
            dropout_emb=dropout_emb,
            norm=norm,
            pos_emb=pos_emb,
        )

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> transformer.TformerHiddenState:
        def make_cache():
            dtype = (
                torch.bfloat16
                if self.attention_type == transformer.FlashAttention
                else torch.float32
            )
            return transformer.Cache(
                device=device,
                dtype=dtype,
                layers=self.n_layers,
                batch_size=batch_size,
                max_seq_len=self.max_seq_len,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
            )

        hidden_state = transformer.TformerHiddenState(
            key_cache=make_cache(),
            val_cache=make_cache(),
            seq_lens=torch.zeros((batch_size,), dtype=torch.int32, device=device),
        )
        return hidden_state

    def reset_hidden_state(
        self, hidden_state: Optional[transformer.TformerHiddenState], dones: np.ndarray
    ) -> Optional[transformer.TformerHiddenState]:
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, transformer.TformerHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[transformer.TformerHiddenState] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[transformer.TformerHiddenState]]:
        assert time_idxs is not None
        return self.tformer(seq, pos_idxs=time_idxs, hidden_state=hidden_state)

    @property
    def emb_dim(self) -> int:
        return self.d_model


try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, norm: str):
        super().__init__()
        self.norm = ff.Normalization(norm, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, seq):
        return seq + self.mamba(self.norm(seq))

    def step(self, seq, conv_state, ssm_state):
        res, new_conv_state, new_ssm_state = self.mamba.step(
            self.norm(seq), conv_state, ssm_state
        )
        return seq + res, new_conv_state, new_ssm_state


class _MambaHiddenState:
    def __init__(self, conv_states: list[torch.Tensor], ssm_states: list[torch.Tensor]):
        assert len(conv_states) == len(ssm_states)
        self.n_layers = len(conv_states)
        self.conv_states = conv_states
        self.ssm_states = ssm_states

    def reset(self, idxs):
        for i in range(self.n_layers):
            # hidden states are initialized to zero
            self.conv_states[i][idxs] = 0.0
            self.ssm_states[i][idxs] = 0.0

    def __getitem__(self, layer_idx: int):
        assert layer_idx < self.n_layers
        return self.conv_states[layer_idx], self.ssm_states[layer_idx]

    def __setitem__(self, layer_idx: int, conv_ssm: tuple[torch.Tensor]):
        conv, ssm = conv_ssm
        self.conv_states[layer_idx] = conv
        self.ssm_states[layer_idx] = ssm


class _MATEFFNBlock(nn.Module):
    """FFN sub-layer of a Transformer block, without self-attention.

    Mirrors the FFN part of :class:`~amago.nets.transformer.TransformerLayer`
    exactly: pre-norm → activation → optional NormFormer norm → dropout →
    residual.  Used by :class:`MateTrajEncoder` so that the per-step processing
    matches a Transformer up to (but not including) the attention sub-layer.

    Args:
        d_model: Residual stream dimension.
        d_ff: Inner (expanded) dimension.

    Keyword Args:
        dropout_ff: Dropout applied after the second linear layer. Defaults to
            0.05.
        activation: Activation function name. Defaults to ``"leaky_relu"``.
        norm: Normalization method for pre-norm and NormFormer norm. Defaults
            to ``"layer"``.
        sigma_reparam: Use SigmaReparam linear layers (same as Transformer).
            Defaults to ``True``.
        normformer_norms: Add extra NormFormer norm after the first activation
            (same as Transformer). Defaults to ``True``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_ff: float = 0.05,
        activation: str = "leaky_relu",
        norm: str = "layer",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
    ):
        super().__init__()
        FF = transformer.SigmaReparam if sigma_reparam else nn.Linear
        self.ff1 = FF(d_model, d_ff)
        self.ff2 = FF(d_ff, d_model)
        self.norm1 = ff.Normalization(norm, d_model)   # pre-norm  (= norm3 in TransformerLayer)
        self.norm2 = (
            ff.Normalization(norm, d_ff) if normformer_norms else nn.Identity()
        )  # NormFormer extra norm  (= norm4 in TransformerLayer)
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = utils.activation_switch(activation)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        q = self.norm2(self.activation(self.ff1(q)))
        q = self.dropout_ff(self.ff2(q))
        return x + q


@gin.configurable
@register_traj_encoder("mate")
class MateTrajEncoder(TrajEncoder):
    r"""MATE (Memory via Additive TransiTion Embeddings) Trajectory Encoder.

    Shares the same preprocessing and per-step FFN structure as
    :class:`TformerTrajEncoder`, but **replaces self-attention** with a causal
    cumulative sum as the cross-timestep memory mechanism, followed by an output
    projection.

    **Structure compared to Transformer:**

    - *Shared* (attention 전까지 동일):
        - ``tstep_dim → d_model`` linear projection.
        - Positional embedding (fixed sinusoidal or learnable).
        - Input dropout.
        - N × :class:`_MATEFFNBlock` — the FFN sub-layer of a
          :class:`~amago.nets.transformer.TransformerLayer` (pre-norm,
          SigmaReparam, NormFormer norm, residual) without self-attention.
    - *Replaced* (attention layer → MATE memory):
        - Causal cumulative sum :math:`h_t = \sum_{i=1}^{t} z_i` over the
          per-step FFN outputs.
    - *Added* (output projection):
        - ``"hyper"`` (default) — projects onto a learnable hypersphere:
          :math:`(h_t + e) / \|h_t + e\| \cdot \sqrt{d}`.
        - ``"mean"`` — running mean: :math:`h_t / t`.

    Args:
        tstep_dim: Dimension of the input timestep representation.
        max_seq_len: Maximum sequence length.

    Keyword Args:
        d_model: Hidden/output dimension. Defaults to 256.
        n_layers: Number of :class:`_MATEFFNBlock` blocks (= Transformer depth
            without attention). Defaults to 2.
        d_ff: Inner dimension of each FFN block. Defaults to ``4 * d_model``.
        dropout_ff: Dropout inside FFN blocks. Defaults to 0.05.
        dropout_emb: Dropout on the input embedding
            (projection + positional embedding). Defaults to 0.05.
        activation: Activation function. Defaults to ``"leaky_relu"``.
        norm: Normalization for per-block norms and final norm. Defaults to
            ``"layer"``.
        pos_emb: Positional embedding type — ``"fixed"`` (sinusoidal, default)
            or ``"learnable"``.
        sigma_reparam: Use SigmaReparam linear layers, same as
            :class:`TformerTrajEncoder`. Defaults to ``True``.
        normformer_norms: Add NormFormer extra norms inside FFN blocks, same as
            :class:`TformerTrajEncoder`. Defaults to ``True``.
        proj: Output projection applied to the cumulative-sum output.
            ``"hyper"`` (default) applies hypersphere projection with a
            learnable offset ``init_emb``.  ``"mean"`` divides by the
            cumulative step count :math:`t` to yield a running mean.
        pos_emb: Positional embedding type to inject before the FFN blocks.
            ``"none"`` (default) disables positional embedding entirely — this
            matches the original MATE design, because the causal cumulative sum
            already encodes order implicitly (h_t = z_1 + ... + z_t, so earlier
            tokens contribute more to later states).  ``"fixed"`` (sinusoidal)
            or ``"learnable"`` can be enabled for experimental comparison with
            the Transformer, where pos_emb is required to make attention
            position-aware.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout_ff: float = 0.05,
        dropout_emb: float = 0.05,
        activation: str = "leaky_relu",
        norm: str = "layer",
        pos_emb: str = "none",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
        proj: str = "hyper",
        obs_shortcut: bool = False,
        obs_shortcut_dim: Optional[int] = None,
    ):
        super().__init__(tstep_dim, max_seq_len)
        assert proj in ("hyper", "mean"), (
            f"proj must be 'hyper' or 'mean', got '{proj}'"
        )
        assert pos_emb in ("none", "fixed", "learnable"), (
            f"pos_emb must be 'none', 'fixed', or 'learnable', got '{pos_emb}'"
        )
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.proj = proj
        self.obs_shortcut = obs_shortcut

        # --- Preprocessing: same as Transformer (pos_emb optional) ---
        self.inp = nn.Linear(tstep_dim, d_model)
        if pos_emb == "none":
            self.position_embedding = None
        elif pos_emb == "fixed":
            self.position_embedding = transformer.FixedPosEmb(d_model)
        elif pos_emb == "learnable":
            self.position_embedding = transformer.LearnablePosEmb(d_model)
        self.dropout_emb = nn.Dropout(dropout_emb)

        # --- Per-step FFN blocks (TransformerLayer FFN sub-layer, no attention) ---
        self.blocks = nn.ModuleList(
            [
                _MATEFFNBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    activation=activation,
                    norm=norm,
                    sigma_reparam=sigma_reparam,
                    normformer_norms=normformer_norms,
                )
                for _ in range(n_layers)
            ]
        )

        # --- Projection ---
        # init_emb is used by both "hyper" and "mean" projections.
        self.init_emb = nn.Parameter(torch.randn(d_model))

        # --- Obs shortcut (mirrors Memory-RL's obs_shortcut / observ_embedder) ---
        # Projects only the *current* observation (non-_prev_ keys) to d_model and
        # concatenates with the MATE memory output, so the actor/critic sees both
        # the current observation directly and the cumulative memory state.
        # emb_dim doubles to 2 * d_model.
        if obs_shortcut:
            # obs_shortcut_dim is the flat dim of current obs (non-_prev_ keys),
            # passed from BaseAgent.init_encoders where obs_space is known.
            # Falls back to tstep_dim if not provided (e.g. standalone usage).
            shortcut_in = obs_shortcut_dim if obs_shortcut_dim is not None else tstep_dim
            self.shortcut_proj = nn.Linear(shortcut_in, d_model)
            self._emb_dim = d_model * 2
        else:
            self._emb_dim = d_model

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    def _preprocess(self, seq: torch.Tensor, time_idxs: Optional[torch.Tensor]) -> torch.Tensor:
        """Input projection + optional positional embedding + dropout."""
        x = self.inp(seq)
        if self.position_embedding is not None:
            x = x + self.position_embedding(time_idxs.squeeze(-1))
        return self.dropout_emb(x)

    def _ffn_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Per-step FFN blocks — Transformer FFN sub-layer without attention."""
        for block in self.blocks:
            x = block(x)
        return x

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sliding window ring buffer — mirrors Transformer KV-cache.
        # Stores the last max_seq_len post-FFN embeddings so the cumsum
        # seen at inference stays within the distribution seen during training.
        # (z_ring, ring_sum, ptr, count)
        #   z_ring  : (B, max_seq_len, d_model) — circular buffer of z vectors
        #   ring_sum: (B, d_model)              — running sum of valid entries
        #   ptr     : (B,) int64               — next write position
        #   count   : (B,) int64               — valid entries (capped at max_seq_len)
        return (
            torch.zeros((batch_size, self.max_seq_len, self.d_model), dtype=torch.float32, device=device),
            torch.zeros((batch_size, self.d_model), dtype=torch.float32, device=device),
            torch.zeros((batch_size,), dtype=torch.int64, device=device),
            torch.zeros((batch_size,), dtype=torch.int64, device=device),
        )

    def reset_hidden_state(
        self,
        hidden_state: Optional[Tuple],
        dones: np.ndarray,
    ) -> Optional[Tuple]:
        if hidden_state is None:
            return None
        z_ring, ring_sum, ptr, count = hidden_state
        z_ring[dones] = 0.0
        ring_sum[dones] = 0.0
        ptr[dones] = 0
        count[dones] = 0
        return (z_ring, ring_sum, ptr, count)

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple] = None,
        log_dict: Optional[dict] = None,
        obs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # seq: (B, T, tstep_dim)
        assert self.position_embedding is None or time_idxs is not None, (
            "time_idxs is required when pos_emb != 'none'."
        )

        # Preprocessing: inp projection + optional pos emb + dropout
        x = self._preprocess(seq, time_idxs)         # (B, T, d_model)

        # Per-step FFN blocks  (Transformer FFN sub-layer, no attention)
        z = self._ffn_blocks(x)                       # (B, T, d_model)

        # Cumulative sum: causal prefix-sum replaces attention as memory mechanism
        T = seq.shape[1]
        local_counts = torch.arange(1, T + 1, device=seq.device, dtype=z.dtype).view(1, T, 1)

        if hidden_state is None:
            # Training: standard prefix-sum over the sampled window.
            cumsum = z.cumsum(dim=1)       # (B, T, d_model)
            counts = local_counts          # (1, T, 1)
            new_hidden_state = None
        else:
            # Inference: sliding window cumsum — mirrors Transformer KV-cache roll_back.
            z_ring, ring_sum, ptr, count = hidden_state
            batch_idx = torch.arange(z.shape[0], device=z.device)

            cumsum_steps = []
            count_steps  = []
            for t_offset in range(T):
                z_t = z[:, t_offset, :]  # (B, D)

                # When ring is full, subtract the entry about to be overwritten.
                is_full = (count >= self.max_seq_len).float().unsqueeze(-1)  # (B, 1)
                oldest  = z_ring[batch_idx, ptr]                             # (B, D)
                ring_sum = ring_sum - oldest * is_full + z_t

                # Write z_t into the ring at the current write head.
                z_ring[batch_idx, ptr] = z_t.detach()

                # Advance write pointer (circular) and update count.
                ptr   = (ptr + 1) % self.max_seq_len
                count = torch.clamp(count + 1, max=self.max_seq_len)

                cumsum_steps.append(ring_sum.unsqueeze(1))        # (B, 1, D)
                count_steps.append(count.view(-1, 1, 1).float())  # (B, 1, 1)

            cumsum = torch.cat(cumsum_steps, dim=1)  # (B, T, D)
            counts = torch.cat(count_steps,  dim=1)  # (B, T, 1)
            new_hidden_state = (z_ring, ring_sum, ptr, count)

        if self.proj == "hyper":
            # Hypersphere projection: (cumsum + init_emb) normalized to hypersphere.
            # No LayerNorm on cumsum: cumsum magnitude grows with t, which lets
            # init_emb dominate early steps and actual memory dominate later —
            # matching the original Memory-RL MATE (no norm before projection).
            output = cumsum + self.init_emb
            output = output / output.norm(dim=-1, keepdim=True).clamp(min=1e-6) * np.sqrt(self.d_model)
        elif self.proj == "mean":
            # Running mean: (cumsum + init_emb) / t  — matches Memory-RL original formula.
            # Do NOT apply out_norm here: LayerNorm would fix the cumsum magnitude to ~const,
            # making the output shrink toward 0 as t grows and destroying the temporal signal.
            output = cumsum + self.init_emb
            output = output / counts

        if self.obs_shortcut:
            # shortcut_proj and emb_dim are set but the actual concat is done
            # by BaseAgent._apply_traj_encoder, which has access to the raw obs
            # dict at all call sites. Nothing to do here.
            pass

        return output, new_hidden_state


@gin.configurable
@register_traj_encoder("mamba")
class MambaTrajEncoder(TrajEncoder):
    """Mamba Trajectory Encoder.

    Implementation of the Mamba architecture from "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces" (https://arxiv.org/abs/2312.00752).

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. The max context length
            of the model during training.

    Keyword Args:
        d_model: Dimension of the main residual stream and output, analogous to
            the d_model in a Transformer. Defaults to 256.
        d_state: Dimension of the SSM in Mamba blocks. Defaults to 16.
        d_conv: Dimension of the convolution layer in Mamba blocks. Defaults to 4.
        expand: Expansion factor of the SSM. Defaults to 2.
        n_layers: Number of Mamba blocks. Defaults to 3.
        norm: Normalization function. Defaults to "layer" (LayerNorm).

    References:
        - https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        - https://github.com/johnma2006/mamba-minimal/tree/master
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 3,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len)

        assert (
            Mamba is not None
        ), "Missing Mamba installation (pip install amago[mamba])"
        self.inp = nn.Linear(tstep_dim, d_model)

        self.mambas = nn.ModuleList(
            [
                _MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = ff.Normalization(norm, d_model)
        self._emb_dim = d_model

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> _MambaHiddenState:
        conv_states, ssm_states = [], []
        for mamba_block in self.mambas:
            conv_state, ssm_state = mamba_block.mamba.allocate_inference_cache(
                batch_size, max_seqlen=self.max_seq_len
            )
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        return _MambaHiddenState(conv_states, ssm_states)

    def reset_hidden_state(
        self, hidden_state: Optional[_MambaHiddenState], dones: np.ndarray
    ) -> Optional[_MambaHiddenState]:
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, _MambaHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    @torch.compile
    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: Optional[torch.Tensor] = None,
        hidden_state: Optional[_MambaHiddenState] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[_MambaHiddenState]]:
        seq = self.inp(seq)
        if hidden_state is None:
            for mamba in self.mambas:
                seq = mamba(seq)
        else:
            assert not self.training
            assert isinstance(hidden_state, _MambaHiddenState)
            for i, mamba in enumerate(self.mambas):
                conv_state_i, ssm_state_i = hidden_state[i]
                seq, new_conv_state_i, new_ssm_state_i = mamba.step(
                    seq, conv_state=conv_state_i, ssm_state=ssm_state_i
                )
                hidden_state[i] = new_conv_state_i, new_ssm_state_i
        return self.out_norm(seq), hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim
