import torch
from torch import nn, Tensor
from .model_utils import PositionalEncoding, TimestepEmbedder, FinalLayer
from .attention import DiTBlock


class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoder = PositionalEncoding(d_hid, dropout)
        self.t_embedder = TimestepEmbedder(d_hid)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_hid)
        self.text_embed_proj = nn.Sequential(
            nn.ReLU(), nn.Linear(768, d_hid))
        # self.text_embed_proj = nn.Sequential(nn.ReLU(), nn.Linear(768, h_dim))0
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([
            DiTBlock(d_hid, nhead, mlp_ratio=4.0) for _ in range(nlayers)
        ])
        self.final_layer = FinalLayer(d_hid, d_model)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: Tensor, t: Tensor = None, attn_mask: Tensor = None, text_emb: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # x = self.dropout(self.activation(self.linear(x)))
        x = self.pos_encoder(x)  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)  # (N, D)  # (N, D)

        text_emb = text_emb.squeeze(1)
        text_emb = self.text_embed_proj(text_emb)
        t = t + text_emb

        for block in self.blocks:
            x = block(x, t, attn_mask)  # (N, T, D)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        return x