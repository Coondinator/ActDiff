import torch
from torch import nn, Tensor
from .layers import PositionalEncoding, TimestepEmbedder, FinalLayer, DiTBlock, CrossDiTBlock


class TransformerModel(nn.Module):

    def __init__(self, act_dim: int, con_dim: int, nhead: int, hid_dim: int, nlayers: int, dropout: float = 0.2):
        super().__init__()
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoder = PositionalEncoding(hid_dim, dropout)

        self.t_embedder = TimestepEmbedder(self.hid_dim)
        self.label_embedder = TimestepEmbedder(self.hid_dim)

        self.linear = nn.Linear(self.act_dim, self.hid_dim)
        self.text_embed_proj = nn.Sequential(
            nn.ReLU(), nn.Linear(768, self.hid_dim))
        # self.text_embed_proj = nn.Sequential(nn.ReLU(), nn.Linear(768, h_dim))0
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([
            CrossDiTBlock(x_hidden_size=self.hid_dim, y_hidden_size=con_dim, num_head=nhead, mlp_ratio=4.0) for _ in range(nlayers)
        ])
        self.final_layer = FinalLayer(self.hid_dim, self.act_dim)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation_x[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_x[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_y[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_y[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: Tensor, t: Tensor = None, cond_act: Tensor = None, label: Tensor = None,
                img = None, point = None, attn_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # x = self.dropout(self.activation(self.linear(x)))
        x = self.pos_encoder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        con_act = self.pos_encoder(cond_act)
        t = self.t_embedder(t)  # (N, D)  # (N, D)
        label_emb = self.label_embedder(label)
        t = t + label_emb

        for block in self.blocks:
            x = block(x=x, y=con_act, c=t)  # (N, T, D)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        return x


