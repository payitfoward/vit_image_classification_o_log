import torch
from torch import nn
from transformer import Transformer


class PositionalEmbedding1D(nn.Module):

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        return x + self.pos_embedding


class VisionTransformer(nn.Module):

    def __init__(
        self, 
        weights_path: str = None,
        patches: tuple = (16, 16),
        dim: int = 1024,
        ff_dim: int = 4096,
        num_heads: int = 16,
        num_layers: int = 24,
        dropout_rate: float = 0.1,
        in_channels: int = 3, 
        image_size: tuple = (384, 384),
        num_classes: int = 10,
    ):
        super().__init__()
        assert weights_path is not None
        self.image_size = image_size                

        h, w = image_size
        fh, fw = patches
        gh, gw = h // fh, w // fw
        seq_len = gh * gw + 1

        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        pre_logits_size = dim
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)
        state_dict = torch.load(weights_path)
        mismatch_keys = ['fc.weight', 'fc.bias']
        for key in mismatch_keys:
            state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)
        x = self.positional_embedding(x)
        x = self.transformer(x)
        x = self.norm(x)[:, 0]
        x = self.fc(x)
        return x

