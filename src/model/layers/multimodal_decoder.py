import torch.nn as nn


class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        
        # 这里的6是代表K条轨迹，K=6
        self.multimodal_proj = nn.Linear(embed_dim, 6 * embed_dim)
        
        # 位置预测头
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, future_steps * 2),
        )
        # 置信度预测头
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, 6, self.embed_dim)
        loc = self.loc(x).view(-1, 6, self.future_steps, 2)
        pi = self.pi(x).squeeze(-1)

        return loc, pi
