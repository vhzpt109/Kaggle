from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(8, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        prediction = self.layer(x)
        return prediction
