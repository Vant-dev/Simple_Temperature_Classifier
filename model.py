import torch
import torch.nn as nn

class TemperatureClassifier(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super().__init__()
        self.ln_1 = nn.Linear(inp_dim, hidden_dim)
        self.ln_out = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        out1 = self.gelu(self.ln_1(x))
        out2 = self.ln_out(out1)

        return out2
