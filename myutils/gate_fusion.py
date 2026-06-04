import torch
import torch.nn as nn

class GatedFusionNetwork(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(GatedFusionNetwork, self).__init__()
        self.pathway1 = nn.Linear(input_size1, hidden_size)
        self.pathway2 = nn.Linear(input_size2, hidden_size)
        self.gating = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.output = nn.Linear(hidden_size, 1)
        self.tanh_f = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        out1 = self.tanh_f(self.pathway1(input1))
        out2 = self.tanh_f(self.pathway2(input2))
        fused = torch.cat((out1, out2), dim=-1)
        gate = self.gating(fused)
        gated_output = gate * out1 + (1 - gate) * out2
        return gated_output