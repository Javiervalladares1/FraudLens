import numpy as np
import torch
from torch import nn

FEATURES=[f'V{i}' for i in range(1,29)]+['logAmount']

class FraudMLP(nn.Module):
    def __init__(self,hidden=(32,16),dropout=0.0):
        super().__init__(); layers=[]; d=29
        for h in hidden:
            layers.extend([nn.Linear(d,h),nn.ReLU()])
            if dropout: layers.append(nn.Dropout(dropout))
            d=h
        layers.append(nn.Linear(d,1)); self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x).squeeze(-1)

def feature_array(frame):
    x=frame[[f'V{i}' for i in range(1,29)]].to_numpy(dtype=np.float64)
    return np.column_stack([x,np.log1p(frame['Amount'].to_numpy(dtype=np.float64))])

def export_network(model):
    return [dict(weight=m.weight.detach().numpy().tolist(),bias=m.bias.detach().numpy().tolist())
            for m in model.layers if isinstance(m,nn.Linear)]
