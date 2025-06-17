import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(features))  # learnable scale
        self.bias = nn.Parameter(torch.randn(features))  # learnable shift
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]), eps =1e-9)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    def getParam(self, prefix, dict) :
        dict[prefix + ".alpha"] = self.alpha.detach().cpu().numpy()
        dict[prefix + ".bias"] = self.bias.detach().cpu().numpy()

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.alpha.shape[0]

        x = torch.randn(batch_size, seq, d_model)
        with torch.no_grad():
            y = self.forward(x)
        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        original_alpha = self.alpha.detach().clone().cpu().numpy()
        original_bias = self.bias.detach().clone().cpu().numpy()
        dict[prefix + ".original_alpha"] = original_alpha
        dict[prefix + ".original_bias"] = original_bias
    
    def getUpdatedParam(self, prefix, dict) :
        updated_alpha = self.alpha.detach().clone().cpu().numpy()
        updated_bias = self.bias.detach().clone().cpu().numpy()
        dict[prefix + ".updated_alpha"] = updated_alpha
        dict[prefix + ".updated_bias"] = updated_bias

    def getBackward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        hidden_size = self.alpha.shape[0]

        x = torch.randn(batch_size, seq, hidden_size)

        self.getOriginalParam(prefix, dict)
        self.optimizer.zero_grad()
        y = self.forward(x)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()


def save_numpy_dict(filename: str, np_dict: dict):
    np.savez(filename, **np_dict)


model = LayerNormalization(get_config()["d_model"], 1e-9)

paramDict = {}
forwardDict = {}
backwardDict = {}

model.getParam("layerNorm", paramDict)
save_numpy_dict("layer_norm_param.npz",paramDict)


for i in range(5) :
    model.getForward("layerNorm", forwardDict)
    save_numpy_dict(f"layer_norm_forward{i}.npz", forwardDict)

for i in range(5):
    model.getBackward("layerNorm", backwardDict)
    save_numpy_dict(f"layer_norm_backward{i}.npz", backwardDict)