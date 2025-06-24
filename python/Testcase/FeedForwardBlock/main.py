import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config

class ManualDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.scale = 1.0 - dropout

    def forward(self, x):
        return x / self.scale

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = ManualDropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))

    def forward(self, x):
        # (batch, seq, d_model) --> (batch, seq, d_ff) --> (batch, seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    def getParam(self, prefix, dict) :
        w1 = self.linear_1.weight.detach().cpu().numpy().T.copy() # (out, in)
        b1 = self.linear_1.bias.detach().cpu().numpy()
        w2 = self.linear_2.weight.detach().cpu().numpy().T.copy()
        b2 = self.linear_2.bias.detach().cpu().numpy()
        dict[prefix + ".w1"] = w1
        dict[prefix + ".b1"] = b1
        dict[prefix + ".w2"] = w2
        dict[prefix + ".b2"] = b2

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]
        d_model = self.linear_1.in_features 

        x = torch.randn(batch_size, seq, d_model)
        with torch.no_grad():
            y = self.forward(x)

        dict[prefix + ".input"] = x.detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".output1"] = self.linear_1(x).detach().cpu().numpy()

    def getOriginalParam(self, prefix, dict) :
        original_w1 = self.linear_1.weight.detach().clone().cpu().numpy()
        original_b1 = self.linear_1.bias.detach().clone().cpu().numpy()
        original_w2 = self.linear_2.weight.detach().clone().cpu().numpy()
        original_b2 = self.linear_2.bias.detach().clone().cpu().numpy()
        dict[prefix + ".original_w1"] = original_w1
        dict[prefix + ".original_b1"] = original_b1
        dict[prefix + ".original_w2"] = original_w2
        dict[prefix + ".original_b2"] = original_b2
    
    def getUpdatedParam(self, prefix, dict) :
        updated_w1 = self.linear_1.weight.detach().clone().cpu().numpy()
        updated_b1 = self.linear_1.bias.detach().clone().cpu().numpy()
        updated_w2 = self.linear_2.weight.detach().clone().cpu().numpy()
        updated_b2 = self.linear_2.bias.detach().clone().cpu().numpy()
        dict[prefix + ".updated_w1"] = updated_w1
        dict[prefix + ".updated_b1"] = updated_b1
        dict[prefix + ".updated_w2"] = updated_w2
        dict[prefix + ".updated_b2"] = updated_b2

    def getBackward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]

        x = torch.randn(batch_size, seq, self.linear_1.in_features)
        
        self.getOriginalParam(prefix, dict)

        y = self.forward(x)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)
        
        dict[prefix + ".input"] = x.numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()
            
def save_numpy_dict(filename: str, np_dict: dict):
    np.savez(filename, **np_dict)


model = FeedForwardBlock(get_config()["d_model"], get_config()["d_ff"], 0.1)

paramDict = {}
forwardDict = {}
backwardDict = {}

model.getParam("feedforward", paramDict)
save_numpy_dict("feedforward_param.npz",paramDict)


for i in range(5) :
    model.getForward("feedforward", forwardDict)
    save_numpy_dict(f"feedforward_forward{i}.npz", forwardDict)

for i in range(5):
    model.getBackward("feedforward", backwardDict)
    save_numpy_dict(f"feedforward_backward{i}.npz", backwardDict)