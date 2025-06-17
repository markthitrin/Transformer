import torch
import torch.nn as nn
import math
import numpy as np
from config import get_config

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_config()["lr"], betas=(get_config()["beta1"], get_config()["beta2"]))
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

    def getParam(self, prefix, dict) :
        dict[prefix + ".weight"] = self.embedding.weight.detach().cpu().numpy()

    def getForward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]

        x = torch.randint(0, self.vocab_size, (batch_size, seq))
        with torch.no_grad() :
            y = self.forward(x)
        dict[prefix + ".input"] = x.float().detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()

    def getOriginalParam(self, prefix, dict) :
        original_weights = self.embedding.weight.detach().clone().cpu().numpy()
        dict[prefix + ".original_weights"] = original_weights
    
    def getUpdatedParam(self, prefix, dict) :
        updated_weights = self.embedding.weight.detach().clone().cpu().numpy()
        dict[prefix + ".updated_weights"] = updated_weights

    def getBackward(self, prefix, dict) :
        batch_size = get_config()["batch_size"]
        seq = get_config()["seq"]

        x = torch.randint(0, self.vocab_size, (batch_size, seq))

        self.getOriginalParam(prefix, dict)

        y = self.forward(x)
        loss = y.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getUpdatedParam(prefix, dict)

        dict[prefix + ".input"] = x.float().detach().numpy()
        dict[prefix + ".output"] = y.detach().numpy()
        dict[prefix + ".loss"] = loss.item()



def save_numpy_dict(filename: str, np_dict: dict):
    np.savez(filename, **np_dict)



    
