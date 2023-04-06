import torch.nn as nn
import copy

# return N number of modules
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])