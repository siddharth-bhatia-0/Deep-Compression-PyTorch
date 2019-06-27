import math

import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict



class Flat(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, modules):
        super(Flat, self).__init__()

        for name, module in modules:
            self.add_module(name, module)


    def forward(self, x):

        for name,module in self.named_children():
            x = module.forward(x)

        return x
