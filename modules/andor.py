from torch.nn.modules.module import Module
from functions.andor import AndOrFunction

class AndOrModule(Module):
    def forward(self, input1, input2):
        return AndOrFunction()(input1, input2)
