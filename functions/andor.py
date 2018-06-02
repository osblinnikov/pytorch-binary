# functions/add.py
import torch
from torch.autograd import Function
from _ext import andor


class AndOrFunction(Function):
    def forward(self, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            andor.andor_forward(input1, input2, output)
        else:
            andor.andor_forward_cuda(input1, input2, output, input1.size()[0], input2.size()[0])
        return output