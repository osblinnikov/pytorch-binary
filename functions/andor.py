# functions/add.py
import torch
from torch.autograd import Function
from _ext import andor


class MyAddFunction(Function):
    def forward(self, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            andor.my_lib_add_forward(input1, input2, output)
        else:
            andor.my_lib_add_forward_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            andor.my_lib_add_backward(grad_output, grad_input)
        else:
            andor.my_lib_add_backward_cuda(grad_output, grad_input)
        return grad_input
