import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.andor import AndOrModule

class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.andor = AndOrModule()

    def forward(self, input1, input2):
        return self.andor(input1, input2)

model = TestNetwork()
x = torch.range(1, 4).view(2, 2)
y = torch.range(1, 6).view(2, 3)
input1, input2 = Variable(x), Variable(y * 4)

if torch.cuda.is_available():
    input1, input2, = input1.cuda(), input2.cuda()
    print(input1)
    print(input2)
    print("matmul:", torch.matmul(input1, input2))
    res = model(input1, input2)
    print("model:", res, res.size())
