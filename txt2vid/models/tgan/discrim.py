import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discrim(nn.Module):
    def __init__(self, sequence_first=True, in_channels=3, mid_ch=64):
        super().__init__()
        self.c0 = nn.Conv3d(in_channels, mid_ch, 4, 2, 1)
        self.c1 = nn.Conv3d(mid_ch, mid_ch * 2, 4, 2, 1)
        self.c2 = nn.Conv3d(mid_ch * 2, mid_ch * 4, 4, 2, 1)
        self.c3 = nn.Conv3d(mid_ch * 4, mid_ch * 8, 4, 2, 1)
        self.bn0 = nn.BatchNorm3d(mid_ch)
        self.bn1 = nn.BatchNorm3d(mid_ch * 2)
        self.bn2 = nn.BatchNorm3d(mid_ch * 4)
        self.bn3 = nn.BatchNorm3d(mid_ch * 8)

    def forward(self, x=None, cond=None, xbar=None):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = h.view(h.size(0), -1)
        return torch.mean(h, 1)
   
if __name__ == "__main__":
    x = Variable(torch.randn(64, 3, 16, 64, 64))
    discr = Discrim(sequence_first=False)
    out = discr(x)
    print(out.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(discr))
