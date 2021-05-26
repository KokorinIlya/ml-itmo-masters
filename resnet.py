from torch import nn
import torch.nn.functional as F


def make_conv(channels):
    return nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1)


class SimpleBlock(nn.Module):
    def __init__(self, channels):
        super(SimpleBlock, self).__init__()
        self.conv1 = make_conv(channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = make_conv(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.bn2(self.conv2(c1))
        return F.relu(x + c2)


class DownBlock(nn.Module):
    def __init__(self, in_chan):
        super(DownBlock, self).__init__()
        self.out_chan = in_chan * 2
        self.conv1 = nn.Conv2d(in_chan, self.out_chan, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = make_conv(self.out_chan)
        self.bn2 = nn.BatchNorm2d(self.out_chan)
        self.conv_down = nn.Conv2d(in_chan, self.out_chan, kernel_size=(1, 1), stride=2, bias=False)
        self.bn_down = nn.BatchNorm2d(self.out_chan)

        self.bn_down.weight.data.fill_(1)
        self.bn_down.bias.data.fill_(0)

        self.bn1 = nn.BatchNorm2d(self.out_chan)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.bn2(self.conv2(c1))
        down = self.bn_down(self.conv_down(x))
        return F.relu(down + c2)


class ResNet(nn.Module):
    def __init__(self, n):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.seq32_32 = nn.Sequential()
        for i in range(n):
            self.seq32_32.add_module(str(i), SimpleBlock(16))

        self.seq16_16 = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.seq16_16.add_module(str(i), DownBlock(16))
            else:
                self.seq16_16.add_module(str(i), SimpleBlock(32))

        self.seq8_8 = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.seq8_8.add_module(str(i), DownBlock(32))
            else:
                self.seq8_8.add_module(str(i), SimpleBlock(64))

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        s32_32 = self.seq32_32(x)
        s16_16 = self.seq16_16(s32_32)
        s8_8 = self.seq8_8(s16_16)

        features = F.avg_pool2d(s8_8, (8, 8))
        flat = features.view(features.size()[0], -1)

        return self.fc(flat)
