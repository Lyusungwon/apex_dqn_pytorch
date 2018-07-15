import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, history_size, hidden_size, action_size):
        super(DQN, self).__init__()
        self.history_size = history_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.conv1 = nn.Conv2d(self.history_size, self.hidden_size, 8, 4)
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size * 2, 4, 2)
        self.conv3 = nn.Conv2d(self.hidden_size * 2, self.hidden_size * 2, 3, 1)
        self.fc = 84 * self.hidden_size * 2
        self.vfc1 = nn.Linear(self.fc, 512)
        self.vfc2 = nn.Linear(512, 1)
        self.afc1 = nn.Linear(self.fc, 512)
        self.afc2 = nn.Linear(512, self.action_size)

        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        torch.nn.init.normal_(self.vfc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.vfc2.weight, 0, 0.02)
        torch.nn.init.normal_(self.afc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.afc2.weight, 0, 0.02)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc)
        a = F.relu(self.afc1(x))
        a = self.afc2(a)
        av = torch.mean(a, 1, True)
        av = av.expand_as(a)
        v = F.relu(self.vfc1(x))
        v = self.vfc2(v)
        v = v.expand_as(a)
        x = a - av + v
        return x
