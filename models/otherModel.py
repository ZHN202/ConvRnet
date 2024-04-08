import torch.nn as nn


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1]

        out = self.linear(x2)

        return out

class Conv1d(nn.Module):
    def __init__(self):
        super(Conv1d, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(512 , 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)

        x2 = self.conv1d(x2).view(-1, 512)

        out = self.linear(x2)

        return out








