import torch.nn as nn
import torch.nn.functional as F


# image input should be 416 x 128 x 6 (2 stacked rgb images)


class PoseEstimator(nn.Module):
    def __init__(self):
        super(PoseEstimator, self).__init__()
        self.conv16 = nn.Conv2d(6, 16, 7, padding=3)
        self.conv32 = nn.Conv2d(16, 32, 5)
        self.conv64 = nn.Conv2d(32, 64, 3)
        self.conv128 = nn.Conv2d(64, 128, 3)
        self.conv256_1 = nn.Conv2d(128, 256, 3)
        self.conv256_2 = nn.Conv2d(256, 256, 3)
        self.conv512 = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1_t = nn.Linear(24576, 512)
        self.fc1_r = nn.Linear(24576, 512)
        self.fc2_t = nn.Linear(512, 512)
        self.fc2_r = nn.Linear(512, 512)
        self.fc3_t = nn.Linear(512, 3)
        self.fc3_r = nn.Linear(512, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv16(x)))
        x = self.pool(F.relu(self.conv32(x)))
        x = self.pool(F.relu(self.conv64(x)))
        x = self.pool(F.relu(self.conv128(x)))
        x = F.relu(self.conv256_1(x))
        x = self.pool(F.relu(self.conv256_2(x)))
        x = self.pool(F.relu(self.conv512(x)))
        x = x.view(x.size(0), -1)
        t = F.relu(self.fc1_t(x))
        t = F.relu(self.fc2_t(t))
        t = F.relu(self.fc3_t(t))
        r = F.relu(self.fc1_r(x))
        r = F.relu(self.fc2_r(r))
        r = F.relu(self.fc3_r(r))
        return t, r


