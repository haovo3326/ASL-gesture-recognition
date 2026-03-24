import torch.nn as nn

class Classifier(nn.Module):
    KEYPOINTS_FLATTEN = 63  # 21 3-vector of keypoints
    ALPHABETS = 26          # 26 letters

    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            # nn.Linear(128, 256),
            # nn.ReLU(),
            #
            # nn.Linear(256, 128),
            # nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)