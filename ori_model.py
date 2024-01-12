import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class PoseModel(nn.Module):
    def __init__(self):
        super(PoseModel, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers of ResNet to prevent from being updated during training
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the fully connected layer of ResNet
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.rnn = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=3)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        out = self.resnet(x)  # (batch_size, 2048, 1, 1)

        out = out.view(out.size(0), -1, 2048)  # (batch_size, 1, 2048)

        out, _ = self.rnn(out)  # (batch_size, 1, 64)

        out = self.fc1(out[:, -1, :])  # (batch_size, 128)

        out = self.relu(out)  # (batch_size, 128)

        out = self.fc2(out)  # (batch_size, 3)

        return out

