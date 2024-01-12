import torch.nn as nn


# Define a custom neural network class named PotatoNet that inherits from nn.Module.
class PotatoNet(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(PotatoNet, self).__init__()

        # Define a convolutional neural network (CNN) module for image feature extraction.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define a recurrent neural network (RNN) module using LSTM.
        self.rnn = nn.LSTM(
            input_size=64 * 8 * 8,  # Input size based on CNN output
            hidden_size=hidden_size,
            num_layers=num_layers,  # Number of LSTM layers
            batch_first=True
        )

        # Define a fully connected (linear) layer for classification.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get the dimensions of the input tensor
        batch_size, timesteps, C, H, W = x.size()

        # Reshape the input tensor to fit the CNN
        x = x.view(batch_size * timesteps, C, H, W)  # (batch_size * timesteps, C, H, W)

        # Pass the input through the CNN
        x = self.cnn(x)  # (batch_size * timesteps, 64, 8, 8)

        # Reshape the output from CNN for RNN input
        x = x.view(batch_size, timesteps, -1)  # (batch_size, timesteps, 64 * 8 * 8)

        # Pass the CNN output through the RNN
        x, _ = self.rnn(x)  # (batch_size, timesteps, hidden_size)

        # Select the final time-step output from RNN
        x = x[:, -1, :]  # (batch_size, hidden_size)

        # Pass the RNN output through the fully connected layer for classification
        x = self.fc(x)  # (batch_size, num_classes)

        return x
