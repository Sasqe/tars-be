import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),

            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),

            # Flatten for fully connected layers
            nn.Flatten(),

            # Fully connected layers
            nn.Linear(128 * 7 * 7, 256),  # Output size after 2x max-pooling is 7x7
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),

            nn.Linear(256, 10)  # Output layer for 10 classes (digits 0-9)
        )

    def forward(self, x):
        return self.model(x)


# Test the model
if __name__ == "__main__":
    model = Net()
    print(model)
    sample_input = torch.randn(1, 1, 28, 28)  # Batch size of 1, grayscale 28x28 image
    output = model(sample_input)
    print("Output shape:", output.shape)  # Should output [1, 10]
