import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            # First conv block (reducing channels to 16)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Stride 2 downsampling
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Stride 2 downsampling
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Global Average Pooling (reduces parameters)
            nn.AdaptiveAvgPool2d(1),

            # Fully connected layers
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 10)  # Output layer for 10 classes
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
