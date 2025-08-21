import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The model is defined as a sequential container of layers
        self.model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Downsampling conv block (with stride 2)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # outputs 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Second convolutional block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Downsampling conv block (with stride 2)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # outputs 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Global Average Pooling to reduce spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 10)
        )
        # Note: Increased initial conv filters to 32 (from 16) and second conv to 64 (from 32)
        # for richer feature maps. Kept final conv output at 64 channels to limit model size.
        # Reordered layers to Conv -> BatchNorm -> ReLU for stability:contentReference[oaicite:8]{index=8}.
        # Retained a dropout before the final layer to help prevent overfitting:contentReference[oaicite:9]{index=9}.

    def forward(self, x):
        return self.model(x)

# Test the model architecture
if __name__ == "__main__":
    model = Net()
    print(model)
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected [1, 10]

