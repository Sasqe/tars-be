import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import Net  # Import the model
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Increase rotation range
    transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # Random scaling and translation
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Simulate perspective distortions
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.RandomRotation(5),  # Small rotation
    transforms.RandomAffine(0, translate=(0.05, 0.05)),  # Small translation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = Net().to(device)

# Define the loss function (CrossEntropyLoss for categorical cross-entropy)

criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


# Early stopping
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=30,
                model_save_path="best_model.pth"):
    early_stopping = EarlyStopping(patience=10, delta=0.001)
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(test_loader.dataset)
        val_accuracy = correct / total * 100

        print(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (Val Loss: {val_loss:.4f})")

        # Step the scheduler
        scheduler.step(val_loss)

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")


if __name__ == "__main__":
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
