import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import Net  # Import the CNN model

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation for training dataset (moderate rotations/translations, no extreme distortions)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # small random rotations (<=10 degrees)
    transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # random shift up to 20% and scale
    # Removed RandomPerspective to avoid unrealistic distortion
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transformation for test/validation dataset (no random augmentation, just normalization)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model and move to device
model = Net().to(device)

# Define loss function (CrossEntropy for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer with weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Changed: Added weight_decay=1e-4 to Adam optimizer for L2 regularization (helps prevent overfitting):contentReference[oaicite:10]{index=10}.

# Learning rate scheduler (reduce LR if validation loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Early stopping utility
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        # Check if validation loss has improved by more than delta
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training loop function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=30,
                model_save_path="best_model.pth"):
    early_stopping = EarlyStopping(patience=10, delta=0.001)
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)  # accumulate summed loss

        # Calculate average training loss for the epoch
        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation phase (no grad)
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Calculate average validation loss and accuracy
        val_loss = running_val_loss / len(test_loader.dataset)
        val_accuracy = 100.0 * correct / total

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (improved Val Loss: {val_loss:.4f})")

        # Adjust learning rate based on val loss trend
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    print("Training complete.")
    return model

if __name__ == "__main__":
    trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
    # Note: You can use PyTorch Lightning's Trainer for a cleaner training loop and built-in early stopping:contentReference[oaicite:11]{index=11}.
    # Here we implemented manually for clarity.
