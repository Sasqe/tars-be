import numpy as np
import matplotlib.pyplot as plt
import os

# Load the activity.npz file
data = np.load('activity.npz')

# List all stored keys (layer names)
print("Keys in the .npz file:")
print(data.files)


# Verify and inspect a sample activation
def inspect_sample(layer_name):
    if layer_name not in data:
        print(f"Layer '{layer_name}' not found in activity.npz")
        return

    activation = data[layer_name]
    print(f"Layer: {layer_name}, Shape: {activation.shape}")

    # Print the first few values of the first sample
    print(f"Sample values (first 5 entries):\n{activation[0][:5]}")


# Visualize activations for a convolutional layer
def visualize_activations(layer_name, output_dir='activations'):
    if layer_name not in data:
        print(f"Layer '{layer_name}' not found in activity.npz")
        return

    activation = data[layer_name]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Visualize first sample
    sample = activation[0]  # First sample
    if sample.ndim == 3:  # For Conv2d layers
        num_channels = sample.shape[0]
        print(f"Visualizing {num_channels} channels from {layer_name}...")

        for i in range(num_channels):
            plt.imshow(sample[i], cmap='viridis')  # Change colormap if needed
            plt.title(f'{layer_name} - Channel {i}')
            plt.colorbar()

            # Save the visualization as PNG
            file_path = os.path.join(output_dir, f'{layer_name}_channel_{i}.png')
            plt.savefig(file_path)
            plt.close()
            print(f"Saved: {file_path}")
    else:
        print(f"Layer {layer_name} is not a 3D tensor. Skipping visualization.")


# Example usage
if __name__ == "__main__":
    # Inspect a specific layer (change the layer name as needed)
    inspect_sample('layer_0')  # Replace with the desired layer key

    # Visualize activations for a specific layer (e.g., first Conv2d layer)
    visualize_activations('layer_0')  # Replace with the desired layer key
