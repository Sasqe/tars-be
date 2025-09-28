import torch
from fastapi import WebSocket
import random
import numpy as np
from respond import TEMPLATE_BANK


class Harness:
    def __init__(self, websocket: WebSocket = None):
        # Dictionary to store activations
        self.activations = {}
        self.input = None
        self.websocket = websocket

    def get_activation(self, name):
        """
        Hook function to capture the output (activations) of a layer.
        """
        def hook(model, input, output):
            activation = output.detach().cpu().numpy()
            self.activations[name] = activation
            # Stream activation data if WebSocket is connected
            if self.websocket:
                # Send the data asynchronously outside the hook
                import asyncio
                asyncio.create_task(self.websocket.send_json({
                    "layer": name,
                    "activation_shape": activation.shape,
                    "activation_data": activation.tolist()  # Convert to list for JSON serialization
                }))
        return hook

    def hook(self, model):
        """
        Registers hooks for all Conv2d and Linear layers in the model.
        """
        for idx, layer in enumerate(model.model):
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                layer.register_forward_hook(self.get_activation(f"layer_{idx}"))

    def set_input(self, input_image):
        """
        Sets the input image for the harness.
        """
        self.input = input_image

    def save_activations(self, filename='activity.npz'):
        """
        Saves the captured activations to a .npz file.
        """
        import numpy as np
        np.savez(filename, **self.activations)
        print(f"Activations saved to '{filename}'")

    def generate_response(self, predicted_digit, confidence, probabilities):
        # Determine category
        if confidence >= 0.80:
            category = "confident"
        elif confidence >= 0.70:
            category = "semi_confident"
        elif confidence >= 0.60:
            category = "a_little_unsure"
        elif confidence >= 0.55:
            category = "unsure"
        else:
            category = "no_idea"

        alt_digit = None
        if category in ("unsure", "a_little_unsure"):
            if random.random() < 0.5:  # 50% chance to use alt-digit template
                alt_idx = np.argsort(probabilities)[-2]  # 2nd highest
                alt_digit = int(alt_idx)

        if category == "a_little_unsure":
            key = "a_little_unsure_multi" if alt_digit is not None else "a_little_unsure_single"
        elif category == "unsure":
            key = "unsure_multi" if alt_digit is not None else "unsure_single"
        else:
            key = category

        templates = TEMPLATE_BANK[key]

        template = random.choice(templates)

        response = template.format(
            digit=predicted_digit,
            alt_digit=alt_digit if alt_digit is not None else ""
        )
        return response
