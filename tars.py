import math
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket
import numpy as np
import cv2
import uvicorn
import os
from scipy.ndimage import center_of_mass
from net import Net
from harness import Harness
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
MODEL_PATH = "best_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Harness model
harness = Harness()
harness.hook(model)


def preprocess_image(image_path):
    """Preprocess the image while preserving the digit's structure as closely as possible to MNIST format."""

    # 1. Read in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Failed to load the image. Ensure it is a valid image file.")

    # Debug: Save raw input
    raw_image_path = "debug_raw_image.png"
    cv2.imwrite(raw_image_path, gray)
    print(f"Raw input image saved at: {raw_image_path}")

    # 2. Detect if background is white or black by looking at the average pixel.
    #    If mean is > 127, we assume a white background and invert to get white digit on black.
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # 3. Optional slight blur to reduce noise while leaving strokes relatively intact.
    #    You can disable this if it hurts your accuracy.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. Use Otsu’s threshold to robustly binarize without losing thin strokes.
    #    (You can also try cv2.adaptiveThreshold if Otsu is too aggressive.)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 5. (Optional) If your strokes are very thin/faint, you can apply a small dilation:
    # kernel = np.ones((2, 2), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)

    # 6. Safe crop: find the bounding box of nonzero pixels and pad slightly so we don't clip strokes.
    def safe_crop(img):
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 4  # Increase or decrease if you find edges being chopped
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + pad * 2)
            h = min(img.shape[0] - y, h + pad * 2)
            img = img[y:y + h, x:x + w]
        return img

    gray = safe_crop(gray)

    # Make sure we didn't end up with an empty crop
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        raise ValueError("The uploaded image is empty or could not be processed.")

    # 7. Resize to a max dimension of 20 in whichever is larger (height or width) to mimic MNIST
    h, w = gray.shape
    # Keep aspect ratio:
    if h > w:
        new_h, new_w = 20, int(20 * (w / h))
    else:
        new_w, new_h = 20, int(20 * (h / w))

    # Use nearest-neighbor so we don’t blur away thin strokes
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 8. Pad up to 28×28, centered
    pad_h = (28 - new_h) // 2
    pad_w = (28 - new_w) // 2
    gray = np.pad(gray,
                  ((pad_h, 28 - new_h - pad_h), (pad_w, 28 - new_w - pad_w)),
                  mode='constant', constant_values=0)

    # 9. (Optional) Recenter by shifting based on center of mass.
    cy, cx = center_of_mass(gray)
    if np.isnan(cy) or np.isnan(cx):
        raise ValueError("Center of mass calculation resulted in NaN. Check the input image.")

    shift_x = int(np.round(28 / 2 - cx))
    shift_y = int(np.round(28 / 2 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    gray = cv2.warpAffine(gray, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))

    # 10. Normalize to [-1, 1]
    gray = gray.astype("float32") / 255.0
    gray = (gray - 0.5) / 0.5

    # Debug: Save final preprocessed image (converted back to [0..255])
    debug_image_path = "debug_preprocessed_image_fixed.png"
    cv2.imwrite(debug_image_path, ((gray * 0.5 + 0.5) * 255).astype("uint8"))
    print(f"Preprocessed image saved at: {debug_image_path}")

    return gray

# Prediction endpoint
@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = "temp_uploaded_image.png"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Preprocess the image
        preprocessed_image = preprocess_image(temp_file_path)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=(0, 1))

        preprocessed_image_tensor = torch.tensor(preprocessed_image, dtype=torch.float32).to(device)

        harness.set_input(preprocessed_image)

        with torch.no_grad():
            if harness.websocket:
                await harness.websocket.send_json({
                    "layer": "input",
                    "activation_shape": preprocessed_image.shape,
                    "activation_data": preprocessed_image.tolist()
                })
            output = model(preprocessed_image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

        # Use harness to save activations
        harness.save_activations()
        print(f"Activations saved to 'activity.npz'")

        grad_cam(model, preprocessed_image_tensor, predicted_digit, save_path="gradcam_output.png")

        # Clean up temporary file
        os.remove(temp_file_path)

        return {
            "digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    harness.websocket = websocket
    await websocket.send_text("WebSocket connected: ready to stream activations")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        harness.websocket = None

def grad_cam(model, input_tensor, target_class, target_layer_idx=0, save_path="gradcam_output.png"):
    model.eval()

    # Hook to capture the activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on the target layer
    target_layer = list(model.model.children())[target_layer_idx]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    target_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    target_score.backward()

    # Extract activations and gradients
    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))  # Global average pooling on gradients

    # Compute Grad-CAM heatmap
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)  # ReLU to keep only positive values
    cam = cv2.resize(cam, (28, 28))  # Resize to match input image size
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0, 1]

    # Overlay heatmap on the original image
    heatmap = (cam * 255).astype("uint8")
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert grayscale input to 3-channel RGB for blending
    input_image_rescaled = ((input_tensor[0, 0].cpu().numpy() * 0.5 + 0.5) * 255).astype("uint8")
    input_image_rgb = cv2.cvtColor(input_image_rescaled, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(heatmap_color, 0.5, input_image_rgb, 0.5, 0)

    # Save heatmap
    cv2.imwrite(save_path, overlay)
    print(f"Grad-CAM heatmap saved to: {save_path}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "TARS API is running!"}

# Run the app if this script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
