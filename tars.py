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

def adjust_thickness(img, target_thickness=67):
    """
    Adjust the thickness of the digit lines in the image.

    Parameters:
    - img: Binary image (0 and 255 values) of the digit.
    - target_thickness: The desired stroke thickness (in pixels).

    Returns:
    - Adjusted binary image with the desired stroke thickness.
    """
    # Kernel for morphological operations
    kernel = np.ones((2, 2), np.uint8)

    # Measure current stroke thickness (approximation)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        bounding_box = cv2.boundingRect(max_contour)
        current_thickness = max(bounding_box[2], bounding_box[3]) // 20  # Approximation

        # Adjust thickness
        if current_thickness < target_thickness:  # Too thin, dilate
            img = cv2.dilate(img, kernel, iterations=1)
        elif current_thickness > target_thickness:  # Too thick, erode
            img = cv2.erode(img, kernel, iterations=1)

    return img

# Helper function to preprocess the image
def preprocess_image(image_path):
    # Read image in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError("Failed to load the image. Ensure it is a valid image file.")

    # Invert colors if needed (ensure black background with white digit)
    gray = cv2.bitwise_not(gray) if np.mean(gray) > 127 else gray

    # Threshold the image (binarization)
    _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Ensure the image is not empty after thresholding
    if np.sum(gray) == 0:
        raise ValueError("The image is completely empty after thresholding. Please check the input.")

    # Adjust line thickness
    gray = adjust_thickness(gray, 67)

    # Trim black borders
    while np.sum(gray[0]) == 0 and gray.shape[0] > 1:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0 and gray.shape[1] > 1:
        gray = np.delete(gray, 0, axis=1)
    while np.sum(gray[-1]) == 0 and gray.shape[0] > 1:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0 and gray.shape[1] > 1:
        gray = np.delete(gray, -1, axis=1)

    # Check if the image is empty
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        raise ValueError("The uploaded image is empty or could not be processed.")

    # Resize to fit in a 20x20 box
    rows, cols = gray.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = max(1, int(round(cols * factor)))  # Ensure cols > 0
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = max(1, int(round(rows * factor)))  # Ensure rows > 0
        gray = cv2.resize(gray, (cols, rows))

    # Pad to 28x28
    cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rows_padding, cols_padding), mode='constant', constant_values=0)

    # Center the digit using the center of mass
    if np.sum(gray) == 0:  # Check if the image is still completely black
        raise ValueError("The image is completely empty or invalid after processing.")

    cy, cx = center_of_mass(gray)
    if np.isnan(cy) or np.isnan(cx):  # Check for NaN values
        raise ValueError("Center of mass calculation resulted in NaN. Check the input image.")

    shift_x = int(np.round(28 / 2 - cx))
    shift_y = int(np.round(28 / 2 - cy))

    # Shift the image
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    gray = cv2.warpAffine(gray, M, (28, 28))

    # Normalize pixel values to range [0, 1]
    gray = gray.astype("float32") / 255.0
    gray = (gray - 0.5) / 0.5

    # Save the preprocessed image for debugging
    debug_image_path = "debug_preprocessed_image.png"
    cv2.imwrite(debug_image_path, ((gray * 0.5 + 0.5) * 255).astype("uint8"))  # Rescale to [0, 255] for saving
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
