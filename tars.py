import torch
from fastapi import FastAPI, File, UploadFile, WebSocket
import numpy as np
import cv2
import uvicorn
from scipy.ndimage import center_of_mass
from net import Net
from harness import Harness
from fastapi.middleware.cors import CORSMiddleware
import base64
import os

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

def _file_to_data_url(path: str, mime: str = "image/png") -> str | None:
    """Read a file and return a data URL, or None if not found."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except FileNotFoundError:
        return None

import cv2
import numpy as np

import cv2
import numpy as np

def preprocess_image(image_path: str) -> np.ndarray:
    """
    White-on-black MNIST-style 28x28, float32 in [-1, 1].
    Saves:
      - debug_raw_image.png        (as loaded)
      - debug_preprocessed.png     (final 28x28 visualization)
    """
    # --- 1) Load ---
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image")
    cv2.imwrite("debug_raw_image.png", img)

    # --- 2) Grayscale & ensure white-on-black ---
    if img.ndim == 3 and img.shape[-1] == 4:
        rgb = img[..., :3].astype(np.float32)
        a = (img[..., 3:4].astype(np.float32) / 255.0)
        rgb = rgb * a
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    if np.max(gray) < 10:
        out = (np.zeros((28, 28), np.float32) - 1.0)
        vis = np.clip(((out + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite("debug_preprocessed.png", vis)
        return out

    # --- 3) Mask & bbox (loosened) ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_for_bbox = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    coords = cv2.findNonZero(mask_for_bbox)
    if coords is None:
        out = (np.zeros((28, 28), np.float32) - 1.0)
        vis = np.clip(((out + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite("debug_preprocessed.png", vis)
        return out

    x, y, w, h = cv2.boundingRect(coords)
    margin = 3
    x0 = max(x - margin, 0); y0 = max(y - margin, 0)
    x1 = min(x + w + margin, gray.shape[1]); y1 = min(y + h + margin, gray.shape[0])
    cropped = gray[y0:y1, x0:x1]

    # --- 4) Resize longest side to 20 ---
    target_inner = 20
    hc, wc = cropped.shape
    if max(hc, wc) == 0:
        out = (np.zeros((28, 28), np.float32) - 1.0)
        vis = np.clip(((out + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite("debug_preprocessed.png", vis)
        return out

    scale = target_inner / float(max(hc, wc))
    new_w = max(1, int(round(wc * scale)))
    new_h = max(1, int(round(hc * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)

    # --- 5) Gentle thicken ---
    res_u8 = resized.astype(np.uint8)
    res_u8 = cv2.dilate(res_u8, np.ones((1, 1), np.uint8), iterations=1)

    # --- 6) Paste, pad, COM-center, crop back ---
    base = np.zeros((28, 28), dtype=np.float32)
    ys = (28 - new_h) // 2
    xs = (28 - new_w) // 2
    base[ys:ys + new_h, xs:xs + new_w] = res_u8.astype(np.float32)

    pad = 5  # safe border for shifting
    padded = cv2.copyMakeBorder(
        base, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0.0,)   # <- Sequence[float] to satisfy stubs
    )

    thr = float(padded.max()) * 0.3
    mass = (padded > thr).astype(np.float32)
    if float(mass.sum()) > 1e-5:
        gy, gx = np.indices(mass.shape, dtype=np.float32)
        cy = float((gy * mass).sum() / mass.sum())
        cx = float((gx * mass).sum() / mass.sum())
        dy = int(round((pad + 14) - cy))
        dx = int(round((pad + 14) - cx))
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        padded = cv2.warpAffine(
            padded, M,
            [int(padded.shape[1]), int(padded.shape[0])],  # <- Sequence[int]
            flags=int(cv2.INTER_LINEAR),
            borderMode=int(cv2.BORDER_CONSTANT),
            borderValue=(0.0,)  # <- Sequence[float]
        )

    h_pad, w_pad = padded.shape
    y0 = (h_pad - 28) // 2
    x0 = (w_pad - 28) // 2
    canvas = padded[y0:y0+28, x0:x0+28]

    # --- 7) Supersample smoothing ---
    up = cv2.resize(canvas, (56, 56), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (5, 5), 0.8)
    canvas = cv2.resize(up, (28, 28), interpolation=cv2.INTER_AREA)

    # --- 8) Contrast stretch in-ink, then normalize to [-1,1] ---
    roi = canvas[canvas > 0]
    if roi.size > 0:
        hi = float(np.percentile(roi, 99.0))
        if hi > 0:
            canvas = np.clip(canvas * (255.0 / hi), 0, 255)

    canvas = (canvas / 255.0 - 0.5) / 0.5
    out = canvas.astype(np.float32)

    # --- Debug visualization ---
    vis = np.clip(((out + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite("debug_preprocessed.png", vis)

    return out


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

        gradcam_path = "gradcam_output.png"
        grad_cam(model, preprocessed_image_tensor, predicted_digit, save_path="gradcam_output.png")
        grad_cam(model, preprocessed_image_tensor, predicted_digit, save_path=gradcam_path)
        gradcam_data_url = _file_to_data_url(gradcam_path)

        # Clean up temporary file
        os.remove(temp_file_path)

        # Generate Natural Language Response

        response = harness.generate_response(predicted_digit, confidence, probabilities)

        return {
            "digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "response": response,
            "gradcam_data_url": gradcam_data_url
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
