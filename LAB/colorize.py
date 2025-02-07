"""
Credits: 
	1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
	2. http://richzhang.github.io/colorization/
	3. https://github.com/richzhang/colorization/
"""
import numpy as np
import cv2
import os
from PIL import Image

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to model files
PROTOTXT = os.path.join(current_dir, "model", "colorization_deploy_v2.prototxt")
POINTS = os.path.join(current_dir, "model", "pts_in_hull.npy")
MODEL = os.path.join(current_dir, "model", "colorization_release_v2.caffemodel")

# Verify files exist
for path in [PROTOTXT, POINTS, MODEL]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing: {path}")

# Load the model
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

net = load_model()

def colorize_image(image, l_adjust=0):
    """Convert B&W image to color with L adjustment"""
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 60  # Original adjustment

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Apply L adjustment
    L = cv2.split(lab)[0]
    L = np.clip(L.astype("float32") + l_adjust, 0, 100)  # Apply adjustment
    
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")
