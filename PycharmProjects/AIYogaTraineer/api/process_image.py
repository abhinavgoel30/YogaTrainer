from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw
import io
import tensorflow as tf
import tensorflow_hub as hub  # Ensure tensorflow_hub is installed using 'pip install tensorflow-hub'
import numpy as np



router = APIRouter()


# Load the MoveNet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures["serving_default"]  # Use the correct signature

def preprocess_image(image: Image):
    # Resize image to 192x192 as expected by MoveNet
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    image = image.resize((192, 192))

    # Convert image to numpy array and ensure it's uint8
    img_array = np.array(image, dtype=np.int32)  # Convert to uint8 (0-255)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def annotate_image(image: Image, keypoints: np.ndarray):
    draw = ImageDraw.Draw(image)
    original_width, original_height = image.size

    # List of keypoint connections (pairs of keypoints to connect with lines)
    # Each tuple contains indices of connected keypoints
    keypoint_connections = [
        (0, 1), (0, 2),  # Nose to eyes
        (1, 3), (2, 4),  # Left eye to left ear, right eye to right ear
        (0, 5), (0, 6),  # Nose to left shoulder, right shoulder
        (5, 7), (7, 9), (9, 11), (11, 13), (13, 15),  # Left arm
        (6, 8), (8, 10), (10, 12), (12, 14), (14, 16),  # Right arm
        (5, 6),  # Left shoulder to right shoulder (spine)
        (11, 12),  # Left hip to right hip (spine)
        (15, 16)  # Left knee to right knee (spine)
    ]

    for i in range(17):  # Iterate over the 17 keypoints
        y, x, confidence = keypoints[i]
        x, y = int(x * original_width), int(y * original_height)

        # Draw the keypoint as a red dot if confidence is above threshold
        if confidence > 0.5:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')

    # Draw lines connecting the keypoints
    for (start, end) in keypoint_connections:
        y1, x1, conf1 = keypoints[start]
        y2, x2, conf2 = keypoints[end]

        # Convert to pixel coordinates
        x1, y1 = int(x1 * original_width), int(y1 * original_height)
        x2, y2 = int(x2 * original_width), int(y2 * original_height)

        # Only draw lines between keypoints if both have high confidence
        if conf1 > 0.5 and conf2 > 0.5:
            draw.line((x1, y1, x2, y2), fill="green", width=2)  # Green line

    return image


@router.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Open and process the image
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess image for the model
    img_array = preprocess_image(image)

    # Convert numpy array to tensor and ensure correct dtype
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.int32)  # Convert to int32

    # Run inference with MoveNet
    outputs = movenet(input_tensor)  # Use correct signature

    # Extract keypoints from the model output
    keypoints = outputs["output_0"].numpy().squeeze()  # Remove the batch dimension

    # Remove the extra singleton dimension
    keypoints = keypoints.squeeze()  # Shape: (17, 3)

    # Debugging: print the keypoints
    print("Keypoints:", keypoints)

    # Annotate image with detected keypoints
    annotated_image = annotate_image(image, keypoints)

    # Convert annotated image to byte array
    img_byte_arr = io.BytesIO()

    # Ensure the image is in RGB mode before saving as JPEG
    annotated_image = annotated_image.convert("RGB")

    annotated_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Return the annotated image in response
    return Response(content=img_byte_arr, media_type="image/jpeg")