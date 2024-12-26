import json
import os

import requests
from PIL import Image

from bboxes import draw_entity_boxes_on_image


# need equivalent of 	curl -X POST -F "content=@IMG_0395.jpg" http://127.0.0.1:8020/predict | jq '.output'
def predict(image_path):
    url = "http://127.0.0.1:8020/predict"
    files = {"content": open(image_path, "rb")}
    response = requests.post(url, files=files)
    return response.json()


def process_image(image_path, prompt, output_dir="out"):
    """
    Process a single image to generate text and extract entities.

    Args:
        image_path (str): Path to the input image.
        prompt (str): Text prompt for the model.
        output_dir (str): Directory to save processed output.

    Returns:
        dict: Entities detected in the image.
    """
    image = Image.open(image_path)
    response = predict(image_path)
    entities = response["entities"]
    # Save output image with entity boxes
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    draw_entity_boxes_on_image(image, entities, show=False, save_path=output_path)

    return response


def process_all_frames(frames_dir, model, processor, prompt, output_dir="out"):
    """
    Process all frames in a directory.

    Args:
        frames_dir (str): Directory containing the frames.
        model: The vision-to-sequence model.
        processor: The processor for the model.
        prompt (str): Text prompt for the model.
        output_dir (str): Directory to save processed output.
    """
    for frame_file in sorted(os.listdir(frames_dir)):
        if frame_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(frames_dir, frame_file)
            data = process_image(image_path, model, processor, prompt, output_dir)
            json.dump(data, open(f"{output_dir}/{frame_file.split('.')[-2]}.json", "w"))


if __name__ == "__main__":
    # Configuration
    FRAMES_DIR = "frames"
    OUTPUT_DIR = "out"
    PROMPT = "<grounding> Describe this image in detail:"

    # Initialize model and processor

    # Process all frames
    # process_all_frames(FRAMES_DIR, model, processor, PROMPT, OUTPUT_DIR)
    print(process_image("IMG_0395.jpg", PROMPT, OUTPUT_DIR))
