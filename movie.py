import json
import os

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from bboxes import draw_entity_boxes_on_image


def initialize_model_and_processor(model_name="microsoft/kosmos-2-patch14-224"):
    """
    Initialize the model and processor for vision-to-sequence tasks.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        tuple: (model, processor)
    """
    model = AutoModelForVision2Seq.from_pretrained(model_name)
    model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def process_image(image_path, model, processor, prompt, output_dir="out"):
    """
    Process a single image to generate text and extract entities.

    Args:
        image_path (str): Path to the input image.
        model: The vision-to-sequence model.
        processor: The processor for the model.
        prompt (str): Text prompt for the model.
        output_dir (str): Directory to save processed output.

    Returns:
        dict: Entities detected in the image.
    """
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=256,
    )

    generated_text = processor.batch_decode(
        generated_ids.cpu(), skip_special_tokens=True
    )[0]
    _processed_text = processor.post_process_generation(
        generated_text, cleanup_and_extract=False
    )
    processed_text, entities = processor.post_process_generation(generated_text)

    # Print processed text and entities
    print(f"Image: {image_path}")
    print("Generated Description:", processed_text)
    for entity in entities:
        print(entity)

    # Save output image with entity boxes
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    draw_entity_boxes_on_image(image, entities, show=False, save_path=output_path)

    return {
        "processed_text": processed_text,
        "entities": entities,
        "_generated_text": _processed_text,
    }


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
    model, processor = initialize_model_and_processor()

    # Process all frames
    process_all_frames(FRAMES_DIR, model, processor, PROMPT, OUTPUT_DIR)
