import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ray
import requests
from PIL import Image
from ray.exceptions import GetTimeoutError, RayTaskError

from bboxes import draw_entity_boxes_on_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init()


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
    output_json_path = f"{output_dir}/{image_path.split('.')[-2]}.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    json.dump(response, open(output_json_path, "w"))
    return response


def process_all_frames(frames_dir, prompt, output_dir="out"):
    """
    Process all frames in a directory.

    Args:
        frames_dir (str): Directory containing the frames.
        prompt (str): Text prompt for the model.
        output_dir (str): Directory to save processed output.
    """
    for frame_file in sorted(os.listdir(frames_dir))[:120]:
        if frame_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(frames_dir, frame_file)
            data = process_image(image_path, prompt, output_dir)


@ray.remote
def process_image_ray(image_path, prompt, output_dir="out", timeout_seconds=30):
    """Ray-compatible version of process_image with timeout"""
    logger.info(f"Starting to process: {image_path}")

    # Wrap the process_image call with timeout
    try:
        result = process_image(image_path, prompt, output_dir)

        # Verify the output file exists
        frame_name = Path(image_path).stem
        expected_output = Path(output_dir) / f"{frame_name}.png"

        if not expected_output.exists():
            raise FileNotFoundError(f"Output file not created: {expected_output}")

        logger.info(f"Successfully processed: {image_path}")
        return result
    except Exception as e:
        logger.error(f"Error in process_image_ray: {str(e)}")
        raise  # Re-raise to be caught by the main function


def process_frames_ray(
    frames_dir,
    prompt,
    output_dir="out",
    max_retries=3,
    timeout_seconds=10,
    retry_delay=3,
    max_concurrent=40,
):  # Control concurrent processing
    """Process frames using Ray with parallel execution"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            f
            for f in Path(frames_dir).iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )

    if not image_files:
        logger.error(f"No image files found in {frames_dir}")
        return []

    logger.info(f"Found {len(image_files)} images to process")

    # Submit all tasks initially
    pending_futures = {
        process_image_ray.remote(str(img), prompt, output_dir): {
            "path": str(img),
            "retries": 0,
        }
        for img in image_files
    }

    results = []
    failed_files = []

    # Process results as they complete
    while pending_futures:
        # Wait for one task to complete with timeout
        done_futures, remaining_futures = ray.wait(
            list(pending_futures.keys()),
            timeout=timeout_seconds,
            num_returns=1,  # Process one result at a time
        )

        for future in done_futures:
            image_info = pending_futures[future]
            image_path = image_info["path"]

            try:
                result = ray.get(future)
                results.append(result)
                logger.info(f"Successfully processed {image_path}")
                pending_futures.pop(future)

            except (RayTaskError, GetTimeoutError, Exception) as e:
                retry_count = image_info["retries"] + 1

                if retry_count < max_retries:
                    logger.warning(
                        f"Attempt {retry_count} failed for {image_path}: {str(e)}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)

                    # Resubmit the task
                    new_future = process_image_ray.remote(
                        image_path, prompt, output_dir
                    )
                    pending_futures[new_future] = {
                        "path": image_path,
                        "retries": retry_count,
                    }

                else:
                    logger.error(f"All attempts failed for {image_path}: {str(e)}")
                    failed_files.append(
                        {"path": image_path, "error": str(e), "attempts": retry_count}
                    )

                pending_futures.pop(future)

        # If no futures completed in this iteration, wait a bit to prevent busy-waiting
        if not done_futures:
            time.sleep(0.1)

    # Final summary
    logger.info("\nProcessing complete!")
    logger.info(f"Successfully processed: {len(results)} images")
    logger.info(f"Failed to process: {len(failed_files)} images")

    if failed_files:
        logger.info("\nFailed files:")
        for failed in failed_files:
            logger.info(
                f"  - {failed['path']}: "
                f"Failed after {failed['attempts']} attempts: {failed['error']}"
            )

    return results


if __name__ == "__main__":
    import time

    # Configuration
    FRAMES_DIR = "frames"
    OUTPUT_DIR = "out"
    PROMPT = "<grounding> Find the white fish in the image:"

    # Initialize model and processor

    # Process all frames
    # print(process_image("IMG_0395.jpg", PROMPT, OUTPUT_DIR))

    start = time.time()
    process_frames_ray(FRAMES_DIR, PROMPT, OUTPUT_DIR)
    # process_all_frames(FRAMES_DIR, PROMPT, OUTPUT_DIR)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    ray.shutdown()
