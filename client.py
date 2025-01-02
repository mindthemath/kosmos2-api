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
def predict(image_path, prompt, session=None):
    if session is None:
        session = requests.Session()
    url = "http://127.0.0.1:8020/predict"
    files = {"content": open(image_path, "rb")}
    data = {"prompt": prompt}
    # add auth header
    auth = os.environ.get("LIGHTNING_API_KEY")
    if auth is None:
        auth_header = None
    else:
        auth_header = {"Authorization": f"Bearer {auth}"}
    response = session.post(url, files=files, data=data, headers=auth_header)
    return response.json()


def process_image(image_path, prompt, output_dir="out", session=None, save_image=True):
    """
    Process a single image to generate text and extract entities.

    Args:
        image_path (str): Path to the input image.
        prompt (str): Text prompt for the model.
        output_dir (str): Directory to save processed output.
        session (requests.Session): Optional session object to reuse.
        save_image (bool): Whether to save the output image.

    Returns:
        dict: Entities detected in the image.
    """
    image = Image.open(image_path)
    response = predict(image_path, prompt, session)
    # Save output image with entity boxes
    os.makedirs(output_dir, exist_ok=True)
    if save_image:
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        draw_entity_boxes_on_image(image, response, show=False, save_path=output_path)
    output_json_path = f"{output_dir}/{image_path.split('.')[-2]}.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    json.dump(response, open(output_json_path, "w"))
    return response


# def process_all_frames(frames_dir, prompt, output_dir="out"):
#     """
#     Process all frames in a directory.

#     Args:
#         frames_dir (str): Directory containing the frames.
#         prompt (str): Text prompt for the model.
#         output_dir (str): Directory to save processed output.
#     """
#     for frame_file in sorted(os.listdir(frames_dir))[:120]:
#         if frame_file.lower().endswith((".png", ".jpg", ".jpeg")):
#             image_path = os.path.join(frames_dir, frame_file)
#             data = process_image(image_path, prompt, output_dir)


@ray.remote
class APIWorker:
    def __init__(self):
        self.session = requests.Session()  # Reuse connection

    def predict(self, image_path, prompt):
        return process_image(image_path, prompt, output_dir="out", session=self.session)


def process_frames_ray(
    frames_dir,
    prompt,
    output_dir="out",
    max_retries=3,
    timeout_seconds=10,
    retry_delay=1,
    num_workers=32,  # Number of concurrent workers
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create a pool of workers
    workers = [APIWorker.remote() for _ in range(num_workers)]
    worker_idx = 0

    image_files = sorted(
        [
            f
            for f in Path(frames_dir).iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )

    # Submit initial batch of tasks
    pending_futures = {}
    for img in image_files:
        worker = workers[worker_idx]
        future = worker.predict.remote(str(img), prompt)
        pending_futures[future] = {"path": str(img), "retries": 0}
        worker_idx = (worker_idx + 1) % num_workers

    results = []
    failed_files = []

    while pending_futures:
        done_futures, remaining_futures = ray.wait(
            list(pending_futures.keys()),
            timeout=timeout_seconds,
            num_returns=min(num_workers, len(pending_futures)),
        )

        for future in done_futures:
            if future not in pending_futures:
                continue

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

                    # Resubmit using next available worker
                    worker = workers[worker_idx]
                    worker_idx = (worker_idx + 1) % num_workers
                    new_future = worker.predict.remote(image_path, prompt)
                    pending_futures[new_future] = {
                        "path": image_path,
                        "retries": retry_count,
                        "prompt": prompt,
                    }

                else:
                    logger.error(f"All attempts failed for {image_path}: {str(e)}")
                    failed_files.append(
                        {"path": image_path, "error": str(e), "attempts": retry_count}
                    )
                pending_futures.pop(future)

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
    # PROMPT = "<grounding> Find the white fish in the image:"
    PROMPT = "<grounding> Describe the scene in detail:"

    # Initialize model and processor

    # Process all frames
    # print(process_image("IMG_0395.jpg", PROMPT, OUTPUT_DIR))

    start = time.time()
    results = process_frames_ray(FRAMES_DIR, PROMPT, OUTPUT_DIR)
    # process_all_frames(FRAMES_DIR, PROMPT, OUTPUT_DIR)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(f"Framerate: {len(results) / (end - start)} fps")
    ray.shutdown()
