# server.py
import logging
import os

# import uuid
from io import BytesIO

import litserve as ls

# import tracemalloc
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# tracemalloc.start()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "2"))
DEFAULT_PROMPT = os.environ.get("DEFAULT_PROMPT", "<grounding> Describe this image:")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Kosmos2API(ls.LitAPI):
    def setup(self, device):
        model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.model = model.to(device)
        self.prompt = DEFAULT_PROMPT
        self.device = device

    def decode_request(self, request):
        file_obj = request["content"]
        prompt = request.get("prompt")
        if prompt is None:
            logger.debug("Using default prompt")
            prompt = self.prompt
        if "<grounding>" not in prompt:
            prompt = f"<grounding> {prompt}"
        try:
            logger.info("Processing file")
            file_bytes = file_obj.file.read()
            # filename = request["content"].filename  # Extract filename from the request
            return prompt, Image.open(BytesIO(file_bytes))
        except AttributeError:
            if "http" in file_obj:
                logger.info("Processing URL")
                return None, file_obj
        finally:
            if not isinstance(file_obj, str):
                file_obj.file.close()  # Explicitly close the file object

    def batch(self, inputs):
        # comes in as a list of tuples
        # logger.debug(f"Type of inputs: {type(inputs)}")
        # comes out as a tuple of lists
        prompts = [i[0] for i in inputs]
        images = [i[1] for i in inputs]
        return prompts, images

    def predict(self, prompts_and_images):
        prompt, images = prompts_and_images
        # if isinstance(images, list):
        #     prompt = [self.prompt] * len(images)
        # else:
        #     prompt = self.prompt
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=256,
        )
        generated_text = self.processor.batch_decode(
            generated_ids.cpu(), skip_special_tokens=True
        )
        data = []
        for i in range(len(generated_text)):
            text = generated_text[i]
            _processed_text = self.processor.post_process_generation(
                text, cleanup_and_extract=False
            )
            processed_text, entities = self.processor.post_process_generation(text)

            data.append(
                {
                    # "filename": original_filename,
                    "generated_text": _processed_text,
                    "entities": entities,
                    "output": processed_text,
                }
            )
        return data

    def unbatch(self, output):
        return [output]  # align with syntax for MAX_BATCH_SIZE=1

    def encode_response(self, output):
        return output[0]


if __name__ == "__main__":
    server = ls.LitServer(
        Kosmos2API(),
        accelerator="auto",
        max_batch_size=MAX_BATCH_SIZE,  # needs to be > 1 to hit self.batch
        track_requests=True,
        api_path="/predict",
    )
    server.run(
        port=8020,
        host="0.0.0.0",
        num_api_servers=NUM_API_SERVERS,
        log_level=LOG_LEVEL.lower(),
    )
