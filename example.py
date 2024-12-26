import requests
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

model = model.to("cuda")
# url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("IMG_0395.jpg")

from bboxes import draw_entity_boxes_on_image


def run_example(prompt):

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

    print(processed_text)
    for e in entities:
        print(e)
    print(_processed_text)
    return entities


if __name__ == "__main__":
    # Phrase Grounding
    # phrase = "a murder of crows"
    # prompt = f"<grounding><phrase> {phrase}</phrase>"
    # run_example(prompt)

    # # Referring Expression Comprehension
    # phrase = "a murder of crows in a tree"
    # prompt = f"<grounding><phrase> {phrase}</phrase>"
    # run_example(prompt)

    # # Referring Expression Generation
    # # prompt = "<grounding><phrase> It</phrase><object><patch_index_0044><patch_index_0863></object> is"
    # # run_example(prompt)

    # # Grounded VQA
    prompt = "<grounding> Question: Where was this photo taken? Answer:"
    entities = run_example(prompt)

    # # Grounded VQA with multimodal referring via bounding boxes
    # # prompt = "<grounding> Question: Where is<phrase> the fire</phrase><object><patch_index_0005><patch_index_0911></object> next to? Answer:"
    # # run_example(prompt)

    # # Grounded Image Captioning
    # prompt = "<grounding> Describe this image:"
    # entities = run_example(prompt)

    # prompt = "<grounding> Describe this image in detail:"
    # entities = run_example(prompt)

    draw_entity_boxes_on_image(image, entities, show=False, save_path="crows.jpg")
