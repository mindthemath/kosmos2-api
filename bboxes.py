import os
from textwrap import wrap

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, response, show=False, save_path=None, seed=19):
    """_summary_
    Args:
        image (_type_): image or image path
        response (dict): response from the model
        show (bool): whether to show the image
        save_path (str): path to save the image, if None, no image is saved
        seed (int): seed for random coloring of bounding boxes
    """
    entities = response["entities"]
    answer = response["output"]  # answer to the prompt
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[
            :, None, None
        ]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[
            :, None, None
        ]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 1
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize(
        "F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line
    )
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    if seed is not None:
        np.random.seed(seed)
    for entity_name, (start, end), bboxes in entities:
        for x1_norm, y1_norm, x2_norm, y2_norm in bboxes:
            orig_x1, orig_y1, orig_x2, orig_y2 = (
                int(x1_norm * image_w),
                int(y1_norm * image_h),
                int(x2_norm * image_w),
                int(y2_norm * image_h),
            )
            # draw bbox
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(
                new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line
            )

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = (
                    orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                )
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(
                f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line
            )
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = (
                x1,
                y1 - (text_height + text_offset_original + 2 * text_spaces),
                x1 + text_width,
                y1,
            )

            for prev_bbox in previous_bboxes:
                while is_overlapping(
                    (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox
                ):
                    text_bg_y1 += text_height + text_offset_original + 2 * text_spaces
                    text_bg_y2 += text_height + text_offset_original + 2 * text_spaces
                    y1 += text_height + text_offset_original + 2 * text_spaces

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(
                            0,
                            image_h
                            - (text_height + text_offset_original + 2 * text_spaces),
                        )
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (
                            alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)
                        ).astype(np.uint8)

            cv2.putText(
                new_image,
                f"  {entity_name}",
                (x1, y1 - text_offset_original - 1 * text_spaces),
                cv2.FONT_HERSHEY_COMPLEX,
                text_size,
                (0, 0, 0),
                text_line,
                cv2.LINE_AA,
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    # add answer to the image - use black border on text and white text
    # Draw black border first
    top_left = (image_w // 50, 5 * image_h // 100)
    fsize = np.round(image_h / 1000, 4)
    border_width = 3
    text_width = 1

    # Calculate maximum width based on font size
    sample_text = "A" * 50  # Sample text to measure character width
    (text_width_px, text_height_px), _ = cv2.getTextSize(
        sample_text, cv2.FONT_HERSHEY_COMPLEX, fsize, text_width
    )
    avg_char_width = text_width_px / 50  # Average width per character
    max_chars_per_line = int(
        (image_w - 2 * top_left[0]) / avg_char_width
    )  # Account for margins

    # wrap answer text in a box with calculated width
    answer_lines = wrap(answer, width=max_chars_per_line)
    line_height = image_h // 20  # Pixels between lines

    for i, line in enumerate(answer_lines):
        # Calculate y position for each line, going top-down
        y_pos = top_left[1] + i * line_height

        # Draw black border
        cv2.putText(
            new_image,
            line,
            (top_left[0], y_pos),
            cv2.FONT_HERSHEY_COMPLEX,
            fsize,
            (0, 0, 0),
            border_width,
            cv2.LINE_AA,
        )

        # Draw white text on top
        cv2.putText(
            new_image,
            line,
            (top_left[0], y_pos),
            cv2.FONT_HERSHEY_COMPLEX,
            fsize,
            (255, 255, 255),
            text_width,
            cv2.LINE_AA,
        )
    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)
    if show:
        pil_image.show()

    return new_image


if __name__ == "__main__":

    # (The same image from the previous code example)
    url = (
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
    )
    image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open("frames/frame_000001.png")
    # From the previous code example
    response = {}
    response["entities"] = [
        ("a snowman", (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]),
        ("a fire", (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)]),
    ]
    response["output"] = (
        "A snowman is sitting in a snowy field. There is a fire in the foreground. This is a much longer piece of text to test the wrapping. More and more lines."
    )

    # Draw the bounding bboxes
    draw_entity_boxes_on_image(image, response, show=False, save_path="snowman.jpg")
