import glob
from typing import Generator, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class NomicEmbedder:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_model.to(self.device)

    @staticmethod
    def load_images(image_paths) -> Generator[Image.Image, None, None]:
        for i in image_paths:
            yield Image.open(i)

    def process_images(self, images: List[Image.Image]):
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        img_emb = self.vision_model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
        return img_embeddings.cpu().detach().numpy()

    def from_images(self, images: List[Image.Image], batch_size: int = 32):
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            embeddings = self.process_images(batch)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)

    def from_image_paths(self, image_paths: List[str], batch_size: int = 32):
        images = list(self.load_images(image_paths))
        return self.from_images(images, batch_size=batch_size)

    def from_folder(self, folder: str, ext: str = "png", batch_size: int = 32):
        image_paths = sorted(glob.glob(f"{folder}/*.{ext}"))
        return self.from_image_paths(image_paths, batch_size=batch_size)

    def from_glob(self, glob_pattern: str, sample: int = 0, batch_size: int = 32):
        image_paths = sorted(glob.glob(glob_pattern))
        if sample > 0:
            image_paths = image_paths[:sample]
        return self.from_image_paths(image_paths, batch_size=batch_size)


if __name__ == "__main__":
    embedder = NomicEmbedder()
    embeddings = embedder.from_glob("frames/*.png", batch_size=128)
    print(embeddings.shape)
    np.save("frames_image_embeddings.npy", embeddings)
