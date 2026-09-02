import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(dataset_dir: Path, split: Optional[str] = None) -> List[dict]:
    rows = []
    with (dataset_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if split is None or row["split"] == split:
                rows.append(row)
    return rows


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to float tensor in [0, 1] without torchvision."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3)
    return data.permute(2, 0, 1).float().div(255.0)


class SimpleImageTransform:
    """Minimal image transform for smoke tests when torchvision is unavailable."""

    def __init__(
        self,
        image_size: Optional[int] = None,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ):
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.image_size is not None and image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        tensor = pil_to_tensor(image)
        return (tensor - self.mean) / self.std


class VSTDDataset(Dataset):
    """Visual State Tracking Dataset reader.

    Returns a dict containing:
      image: transformed image tensor or PIL image
      label: int class id
      metadata: original metadata row
      prompt: class prompt string
      class_name: readable class name
    """

    def __init__(
        self,
        dataset_dir,
        split: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        return_pil: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.return_pil = return_pil
        self.rows = load_metadata(self.dataset_dir, split=split)
        if not self.rows:
            raise ValueError(f"No samples found for split={split!r} in {self.dataset_dir}")
        self.classes = load_json(self.dataset_dir / "classes.json")
        self.prompts = [item["prompt"] for item in self.classes]
        self.class_names = [item["name"] for item in self.classes]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["image"]).convert("RGB")
        if self.transform is not None:
            image_out = self.transform(image)
        elif self.return_pil:
            image_out = image
        else:
            image_out = pil_to_tensor(image)
        return {
            "image": image_out,
            "label": int(row["label"]),
            "metadata": row,
            "prompt": row["prompt"],
            "class_name": row["class_name"],
        }


def collate_vstd(batch: Iterable[Dict]) -> Dict:
    batch = list(batch)
    images = [item["image"] for item in batch]
    if torch.is_tensor(images[0]):
        images = torch.stack(images, dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "metadata": [item["metadata"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "class_name": [item["class_name"] for item in batch],
    }
