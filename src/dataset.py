import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset, DatasetDict
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

ZOOM_RE = re.compile(r"(\d+)x")
FOCUS_RE = re.compile(r"focus")
# Gemma 3 image soft token, masked out of the loss alongside <start_of_image>
IMAGE_SOFT_TOKEN_ID = 262144
FOCUS_NAMES = ["focused", "unfocused"]


def parse_filename(name: str) -> tuple[str, int, int, str]:
    """Return (tissue_type, zoom, focus, slide_key) from an image filename.

    focus is 0 when the name contains "focus", 1 otherwise. slide_key is the
    name stripped of zoom/focus tokens, so every shot of the same slide shares it.
    """
    name = name.lower()
    zoom_match = ZOOM_RE.search(name)
    if zoom_match is None:
        raise ValueError(f"No zoom level (e.g. '10x') in filename: {name}")
    tissue_type = name[: zoom_match.start()].split("-")[0].strip()
    focus = 0 if FOCUS_RE.search(name) else 1
    slide_key = FOCUS_RE.sub("", ZOOM_RE.sub("", Path(name).stem))
    slide_key = re.sub(r"[-_ ]+", "-", slide_key).strip("-")
    return tissue_type, int(zoom_match.group(1)), focus, slide_key


class tissue_dataset:
    def __init__(
        self,
        images_dir_path="src/dataset/Motic-Human-tissues",
        split_ratio=0.8,
        train_batch_size=8,
        val_batch_size=8,
        seed=42,
    ):
        """Motic human-tissue images as a chat dataset for MedGemma fine-tuning.

        The train/validation split is grouped by slide, so shots of the same
        slide at different zooms or focus never land on both sides.
        """
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        images, type_labels, zooms, focus, groups = [], [], [], [], []
        for image in sorted(Path(images_dir_path).glob("*.jpg")):
            if "calibration" in image.name.lower():
                continue
            tissue_type, zoom, focus_idx, slide_key = parse_filename(image.name)
            images.append(str(image))
            type_labels.append(tissue_type)
            zooms.append(zoom)
            focus.append(focus_idx)
            groups.append(slide_key)
        print(f"Found {len(images)} images ({len(set(groups))} slides) in {images_dir_path}")

        self.type_classes = sorted(set(type_labels))
        self.zoom_classes = sorted(set(zooms))
        self.focus_classes = [0, 1]
        type_to_idx = {t: i for i, t in enumerate(self.type_classes)}
        zoom_to_idx = {z: i for i, z in enumerate(self.zoom_classes)}

        data = {
            "image_path": images,
            "type": [type_to_idx[t] for t in type_labels],
            "zoom": [zoom_to_idx[z] for z in zooms],
            "focus": focus,
        }
        splitter = GroupShuffleSplit(n_splits=1, train_size=split_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(images, groups=groups))

        def subset(indices):
            return HFDataset.from_dict({k: [v[i] for i in indices] for k, v in data.items()})

        type_options = "\n".join(f"{i}: {t}" for i, t in enumerate(self.type_classes))
        zoom_options = "\n".join(f"{i}: {z}x" for i, z in enumerate(self.zoom_classes))
        focus_options = "\n".join(f"{i}: {f}" for i, f in enumerate(FOCUS_NAMES))
        self.PROMPT = (
            "Analyze this histopathology image and provide the following information:\n\n"
            f"Tissue Type:\n{type_options}\n\n"
            f"Zoom Level:\n{zoom_options}\n\n"
            f"Focus Quality:\n{focus_options}\n\n"
            "Please respond in the following JSON format:\n"
            '{\n"tissue_type": "X: tissue_name",\n"zoom_level": "Y: Zx",\n"focus_quality": "Z: focus_status"\n}'
        )

        self.dataset = DatasetDict(
            {
                "train": subset(train_idx).map(self._format_data),
                "validation": subset(val_idx).map(self._format_test_data),
            }
        )

    def set_processor(self, processor):
        self.processor = processor

    def build_train_val_loaders(self) -> None:
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["validation"]
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda ex: self._collate_fn(ex, pad_left=False),
        )
        # Generation needs left padding so every prompt ends right before the new tokens
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=lambda ex: self._collate_fn(ex, pad_left=True),
        )

    def _collate_fn(self, examples: list[dict[str, Any]], pad_left: bool):
        pad_id = self.processor.tokenizer.pad_token_id
        encoded = []
        for example in examples:
            image = Image.open(example["image_path"]).convert("RGB")
            text = self.processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=pad_left,
                tokenize=False,
            )
            encoded.append(self.processor(text=[text], images=[image], return_tensors="pt"))

        batch = {"pixel_values": torch.cat([e["pixel_values"] for e in encoded])}
        max_len = max(e["input_ids"].shape[1] for e in encoded)
        for key, fill in (("input_ids", pad_id), ("attention_mask", 0), ("token_type_ids", 0)):
            if key not in encoded[0]:
                continue
            rows = []
            for e in encoded:
                t = e[key]
                pad = torch.full((1, max_len - t.shape[1]), fill, dtype=t.dtype)
                rows.append(torch.cat([pad, t] if pad_left else [t, pad], dim=1))
            batch[key] = torch.cat(rows)

        labels = batch["input_ids"].clone()
        boi_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.tokenizer.special_tokens_map["boi_token"]
        )
        labels[batch["attention_mask"] == 0] = -100
        labels[labels == boi_id] = -100
        labels[labels == IMAGE_SOFT_TOKEN_ID] = -100
        if not pad_left:
            # Train only on the answer: mask everything up to the model's turn
            marker = self.processor.tokenizer.encode(
                "<start_of_turn>model\n", add_special_tokens=False
            )
            for row, ids in enumerate(batch["input_ids"].tolist()):
                answer_start = _find_last(ids, marker) + len(marker)
                labels[row, :answer_start] = -100
        batch["labels"] = labels
        return batch

    def _format_data(self, example: dict[str, Any]) -> dict[str, Any]:
        response = json.dumps(
            {
                "tissue_type": f"{example['type']}: {self.type_classes[example['type']]}",
                "zoom_level": f"{example['zoom']}: {self.zoom_classes[example['zoom']]}x",
                "focus_quality": f"{example['focus']}: {FOCUS_NAMES[example['focus']]}",
            },
            indent=4,
        )
        example = self._format_test_data(example)
        example["messages"].append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )
        return example

    def _format_test_data(self, example: dict[str, Any]) -> dict[str, Any]:
        example["messages"] = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}],
            }
        ]
        return example

    def postprocess(self, generated_text: str) -> dict[str, int]:
        """Parse the model's JSON answer into class indices (-1 when unparseable)."""
        for candidate in re.findall(r"\{[^{}]*\}", generated_text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and {"tissue_type", "zoom_level", "focus_quality"} <= parsed.keys():
                return {key: _leading_index(parsed[key]) for key in ("tissue_type", "zoom_level", "focus_quality")}
        return {"tissue_type": -1, "zoom_level": -1, "focus_quality": -1}


def _leading_index(value: Any) -> int:
    try:
        return int(str(value).split(":")[0].strip())
    except ValueError:
        return -1


def _find_last(ids: list[int], pattern: list[int]) -> int:
    for start in range(len(ids) - len(pattern), -1, -1):
        if ids[start : start + len(pattern)] == pattern:
            return start
    raise ValueError("Model turn marker not found in the training example")
