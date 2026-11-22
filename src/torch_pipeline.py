# -*- coding: utf-8 -*-
"""
Torch side of the pipeline: dataset, per-model transforms, and the ResNet50
builder. These are FAITHFUL copies of the released code
(train_cnn_final.py:88-171, 191-200) with no behavioural changes — Target A.

Imported only by smoke_test.py and (later) train.py. Requires torch+torchvision.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import config as C


class PADUFESDataset(Dataset):
    """Verbatim behaviour of train_cnn_final.py:88-143 (ID + is_cancer)."""
    def __init__(self, img_dirs, labels_df, label_columns=("is_cancer",), transform=None):
        self.img_dirs = [Path(d) for d in (img_dirs if isinstance(img_dirs, (list, tuple)) else [img_dirs])]
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.label_columns = list(label_columns)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = str(row[C.COL_IMG]) if C.COL_IMG in self.labels_df.columns else str(row["ID"])
        img_path = None
        for d in self.img_dirs:
            for ext in C.IMG_EXTS:
                cand = d / f"{img_id}{ext}"
                if cand.exists():
                    img_path = cand; break
            if img_path is not None:
                break
        if img_path is None:                      # fallback: id already has extension
            for d in self.img_dirs:
                cand = d / img_id
                if cand.exists():
                    img_path = cand; break
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"Image not found for ID={img_id}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_values = row[self.label_columns].values.astype(np.float32)
        return image, torch.tensor(label_values, dtype=torch.float32)


def get_resnet50_transforms(train: bool):
    """train_cnn_final.py:148-171 for resnet50."""
    base = ResNet50_Weights.IMAGENET1K_V1.transforms(antialias=True)
    if not train:
        return base
    aug = [
        transforms.RandomHorizontalFlip(C.AUG_HFLIP_P),
        transforms.ColorJitter(**C.AUG_COLORJITTER),
        transforms.RandomAffine(**C.AUG_AFFINE),
    ]
    return transforms.Compose(
        aug + [base, transforms.RandomErasing(p=C.AUG_RANDOM_ERASING_P,
                                              scale=C.AUG_RANDOM_ERASING_SCALE)]
    )


def build_resnet50(num_classes: int = 1):
    """train_cnn_final.py:191-200 — shared binary head, ImageNet pretrained."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(C.HEAD_DROPOUT_1),
        nn.Linear(in_f, C.HEAD_HIDDEN),
        nn.ReLU(),
        nn.Dropout(C.HEAD_DROPOUT_2),
        nn.Linear(C.HEAD_HIDDEN, num_classes),
    )
    return model


def seed_everything(seed: int = C.SEED):
    """train_cnn_final.py:77-83."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
