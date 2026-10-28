"""U-Net tooth segmentation: training, prediction at full resolution, evaluation."""

from __future__ import annotations

import copy
import time

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..io_utils import read_gray, read_mask
from ..paths import Sample


def seg_shape(size: int) -> tuple[int, int]:
    """Panoramic radiographs are about twice as wide as tall: keep that aspect ratio (H, W)."""
    return size, size * 2


class ToothDataset(Dataset):
    def __init__(self, samples: list[Sample], size: int, train: bool):
        self.samples = samples
        self.h, self.w = seg_shape(size)
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.04, 0.04), rotate=(-5, 5), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ]) if train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = cv2.resize(read_gray(s.image), (self.w, self.h), interpolation=cv2.INTER_AREA)
        mask, _ = read_mask(s.mask)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self.aug is not None:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        x = torch.from_numpy(np.repeat(img[None], 3, 0).astype(np.float32) / 255.0)
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y


def build_model(encoder: str, pretrained: bool = True) -> nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def _dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


def _loss(logits, target):
    return 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, target) + 0.5 * _dice_loss(logits, target)


def train_model(train: list[Sample], val: list[Sample], *, size: int, encoder: str, epochs: int, lr: float,
                batch: int, device: str, init_state: dict | None = None, log=print) -> tuple[nn.Module, dict]:
    """Train and keep the weights with the best validation Dice (validation only, never test)."""
    model = build_model(encoder, pretrained=init_state is None).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    dl = DataLoader(ToothDataset(train, size, True), batch_size=batch, shuffle=True, num_workers=2, drop_last=len(train) > batch)
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = _loss(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_dice = float(np.mean([r["dice"] for r in evaluate(model, val, size=size, device=device)])) if val else float("nan")
        history.append({"epoch": ep + 1, "loss": float(np.mean(losses)) if losses else float("nan"), "val_dice": val_dice})
        log(f"  epoch {ep + 1}/{epochs}  loss {history[-1]['loss']:.4f}  val Dice {val_dice:.4f}  ({time.time() - t0:.0f}s)")
        if val_dice > best or best_state is None:
            best, best_state = val_dice, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"best_val_dice": best, "history": history}


@torch.no_grad()
def predict_prob(model: nn.Module, image: np.ndarray, *, size: int, device: str, tta: bool = True) -> np.ndarray:
    """Tooth probability map at the image's *original* resolution (float32 in [0,1])."""
    model.eval()
    h0, w0 = image.shape
    H, W = seg_shape(size)
    x = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(np.repeat(x[None, None], 3, 1)).to(device)
    p = torch.sigmoid(model(x))
    if tta:
        p = 0.5 * (p + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3]))
    p = p[0, 0].float().cpu().numpy()
    return cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR)


def evaluate(model: nn.Module, samples: list[Sample], *, size: int, device: str) -> list[dict]:
    """Per-image Dice and IoU, computed at the original mask resolution."""
    rows = []
    for s in samples:
        img = read_gray(s.image)
        gt, _ = read_mask(s.mask)
        if gt.shape != img.shape:
            gt = cv2.resize(gt, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
        pred = predict_prob(model, img, size=size, device=device) > 0.5
        gt = gt.astype(bool)
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        denom = pred.sum() + gt.sum()
        rows.append({"image_id": s.id,
                     "dice": float(2 * inter / denom) if denom else 1.0,
                     "iou": float(inter / union) if union else 1.0})
    return rows
