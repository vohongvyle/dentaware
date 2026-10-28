"""Find every subset of the Children's Dental Panoramic Radiographs dataset explicitly.

The class project let an auto-discovery heuristic choose the training folder, and it silently
picked the *adult* tooth set. Here each subset is located by name and reported, and a run fails
loudly if a required subset is missing.

Expected layout (Zhang et al., Scientific Data 2023; Kaggle mirror):

Dental_dataset/
  Childrens dental caries segmentation dataset/{Train,Test,Supplemental content93}/{images,mask}
  Pediatric dental disease detection dataset/{Train,Test}/{images,label}
  Adult tooth segmentation dataset/
      Dataset and code/{train,test}/{images,masks}
      Panoramic radiography database/{images,mask}
      Archive/{images,mask}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import list_images

CHILD_SEG = "Childrens dental caries segmentation dataset"
CHILD_DET = "Pediatric dental disease detection dataset"
ADULT_SEG = "Adult tooth segmentation dataset"

MASK_DIR_NAMES = ("mask", "masks")
LABEL_DIR_NAMES = ("label", "labels", "json", "annotations")


@dataclass
class Sample:
    image: Path
    mask: Path | None = None       # tooth mask (segmentation subsets)
    label: Path | None = None      # LabelMe json (detection subset)
    subset: str = ""
    split: str = ""                # official split name: train / test / supplement

    @property
    def id(self) -> str:
        return f"{self.subset}/{self.split}/{self.image.stem}"


@dataclass
class DatasetIndex:
    root: Path
    child_seg: list[Sample] = field(default_factory=list)
    child_det: list[Sample] = field(default_factory=list)
    adult_seg: list[Sample] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        def by_split(samples):
            out: dict[str, int] = {}
            for s in samples:
                out[s.split] = out.get(s.split, 0) + 1
            return out
        return {
            "root": str(self.root),
            "child_seg": by_split(self.child_seg),
            "child_det": by_split(self.child_det),
            "adult_seg": by_split(self.adult_seg),
            "notes": self.notes,
        }


def find_dataset_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if (root / CHILD_DET).exists():
        return root
    candidates = [p for p in root.rglob("Dental_dataset") if p.is_dir()]
    if candidates:
        return candidates[0]
    for p in root.rglob(CHILD_DET):
        return p.parent
    raise FileNotFoundError(
        f"Could not find '{CHILD_DET}' under {root}. Point data_root at the folder that contains Dental_dataset."
    )


def _child_dir(parent: Path, names: tuple[str, ...]) -> Path | None:
    if not parent.exists():
        return None
    lower = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def _split_name(name: str) -> str:
    n = name.lower()
    if n.startswith("train"):
        return "train"
    if n.startswith("test"):
        return "test"
    if "supplement" in n:
        return "supplement"
    return n


def _pair_by_stem(images: list[Path], others: list[Path]) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Exact stem match first, then a unique case-insensitive prefix match. Returns pairs and unmatched images."""
    by_stem = {p.stem.lower(): p for p in others}
    pairs, unmatched = [], []
    for img in images:
        key = img.stem.lower()
        if key in by_stem:
            pairs.append((img, by_stem[key]))
            continue
        cands = [p for k, p in by_stem.items() if k.startswith(key) or key.startswith(k)]
        if len(cands) == 1:
            pairs.append((img, cands[0]))
        else:
            unmatched.append(img)
    return pairs, unmatched


def _scan_seg_dir(split_dir: Path, subset: str, split: str, idx: DatasetIndex) -> list[Sample]:
    img_dir = _child_dir(split_dir, ("images", "image", "img"))
    mask_dir = _child_dir(split_dir, MASK_DIR_NAMES)
    if img_dir is None or mask_dir is None:
        return []
    pairs, unmatched = _pair_by_stem(list_images(img_dir), list_images(mask_dir))
    if unmatched:
        idx.notes.append(f"{subset}/{split}: {len(unmatched)} images had no mask and were skipped")
    return [Sample(image=i, mask=m, subset=subset, split=split) for i, m in pairs]


def build_index(data_root: str | Path) -> DatasetIndex:
    root = find_dataset_root(data_root)
    idx = DatasetIndex(root=root)

    # Pediatric tooth segmentation
    cs = root / CHILD_SEG
    if cs.exists():
        for d in sorted(p for p in cs.iterdir() if p.is_dir()):
            idx.child_seg += _scan_seg_dir(d, "child_seg", _split_name(d.name), idx)

    # Pediatric disease detection (LabelMe json)
    cd = root / CHILD_DET
    for d in sorted(p for p in cd.iterdir() if p.is_dir()):
        img_dir = _child_dir(d, ("images", "image", "img"))
        lab_dir = _child_dir(d, LABEL_DIR_NAMES)
        if img_dir is None or lab_dir is None:
            continue
        labels = sorted(lab_dir.glob("*.json"))
        pairs, unmatched = _pair_by_stem(list_images(img_dir), labels)
        if unmatched:
            idx.notes.append(f"child_det/{d.name}: {len(unmatched)} images had no label file and were skipped")
        idx.child_det += [Sample(image=i, label=l, subset="child_det", split=_split_name(d.name)) for i, l in pairs]

    # Adult tooth segmentation (three source collections)
    ad = root / ADULT_SEG
    if ad.exists():
        for coll in sorted(p for p in ad.iterdir() if p.is_dir()):
            if _child_dir(coll, ("images",)) is not None:
                idx.adult_seg += _scan_seg_dir(coll, "adult_seg", coll.name.lower(), idx)
            else:
                for d in sorted(p for p in coll.iterdir() if p.is_dir()):
                    idx.adult_seg += _scan_seg_dir(d, "adult_seg", f"{coll.name.lower()}-{_split_name(d.name)}", idx)

    if not idx.child_det:
        raise FileNotFoundError("Pediatric detection subset is empty; check the dataset path.")
    if not idx.child_seg:
        idx.notes.append("Pediatric tooth masks not found: segmentation will rely on adult data only.")
    return idx


def match_det_to_child_masks(idx: DatasetIndex, max_diff: float = 3.0) -> dict[str, Path]:
    """Detection images that are the *same radiograph* as a child segmentation image.

    A shared file name is not enough (both subsets reuse names such as cate1_012), so the pixels
    are compared too. Matches allow an 'oracle' analysis with true instead of predicted tooth masks.
    """
    import cv2
    import numpy as np

    from .io_utils import read_gray

    def thumb(p):
        return cv2.resize(read_gray(p), (96, 48), interpolation=cv2.INTER_AREA).astype(np.float32)

    by_stem: dict[str, list[Sample]] = {}
    for s in idx.child_seg:
        by_stem.setdefault(s.image.stem.lower(), []).append(s)
    out: dict[str, Path] = {}
    for d in idx.child_det:
        cands = by_stem.get(d.image.stem.lower(), [])
        if not cands:
            continue
        t = thumb(d.image)
        for c in cands:
            if float(np.abs(thumb(c.image) - t).mean()) <= max_diff:
                out[d.id] = c.mask
                break
    return out
