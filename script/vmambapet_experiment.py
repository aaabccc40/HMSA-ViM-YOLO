"""VMambaPet detection experiments.

用法示例：
    python scripts/vmambapet_experiment.py train --dataset pdt --data-root /data/processed_dataset_VMambaPet
    python scripts/vmambapet_experiment.py train --dataset banana_ms --data-root /data/processed_dataset_VMambaPet
    python scripts/vmambapet_experiment.py test --dataset pdt --data-root /data/processed_dataset_VMambaPet --weights runs/pdt/best.pt
    python scripts/vmambapet_experiment.py robust --dataset pdt --data-root /home/lenovo/datasets/processed_dataset_VMambaPet --weights /home/lenovo/PycharmProject/vmambapet/runs_vmambapet/pdt/20260625-215748-seed2026/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import warnings
try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign, batched_nms


# ----------------------------- 通用工具 -----------------------------


def set_seed(seed: int) -> None:
    """固定随机种子，保证数据划分和增强尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_stamp() -> str:
    """权重文件统一使用日期-时间命名。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collate_fn(batch):
    return tuple(zip(*batch))

def measure_model_efficiency(model, input_shape, device, batch_size=1, warmup=10, runs=50):
    """测量模型的参数量、FLOPs 与 FPS。

    Args:
        model: 待测模型（eval 模式）。
        input_shape: (C, H, W) 输入张量形状。
        device: torch 设备。
        batch_size: 测试 FPS 时的 batch size。
        warmup: 预热次数。
        runs: 正式计时次数。

    Returns:
        dict: {"params_M": float, "flops_G": float, "fps": float}
    """
    model.eval()
    # ---------- 参数量 ----------
    total_params = sum(p.numel() for p in model.parameters())
    params_M = total_params / 1e6

    # ---------- FLOPs ----------
    flops_G = 0.0
    if profile is not None:
        try:
            dummy_input = torch.randn(1, *input_shape).to(device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_G = flops / 1e9
        except Exception as e:
            print(f"FLOPs calculation failed: {e}. Falling back to 0.")
    else:
        print("thop not installed, FLOPs calculation skipped. Install with: pip install thop")

    # ---------- FPS ----------
    # 构造一批真实尺寸的随机张量
    dummy_batch = torch.randn(batch_size, *input_shape).to(device)
    model(dummy_batch)  # 首次调用可能触发 CUDA 初始化

    # 预热
    for _ in range(warmup):
        _ = model(dummy_batch)

    # 计时
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = (runs * batch_size) / elapsed if elapsed > 0 else 0.0

    return {"params_M": params_M, "flops_G": flops_G, "fps": fps}

def to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def read_yolo_label(label_path: Path, width: int, height: int, class_offset: int = 1):
    """读取 YOLO txt 标签并转为 torchvision 需要的 xyxy 格式。"""
    boxes, labels = [], []
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height
                x1 = max(0.0, min(float(width - 1), x1))
                y1 = max(0.0, min(float(height - 1), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))
                if x2 > x1 + 1 and y2 > y1 + 1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + class_offset)
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)


def pil_to_tensor_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def apply_corruption_tensor(image: torch.Tensor, mode: str, severity: int) -> torch.Tensor:
    """测试鲁棒性用的输入扰动。

    severity 取 1-3。训练增强和鲁棒性测试分开，正式论文中应报告 clean 与
    corrupted 两套结果。
    """
    if mode == "clean":
        return image
    severity = int(max(1, min(3, severity)))
    x = image.clone()
    if mode == "brightness":
        gain = [0.75, 0.6, 0.45][severity - 1]
        return torch.clamp(x * gain, 0.0, 1.0)
    if mode == "noise":
        sigma = [0.02, 0.04, 0.06][severity - 1]
        return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)
    if mode == "banddrop" and x.shape[0] > 3:
        band = severity % x.shape[0]
        x[band] = 0.0
        return x
    if mode == "blur":
        if x.shape[0] != 3:
            return x
        pil = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=[1.0, 1.6, 2.2][severity - 1]))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    if mode == "jpeg":
        if x.shape[0] != 3:
            return x
        import io

        quality = [70, 50, 30][severity - 1]
        pil = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    return x


# ----------------------------- 实验配置 -----------------------------


DATASET_ORDER = ["pdt", "banana_rgb", "banana_ms"]


@dataclass(frozen=True)
class ExperimentSpec:

    datasets: Tuple[str, ...]
    dataset_key: str
    input_channels: int
    class_names: Tuple[str, ...]
    pdt_label: Optional[int]
    banana_healthy_label: Optional[int]
    banana_fusarium_label: Optional[int]
    banana_label_policy: str

    @property
    def num_foreground_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_model_classes(self) -> int:
        # torchvision 检测模型的类别 0 固定为背景。
        return self.num_foreground_classes + 1


@dataclass(frozen=True)
class AblationSpec:
    """消融实验开关。"""

    name: str
    use_stress: bool
    use_directional: bool
    use_p2: bool
    force_no_aug: bool = False


ABLATION_SPECS: Dict[str, AblationSpec] = {
    "full": AblationSpec("full", use_stress=True, use_directional=True, use_p2=True),
    "baseline": AblationSpec("baseline", use_stress=False, use_directional=False, use_p2=False),
    "no_stress": AblationSpec("no_stress", use_stress=False, use_directional=True, use_p2=True),
    "no_directional": AblationSpec("no_directional", use_stress=True, use_directional=False, use_p2=True),
    "no_p2": AblationSpec("no_p2", use_stress=True, use_directional=True, use_p2=False),
    "no_aug": AblationSpec("no_aug", use_stress=True, use_directional=True, use_p2=True, force_no_aug=True),
    "stress_only": AblationSpec("stress_only", use_stress=True, use_directional=False, use_p2=False),
    "directional_only": AblationSpec("directional_only", use_stress=False, use_directional=True, use_p2=False),
}


def normalize_dataset_names(names: Sequence[str]) -> Tuple[str, ...]:
    """示例：
        --dataset pdt banana_ms -> ("pdt", "banana_ms")
    """
    seen = set()
    normalized = []
    valid = set(DATASET_ORDER)
    for name in names:
        if name not in valid:
            raise ValueError(f"未知数据集：{name}，可选值为 {DATASET_ORDER}")
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    if not normalized:
        raise ValueError("至少需要选择一个数据集")
    return tuple(normalized)


def build_experiment_spec(dataset_names: Sequence[str], banana_label_policy: str = "two_class") -> ExperimentSpec:
    """根据单数据集或多数据集组合生成统一输入通道和统一类别空间。"""
    datasets = normalize_dataset_names(dataset_names)
    has_pdt = "pdt" in datasets
    has_banana = any(name in datasets for name in ("banana_rgb", "banana_ms"))
    if banana_label_policy not in {"two_class", "disease_only"}:
        raise ValueError("banana_label_policy 只能为 two_class 或 disease_only")
    input_channels = 9 if "banana_ms" in datasets else 3

    class_names: List[str] = []
    pdt_label = None
    banana_healthy_label = None
    banana_fusarium_label = None
    if has_pdt:
        class_names.append("pdt_unhealthy")
        pdt_label = len(class_names)
    if has_banana:
        if banana_label_policy == "two_class":
            class_names.extend(["banana_healthy_plant", "banana_fusarium_wilt_plant"])
            banana_healthy_label = len(class_names) - 1
            banana_fusarium_label = len(class_names)
        else:
            class_names.append("banana_fusarium_wilt_plant")
            banana_fusarium_label = len(class_names)

    dataset_key = "+".join(datasets)
    if has_banana and banana_label_policy == "disease_only":
        dataset_key = f"{dataset_key}_disease_only"

    return ExperimentSpec(
        datasets=datasets,
        dataset_key=dataset_key,
        input_channels=input_channels,
        class_names=tuple(class_names),
        pdt_label=pdt_label,
        banana_healthy_label=banana_healthy_label,
        banana_fusarium_label=banana_fusarium_label,
        banana_label_policy=banana_label_policy,
    )


def get_ablation_spec(name: str) -> AblationSpec:
    if name not in ABLATION_SPECS:
        raise ValueError(f"未知消融方案：{name}，可选值为 {sorted(ABLATION_SPECS)}")
    return ABLATION_SPECS[name]


def experiment_root(output_dir: str, spec: ExperimentSpec, ablation: AblationSpec) -> Path:
    """生成实验输出根目录。

    完整模型不是消融实验，直接放在 runs_vmambapet/<数据集组合> 下；具体消融
    方案放在 runs_vmambapet/<数据集组合>/<消融名> 下。
    """
    root = Path(output_dir) / spec.dataset_key
    if ablation.name != "full":
        root = root / ablation.name
    return root

def _infer_ablation_from_weights(weights_path: str) -> str:
    """从标准目录结构 .../ablation_name/timestamp/best.pt 中提取消融名"""
    ablation = Path(weights_path).resolve().parent.parent.name
    if ablation not in ABLATION_SPECS:
        raise ValueError(
            f"从权重路径自动推断的消融 '{ablation}' 不在可选列表中。"
            f"请确认目录结构为 runs_vmambapet/<dataset>/<ablation>/<timestamp>/best.pt，"
            f"或手动指定 --ablation。可选值：{sorted(ABLATION_SPECS)}"
        )
    return ablation

# ----------------------------- 数据集 -----------------------------


class PdtYoloDataset(Dataset):
    """PDT 官方 YOLO split 数据集。"""

    def __init__(
        self,
        root: Path,
        split: str,
        corrupt: str = "clean",
        severity: int = 1,
        train_aug: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.image_dir = root / "pdt_yolo" / "images" / split
        self.label_dir = root / "pdt_yolo" / "labels" / split
        self.images = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.corrupt = corrupt
        self.severity = severity
        self.train_aug = train_aug
        if not self.images:
            raise FileNotFoundError(f"未找到 PDT 图像：{self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = pil_to_tensor_rgb(img_path)
        _, h, w = image.shape
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes, labels = read_yolo_label(label_path, width=w, height=h, class_offset=1)

        if self.train_aug and random.random() < 0.5:
            image, boxes = hflip_image_boxes(image, boxes)
        if self.train_aug and random.random() < 0.35:
            mode = random.choice(["brightness", "noise", "blur", "jpeg"])
            image = apply_corruption_tensor(image, mode, random.randint(1, 2))
        image = apply_corruption_tensor(image, self.corrupt, self.severity)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image, target


class BananaDataset(Dataset):
    """香蕉枯萎病 patch 数据集。

    mode=rgb 使用 `rgb_yolo` 图像；mode=ms 使用 `ms_npz` 的 9 通道张量。
    ms 通道顺序：Blue, Green, Red, RedEdge, NIR, NDVI, NDRE, GNDVI, VARI。
    """

    def __init__(
        self,
        root: Path,
        split: str,
        mode: str,
        corrupt: str = "clean",
        severity: int = 1,
        train_aug: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.mode = mode
        self.corrupt = corrupt
        self.severity = severity
        self.train_aug = train_aug

        manifest_path = root / "banana_fusarium" / "banana_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"未找到香蕉 manifest：{manifest_path}")
        self.rows = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(row)
        if not self.rows:
            raise FileNotFoundError(f"香蕉数据 split 为空：{split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        if self.mode == "rgb":
            image = pil_to_tensor_rgb(Path(row["image"]))
        elif self.mode == "ms":
            data = np.load(row["npz"])
            image = torch.from_numpy(data["image"].astype(np.float32)).permute(2, 0, 1).contiguous()
            image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"未知香蕉输入模式：{self.mode}")

        _, h, w = image.shape
        label_path = Path(row["label"])
        boxes, labels = read_yolo_label(label_path, width=w, height=h, class_offset=1)

        if self.train_aug and random.random() < 0.5:
            image, boxes = hflip_image_boxes(image, boxes)
        if self.train_aug and random.random() < 0.25:
            mode = "banddrop" if self.mode == "ms" else random.choice(["brightness", "noise", "blur"])
            image = apply_corruption_tensor(image, mode, random.randint(1, 2))
        image = apply_corruption_tensor(image, self.corrupt, self.severity)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image, target


def hflip_image_boxes(image: torch.Tensor, boxes: torch.Tensor):
    image = image.flip(-1)
    if boxes.numel() > 0:
        width = image.shape[-1]
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        boxes = flipped
    return image, boxes


class UnifiedDataset(Dataset):
    """把单个原始数据集包装成统一实验数据集。

    该类不改变原始数据文件，只在取样时完成三项实验级处理：
    1. 输入通道补齐：联合实验包含 `banana_ms` 时，模型为 9 通道，RGB 数据补 0。
    2. 标签重映射：PDT 和香蕉类别进入同一标签空间。
    3. 来源记录：target 中写入 source_id，便于合并测试后再按子数据集统计。
    """

    def __init__(self, base: Dataset, dataset_name: str, spec: ExperimentSpec, source_id: int) -> None:
        self.base = base
        self.dataset_name = dataset_name
        self.spec = spec
        self.source_id = source_id

    def __len__(self) -> int:
        return len(self.base)

    def _remap_target(self, target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """按实验标签策略重映射目标。

        two_class：香蕉健康株和枯萎株都作为前景类，兼容已有实验。
        disease_only：只把枯萎病株作为前景类，健康株样本保留为无框背景图。
        第二种策略更符合“病害检测”任务，能减少联合训练时的跨域语义冲突。
        """
        target = dict(target)
        labels = target["labels"]
        boxes = target["boxes"]
        if labels.numel() == 0:
            return target
        if self.dataset_name == "pdt":
            if self.spec.pdt_label is None:
                raise ValueError("实验配置缺少 PDT 类别")
            target["labels"] = torch.full_like(labels, self.spec.pdt_label)
        else:
            if self.spec.banana_fusarium_label is None:
                raise ValueError("实验配置缺少香蕉类别")
            # BananaDataset 的局部标签：1=healthy，2=fusarium_wilt。
            if self.spec.banana_label_policy == "disease_only":
                keep = labels == 2
                target["boxes"] = boxes[keep]
                target["labels"] = torch.full(
                    (int(keep.sum().item()),),
                    self.spec.banana_fusarium_label,
                    dtype=labels.dtype,
                    device=labels.device,
                )
            else:
                if self.spec.banana_healthy_label is None:
                    raise ValueError("实验配置缺少香蕉健康类别")
                target["labels"] = torch.where(
                    labels == 1,
                    torch.full_like(labels, self.spec.banana_healthy_label),
                    torch.full_like(labels, self.spec.banana_fusarium_label),
                )
        return target

    def _pad_channels(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[0]
        if channels == self.spec.input_channels:
            return image
        if channels > self.spec.input_channels:
            return image[: self.spec.input_channels]
        pad = torch.zeros(
            (self.spec.input_channels - channels, image.shape[1], image.shape[2]),
            dtype=image.dtype,
            device=image.device,
        )
        return torch.cat([image, pad], dim=0)

    def __getitem__(self, idx: int):
        image, target = self.base[idx]
        image = self._pad_channels(image)
        target = self._remap_target(target)
        target["source_id"] = torch.tensor([self.source_id], dtype=torch.int64)
        target["local_image_id"] = target.get("image_id", torch.tensor([idx], dtype=torch.int64))
        return image, target


def build_base_dataset(
    data_root: Path,
    dataset: str,
    split: str,
    corrupt: str = "clean",
    severity: int = 1,
    train_aug: bool = False,
) -> Dataset:
    """构建单个数据集的原始读取器。"""
    if dataset == "pdt":
        return PdtYoloDataset(data_root, split, corrupt=corrupt, severity=severity, train_aug=train_aug)
    if dataset == "banana_rgb":
        return BananaDataset(data_root, split, mode="rgb", corrupt=corrupt, severity=severity, train_aug=train_aug)
    if dataset == "banana_ms":
        return BananaDataset(data_root, split, mode="ms", corrupt=corrupt, severity=severity, train_aug=train_aug)
    raise ValueError(f"未知数据集：{dataset}")


def make_experiment_dataset(
    data_root: Path,
    spec: ExperimentSpec,
    split: str,
    corrupt: str = "clean",
    severity: int = 1,
    train_aug: bool = False,
) -> Dataset:
    """构建单数据集或多数据集合并后的训练/测试数据集。"""
    datasets = []
    for source_id, dataset_name in enumerate(spec.datasets):
        base = build_base_dataset(
            data_root,
            dataset_name,
            split,
            corrupt=corrupt,
            severity=severity,
            train_aug=train_aug,
        )
        datasets.append(UnifiedDataset(base, dataset_name, spec, source_id=source_id))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def parse_source_sampling(text: str, dataset_names: List[str]) -> Dict[str, float]:
    """解析合并训练的数据源采样比例。

    示例：--source-sampling pdt:0.7,banana_rgb:0.3
    不填写时保持原来的数据源均衡策略；只填写部分数据源时，剩余比例平均分给未填写数据源。
    """
    if not text:
        return {name: 1.0 / len(dataset_names) for name in dataset_names}
    specified: Dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--source-sampling 格式错误：{text}，应类似 pdt:0.7,banana_rgb:0.3")
        name, value = item.split(":", 1)
        name = name.strip()
        if name not in dataset_names:
            raise ValueError(f"--source-sampling 中的数据集名称 {name} 不在当前训练集 {dataset_names} 内")
        prob = float(value)
        if prob < 0:
            raise ValueError(f"--source-sampling 中 {name} 的比例不能为负数")
        specified[name] = prob
    missing = [name for name in dataset_names if name not in specified]
    used = sum(specified.values())
    if used > 1.0 + 1e-8:
        raise ValueError(f"--source-sampling 比例总和不能超过 1，当前为 {used:.4f}")
    if missing:
        remain = max(0.0, 1.0 - used)
        for name in missing:
            specified[name] = remain / len(missing)
    total = sum(specified.values())
    if total <= 0:
        raise ValueError("--source-sampling 的有效比例总和必须大于 0")
    return {name: specified[name] / total for name in dataset_names}


def build_source_balanced_sampler(
    dataset_obj: Dataset,
    seed: int,
    samples_per_epoch: int = 0,
    source_sampling: str = "",
):
    """为合并数据集构建数据源均衡采样器。"""
    if not isinstance(dataset_obj, ConcatDataset):
        return None
    lengths = [len(ds) for ds in dataset_obj.datasets]
    if any(length == 0 for length in lengths):
        raise ValueError(f"合并数据集中存在空数据源：{lengths}")
    dataset_names = [getattr(ds, "dataset_name", f"dataset_{idx}") for idx, ds in enumerate(dataset_obj.datasets)]
    source_probs = parse_source_sampling(source_sampling, dataset_names)
    weights = []
    for name, length in zip(dataset_names, lengths):
        # WeightedRandomSampler 按样本权重抽样；每个数据源的总权重等于 source_probs[name]。
        weights.extend([source_probs[name] / float(length)] * length)
    num_samples = int(samples_per_epoch) if samples_per_epoch and samples_per_epoch > 0 else sum(lengths)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


# ----------------------------- 模型 -----------------------------


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DirectionalScan(nn.Module):
    """轻量二维多方向扫描。"""

    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Conv2d(channels * 5, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _norm_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
        size = x.shape[dim]
        denom_shape = [1] * x.ndim
        denom_shape[dim] = size
        denom = torch.arange(1, size + 1, device=x.device, dtype=x.dtype).view(denom_shape)
        return torch.cumsum(x, dim=dim) / denom

    def forward(self, x):
        left = self._norm_cumsum(x, dim=3)
        right = torch.flip(self._norm_cumsum(torch.flip(x, dims=[3]), dim=3), dims=[3])
        down = self._norm_cumsum(x, dim=2)
        up = torch.flip(self._norm_cumsum(torch.flip(x, dims=[2]), dim=2), dims=[2])
        ctx = self.mix(torch.cat([x, left, right, up, down], dim=1))
        return x + ctx * self.gate(x)


class VssBlock(nn.Module):
    """VMamba-style 视觉状态块：局部深度卷积 + 多方向扫描 + MLP。"""

    def __init__(self, channels: int, use_directional: bool = True, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.scan = DirectionalScan(channels) if use_directional else nn.Identity()
        self.norm2 = nn.BatchNorm2d(channels)
        hidden = int(channels * 2.0)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.drop_path = drop_path

    def forward(self, x):
        y = self.scan(self.dwconv(self.norm1(x)))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class StressPrior(nn.Module):
    """光谱/植被指数先验。

    RGB 输入时根据 RGB 构造 VARI、ExG、GRRI；多光谱输入时优先使用预处理
    阶段保存的 NDVI、NDRE、GNDVI、VARI 通道。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        prior_ch = 4
        self.proj = nn.Sequential(
            nn.Conv2d(prior_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.in_channels = in_channels

    @staticmethod
    def _safe_div(a, b):
        eps = 1e-4
        sign = torch.where(b >= 0, torch.ones_like(b), -torch.ones_like(b))
        den = torch.where(torch.abs(b) < eps, sign * eps, b)
        return a / den

    def forward(self, x):
        if self.in_channels >= 9:
            prior = x[:, 5:9].clamp(-3.0, 3.0)
        else:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            vari = self._safe_div(g - r, g + r - b + 1e-4).clamp(-3.0, 3.0)
            exg = (2 * g - r - b).clamp(-1.0, 1.0)
            grri = self._safe_div(g, r + 1e-4).clamp(0.0, 3.0) / 3.0
            intensity = ((r + g + b) / 3.0).clamp(0.0, 1.0)
            prior = torch.cat([vari / 3.0, exg, grri, intensity], dim=1)
        return self.proj(torch.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0))


class RavMambaFPNBackbone(nn.Module):
    """RAVMamba-Det 骨干与 P2-P5 FPN。"""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 48,
        depth: int = 2,
        use_stress: bool = True,
        use_directional: bool = True,
        use_p2: bool = True,
    ):
        super().__init__()
        self.out_channels = 128
        self.use_p2 = use_p2
        chs = [width, width * 2, width * 4, width * 6]

        self.stress = StressPrior(in_channels) if use_stress else None
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, chs[0] // 2, 3, 2),
            ConvNormAct(chs[0] // 2, chs[0], 3, 2),
        )
        self.stage2 = self._make_stage(chs[0], chs[1], depth, stride=2, use_directional=use_directional)
        self.stage3 = self._make_stage(chs[1], chs[2], depth, stride=2, use_directional=use_directional)
        self.stage4 = self._make_stage(chs[2], chs[3], max(1, depth), stride=2, use_directional=use_directional)

        fpn_channels = chs if use_p2 else chs[1:]
        self.lateral = nn.ModuleList([nn.Conv2d(c, self.out_channels, 1) for c in fpn_channels])
        self.smooth = nn.ModuleList([ConvNormAct(self.out_channels, self.out_channels, 3, 1) for _ in fpn_channels])
        self.prior_gates = nn.ModuleList([nn.Conv2d(1, c, 1) for c in chs])

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, depth: int, stride: int, use_directional: bool):
        blocks = [ConvNormAct(in_ch, out_ch, 3, stride)]
        blocks += [VssBlock(out_ch, use_directional=use_directional) for _ in range(depth)]
        return nn.Sequential(*blocks)

    def _apply_prior(self, feat: torch.Tensor, prior: torch.Tensor, idx: int) -> torch.Tensor:
        if prior is None:
            return feat
        p = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.prior_gates[idx](p))
        return feat * (1.0 + 0.25 * gate)

    def forward(self, x):
        prior = self.stress(x) if self.stress is not None else None
        p2 = self._apply_prior(self.stem(x), prior, 0)
        p3 = self._apply_prior(self.stage2(p2), prior, 1)
        p4 = self._apply_prior(self.stage3(p3), prior, 2)
        p5 = self._apply_prior(self.stage4(p4), prior, 3)
        feats = [p2, p3, p4, p5] if self.use_p2 else [p3, p4, p5]

        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        outs = [smooth(f) for smooth, f in zip(self.smooth, laterals)]
        return OrderedDict((str(i), o) for i, o in enumerate(outs))


def _legacy_single_dataset_build_model(dataset: str, width: int, depth: int):
    if dataset == "pdt":
        in_channels, num_classes = 3, 2
    elif dataset == "banana_rgb":
        in_channels, num_classes = 3, 3
    elif dataset == "banana_ms":
        in_channels, num_classes = 9, 3
    else:
        raise ValueError(f"未知数据集：{dataset}")

    backbone = RavMambaFPNBackbone(in_channels=in_channels, width=width, depth=depth)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0.0] * in_channels,
        image_std=[1.0] * in_channels,
        min_size=512,
        max_size=960,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=300,
        rpn_pre_nms_top_n_train=1200,
        rpn_post_nms_top_n_train=600,
        rpn_pre_nms_top_n_test=1200,
        rpn_post_nms_top_n_test=600,
    )
    # 小目标密集数据需要比默认 RPN 更细的 anchor 配置。
    model.rpn.head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    return model


# ----------------------------- 训练和评估 -----------------------------


def make_dataset(data_root: Path, dataset: str, split: str, corrupt: str = "clean", severity: int = 1, train_aug=False):
    if dataset == "pdt":
        return PdtYoloDataset(data_root, split, corrupt=corrupt, severity=severity, train_aug=train_aug)
    if dataset == "banana_rgb":
        return BananaDataset(data_root, split, mode="rgb", corrupt=corrupt, severity=severity, train_aug=train_aug)
    if dataset == "banana_ms":
        return BananaDataset(data_root, split, mode="ms", corrupt=corrupt, severity=severity, train_aug=train_aug)
    raise ValueError(f"未知数据集：{dataset}")


def build_model(spec: ExperimentSpec, width: int, depth: int, ablation: AblationSpec):
    """根据数据集组合和消融方案构建检测模型。 """
    in_channels = spec.input_channels
    num_classes = spec.num_model_classes
    backbone = RavMambaFPNBackbone(
        in_channels=in_channels,
        width=width,
        depth=depth,
        use_stress=ablation.use_stress,
        use_directional=ablation.use_directional,
        use_p2=ablation.use_p2,
    )
    anchor_sizes = ((16,), (32,), (64,), (128,)) if ablation.use_p2 else ((32,), (64,), (128,))
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes),
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(len(anchor_sizes))],
        output_size=7,
        sampling_ratio=2,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0.0] * in_channels,
        image_std=[1.0] * in_channels,
        min_size=512,
        max_size=960,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=300,
        rpn_pre_nms_top_n_train=1200,
        rpn_post_nms_top_n_train=600,
        rpn_pre_nms_top_n_test=1200,
        rpn_post_nms_top_n_test=600,
    )
    model.rpn.head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    return model


def build_loader(dataset_obj: Dataset, batch_size: int, workers: int, shuffle: bool, sampler=None):
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=workers > 0,
    )


def load_weights(model: nn.Module, weights: Optional[Path], device, strict: bool = True) -> int:
    """加载已有模型继续训练。

    如果想从头训练，不传 --resume 即可；如果想从已有权重继续训练，传入
    --resume /path/to/best.pt。脚本会恢复模型参数，并从 checkpoint 记录的
    epoch 后继续。
    """
    if weights is None:
        return 0
    ckpt = torch.load(weights, map_location=device)
    if strict:
        model.load_state_dict(ckpt["model"], strict=True)
        return int(ckpt.get("epoch", 0))

    current = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in ckpt["model"].items():
        if key in current and current[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped.append(key)
    current.update(loaded)
    model.load_state_dict(current, strict=True)
    print(f"partial resume loaded {len(loaded)} tensors, skipped {len(skipped)} mismatched tensors")
    # 部分加载用于迁移初始化，检测头和优化器重新训练，因此从 epoch 0 开始。
    return 0


def save_checkpoint(model, optimizer, epoch: int, metrics: Dict, out_dir: Path, tag: str) -> Path:
    ensure_dir(out_dir)
    stamp = now_stamp()
    metric_text = f"map50_{metrics.get('map50', 0.0):.4f}".replace(".", "p")
    path = out_dir / f"{stamp}_epoch{epoch:03d}_{metric_text}_{tag}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "saved_at": stamp,
        },
        path,
    )
    latest = out_dir / "latest.pt"
    shutil.copy2(path, latest)
    return path


def build_optimizer(model: nn.Module, args):
    """构建优化器。

    AdamW 是当前脚本的默认优化器，适合自定义 backbone 和从头训练；SGD 保留为
    对照选项，便于和 torchvision/Faster R-CNN 常见训练设置比较。
    """
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"未知优化器：{args.optimizer}")


def build_scheduler(optimizer, args, steps_per_epoch: int, start_epoch: int):
    """构建学习率调度器。

    onecycle：按 batch 更新，适合从头训练或联合训练，能先升高学习率再退火。
    warmup_cosine：按 epoch 更新，前若干 epoch 线性 warmup，之后余弦退火。
    plateau：按验证集 map50_95 更新，平台期自动降学习率。
    cosine：保留原始余弦退火，作为可复现实验对照。
    """
    remain_epochs = max(1, args.epochs - start_epoch)
    if args.scheduler == "none":
        return None, "none"
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=remain_epochs,
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            anneal_strategy="cos",
        )
        return scheduler, "batch"
    if args.scheduler == "warmup_cosine":
        warmup_epochs = max(1, min(args.warmup_epochs, remain_epochs))
        min_ratio = max(0.0, min(args.min_lr_ratio, 1.0))

        def lr_lambda(epoch_idx: int):
            epoch_num = epoch_idx + 1
            if epoch_num <= warmup_epochs:
                return max(min_ratio, epoch_num / float(warmup_epochs))
            progress = (epoch_num - warmup_epochs) / float(max(1, remain_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), "epoch"
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=1e-4,
            min_lr=args.lr * args.min_lr_ratio,
        )
        return scheduler, "metric"
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remain_epochs,
            eta_min=args.lr * args.min_lr_ratio,
        )
        return scheduler, "epoch"
    raise ValueError(f"未知学习率调度器：{args.scheduler}")


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec = build_experiment_spec(args.dataset, banana_label_policy=args.banana_label_policy)
    ablation = get_ablation_spec(args.ablation)
    train_aug = args.train_aug and not ablation.force_no_aug
    run_dir = experiment_root(args.output_dir, spec, ablation) / now_stamp()
    ensure_dir(run_dir)


    train_set = make_experiment_dataset(Path(args.data_root), spec, "train", train_aug=train_aug)
    val_set = make_experiment_dataset(Path(args.data_root), spec, "val")
    train_sampler = None
    if args.source_balanced and len(spec.datasets) > 1:
        train_sampler = build_source_balanced_sampler(
            train_set,
            seed=args.seed,
            samples_per_epoch=args.balanced_samples_per_epoch,
            source_sampling=args.source_sampling,
        )
    train_loader = build_loader(train_set, args.batch_size, args.workers, shuffle=train_sampler is None, sampler=train_sampler)
    val_loader = build_loader(val_set, max(1, args.batch_size // 2), args.workers, shuffle=False)

    model = build_model(spec, width=args.width, depth=args.depth, ablation=ablation).to(device)
    start_epoch = load_weights(
        model,
        Path(args.resume) if args.resume else None,
        device,
        strict=args.resume_mode == "strict",
    )
    # 测量模型效率
    eff_input_shape = (spec.input_channels, 512, 512)  # 典型输入尺寸
    eff_batch = args.batch_size if args.batch_size > 0 else 1
    eff_metrics = measure_model_efficiency(model, eff_input_shape, device, batch_size=eff_batch)
    print(f"Model efficiency: Params={eff_metrics['params_M']:.2f}M, "
          f"FLOPs={eff_metrics['flops_G']:.2f}G, FPS={eff_metrics['fps']:.2f}")
    run_config = {
        "datasets": spec.datasets,
        "dataset_key": spec.dataset_key,
        "input_channels": spec.input_channels,
        "class_names": spec.class_names,
        "banana_label_policy": spec.banana_label_policy,
        "ablation": ablation.__dict__,
        "train_aug_effective": train_aug,
        "source_balanced": args.source_balanced and len(spec.datasets) > 1,
        "source_sampling": args.source_sampling,
        "balanced_samples_per_epoch": args.balanced_samples_per_epoch,
        "early_stop_patience": args.early_stop_patience,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "args": vars(args),
        "model_efficiency": eff_metrics,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # 学习率调整建议：
    # 1. loss 震荡或出现 NaN：把 --lr 降到当前的 1/2 或 1/3。
    # 2. val mAP 长期不升且训练 loss 很慢：可把 --lr 提高 1.5 倍。
    # 3. batch size 改小后，通常也要等比例降低 lr。
    if args.lr < 5e-5 and not args.allow_very_low_lr:
        print(
            f"warning: lr={args.lr:g} 过低，检测模型从头训练容易长期欠拟合；"
            "建议使用 --lr 1e-4 或 --lr 2e-4，若确认只做微调可加 --allow-very-low-lr。"
        )
    optimizer = build_optimizer(model, args)
    scheduler, scheduler_step = build_scheduler(optimizer, args, len(train_loader), start_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_map = -1.0
    best_epoch = 0
    bad_epochs = 0
    log_path = run_dir / "train_log.jsonl"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for step, (images, targets) in enumerate(train_loader, 1):
            images, targets = to_device(images, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(v for v in loss_dict.values())
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scheduler_step == "batch":
                scheduler.step()
            total_loss += float(loss.detach().cpu())
            if step % args.print_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss {loss.item():.4f} lr {lr:.6g}")
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        metrics = evaluate(model, val_loader, device=device, num_classes=spec.num_foreground_classes)
        if args.val_per_dataset and len(spec.datasets) > 1:
            metrics["per_dataset"] = evaluate_each_dataset(
                model,
                Path(args.data_root),
                spec,
                ablation,
                args,
                device,
                split="val",
            )
        if scheduler is not None and scheduler_step == "metric":
            scheduler.step(metrics["map50_95"])
        metrics["train_loss"] = total_loss / max(1, len(train_loader))
        metrics["epoch_time_sec"] = round(time.time() - t0, 2)
        metrics["dataset"] = spec.dataset_key
        metrics["ablation"] = ablation.name
        ckpt = save_checkpoint(model, optimizer, epoch, metrics, run_dir, tag="val")
        if metrics["map50_95"] > best_map:
            best_map = metrics["map50_95"]
            best_epoch = epoch
            bad_epochs = 0
            shutil.copy2(ckpt, run_dir / "best.pt")
        else:
            bad_epochs += 1
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False) + "\n")
        print(f"epoch {epoch} finished: {metrics}, checkpoint={ckpt}")
        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(
                f"early stop at epoch {epoch}: best map50_95={best_map:.6f} "
                f"from epoch {best_epoch}, no improvement for {bad_epochs} epochs"
            )
            break

    print(f"训练完成。最佳权重：{run_dir / 'best.pt'}")


def num_classes_for(dataset: str) -> int:
    return 1 if dataset == "pdt" else 2


@torch.no_grad()
def test(args):
    if args.ablation is None:
        args.ablation = _infer_ablation_from_weights(args.weights)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec = build_experiment_spec(args.dataset, banana_label_policy=args.banana_label_policy)
    ablation = get_ablation_spec(args.ablation)
    ds = make_experiment_dataset(Path(args.data_root), spec, args.split)
    loader = build_loader(ds, args.batch_size, args.workers, shuffle=False)
    model = build_model(spec, width=args.width, depth=args.depth, ablation=ablation).to(device)
    load_weights(model, Path(args.weights), device)
    metrics = evaluate(model, loader, device=device, num_classes=spec.num_foreground_classes)
    metrics.update({"dataset": spec.dataset_key, "ablation": ablation.name, "split": args.split})
    eff_input_shape = (spec.input_channels, 512, 512)
    eff_batch = args.batch_size if args.batch_size > 0 else 1
    eff_metrics = measure_model_efficiency(model, eff_input_shape, device, batch_size=eff_batch)
    metrics.update(eff_metrics)
    print(f"Model efficiency: Params={eff_metrics['params_M']:.2f}M, "
          f"FLOPs={eff_metrics['flops_G']:.2f}G, FPS={eff_metrics['fps']:.2f}")
    out_dir = experiment_root(args.output_dir, spec, ablation) / "test_reports"
    ensure_dir(out_dir)
    report = out_dir / f"{now_stamp()}_{args.split}_metrics.json"
    with report.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if len(spec.datasets) > 1:
        per_dataset = evaluate_each_dataset(model, Path(args.data_root), spec, ablation, args, device)
        per_report = out_dir / f"{now_stamp()}_{args.split}_per_dataset_metrics.json"
        with per_report.open("w", encoding="utf-8") as f:
            json.dump(per_dataset, f, ensure_ascii=False, indent=2)
    if args.save_result_images:
        save_test_result_images(model, loader, device, spec, args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"测试报告：{report}")


@torch.no_grad()
def robust(args):
    if args.ablation is None:
        args.ablation = _infer_ablation_from_weights(args.weights)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec = build_experiment_spec(args.dataset, banana_label_policy=args.banana_label_policy)
    ablation = get_ablation_spec(args.ablation)
    model = build_model(spec, width=args.width, depth=args.depth, ablation=ablation).to(device)
    load_weights(model, Path(args.weights), device)
    corruptions = ["clean", "brightness", "noise", "blur", "jpeg"]
    if "banana_ms" in spec.datasets:
        corruptions.append("banddrop")
    rows = []
    for mode in corruptions:
        for severity in ([1] if mode == "clean" else [1, 2, 3]):
            ds = make_experiment_dataset(Path(args.data_root), spec, args.split, corrupt=mode, severity=severity)
            loader = build_loader(ds, args.batch_size, args.workers, shuffle=False)
            metrics = evaluate(model, loader, device=device, num_classes=spec.num_foreground_classes)
            metrics.update({"dataset": spec.dataset_key, "ablation": ablation.name, "corruption": mode, "severity": severity})
            rows.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))

    out_dir = experiment_root(args.output_dir, spec, ablation) / "robust_reports"
    ensure_dir(out_dir)
    report = out_dir / f"{now_stamp()}_{args.split}_robust.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"鲁棒性报告：{report}")


@torch.no_grad()
def evaluate_each_dataset(
    model: nn.Module,
    data_root: Path,
    spec: ExperimentSpec,
    ablation: AblationSpec,
    args,
    device,
    split: Optional[str] = None,
):
    """联合测试时按子数据集分别输出指标。"""
    eval_split = split or getattr(args, "split", "val")
    results = {}
    for dataset_name in spec.datasets:
        single_spec = build_experiment_spec([dataset_name], banana_label_policy=spec.banana_label_policy)
        base = build_base_dataset(data_root, dataset_name, eval_split)
        wrapped = UnifiedDataset(base, dataset_name, spec, source_id=0)
        loader = build_loader(wrapped, args.batch_size, args.workers, shuffle=False)
        metrics = evaluate(model, loader, device=device, num_classes=spec.num_foreground_classes)
        metrics.update(
            {
                "dataset": dataset_name,
                "trained_dataset_key": spec.dataset_key,
                "ablation": ablation.name,
                "split": eval_split,
                "local_class_names": single_spec.class_names,
            }
        )
        results[dataset_name] = metrics
    return results


def tensor_to_pil_rgb(image: torch.Tensor, dataset_name: str) -> Image.Image:
    """把模型输入张量还原为彩色 PNG 底图。

    RGB 数据直接使用前 3 个通道；banana_ms 的通道顺序为 Blue/Green/Red/...，
    因此可视化时使用 Red/Green/Blue 组合。
    """
    image = image.detach().cpu().float()
    if dataset_name == "banana_ms" and image.shape[0] >= 5:
        rgb = image[[2, 1, 0]]
    else:
        rgb = image[:3]
    rgb = torch.clamp(rgb, 0.0, 1.0)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def class_color(label: int) -> Tuple[int, int, int]:
    colors = [
        (230, 57, 70),
        (42, 157, 143),
        (29, 53, 87),
        (245, 166, 35),
        (131, 56, 236),
    ]
    return colors[(max(label, 1) - 1) % len(colors)]


@torch.no_grad()
def save_test_result_images(model: nn.Module, loader: DataLoader, device, spec: ExperimentSpec, args) -> Path:
    """保存彩色加红框结果图、黑白二值图，并复制原图与标签。

    输出路径固定为：
    <test_result_root>/<数据集名称>/<权重所在文件夹>/
    在该目录下生成 image/ 和 label/ 子文件夹存放原始数据，
    同时保存 {编号}_test_result_color.png 和 {编号}_test_result_binary.png。
    图片/标签编号按遍历顺序从 0001 重新编排。
    """
    weight_folder = Path(args.weights).resolve().parent.name
    out_dir = Path(args.test_result_root) / spec.dataset_key / weight_folder
    img_out_dir = out_dir / "image"
    lbl_out_dir = out_dir / "label"
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    font = ImageFont.load_default()
    model.eval()
    img_counter = 1                       # 按先后顺序重新编号

    for batch_idx, (images, targets) in enumerate(loader):
        images_dev = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images_dev)
        for item_idx, (image, target, output) in enumerate(zip(images, targets, outputs)):
            # 确定数据集名称
            source_id = int(target.get("source_id", torch.tensor([0]))[0].item())
            dataset_name = spec.datasets[source_id] if source_id < len(spec.datasets) else spec.datasets[0]

            _, h, w = image.shape
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            keep = scores >= args.vis_score_thr
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            base_name = f"{img_counter:04d}"

            # ----- 彩色加红框结果图 -----
            canvas = tensor_to_pil_rgb(image, dataset_name)
            draw = ImageDraw.Draw(canvas)
            for box, score, label in zip(boxes, scores, labels):
                label_int = int(label.item())
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=args.vis_box_width)
                class_name = spec.class_names[label_int - 1] if 1 <= label_int <= len(spec.class_names) else str(label_int)
                text = f"{class_name} {float(score):.2f}"
                text_box = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(text_box, fill="red")
                draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
            canvas.save(out_dir / f"{base_name}_test_result_color.png")

            # ----- 黑白二值图 -----
            heatmap = np.zeros((h, w), dtype=np.float32)
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], float(score))
            binary_mask = (heatmap >= args.vis_score_thr).astype(np.uint8) * 255
            mask_img = Image.fromarray(binary_mask, mode='L')
            mask_img.save(out_dir / f"{base_name}_test_result_binary.png")

            # ----- 复制原图与标签（重命名） -----
            # 原图：直接使用输入张量还原为 PNG，保证格式统一且避免路径查询问题
            orig_img = tensor_to_pil_rgb(image, dataset_name)
            orig_img.save(img_out_dir / f"{base_name}.png")

            # 标签：从原始数据集获取真实标签文件路径
            local_id = int(target.get("local_image_id", torch.tensor([item_idx]))[0].item())
            if isinstance(loader.dataset, ConcatDataset):
                unified_ds = loader.dataset.datasets[source_id]
            else:
                unified_ds = loader.dataset
            base_ds = unified_ds.base
            if isinstance(base_ds, PdtYoloDataset):
                label_path = base_ds.label_dir / f"{base_ds.images[local_id].stem}.txt"
            elif isinstance(base_ds, BananaDataset):
                label_path = Path(base_ds.rows[local_id]["label"])
            else:
                label_path = None

            if label_path is not None and label_path.exists():
                shutil.copy2(label_path, lbl_out_dir / f"{base_name}.txt")

            img_counter += 1

    print(f"saved {img_counter - 1} test result images to {out_dir}")
    return out_dir


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    all_preds, all_gts = [], []
    for images, targets in loader:
        images, targets = to_device(images, targets, device)
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            pred = {
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            }
            gt = {
                "boxes": tgt["boxes"].detach().cpu(),
                "labels": tgt["labels"].detach().cpu(),
            }
            all_preds.append(pred)
            all_gts.append(gt)
    return detection_metrics(all_preds, all_gts, num_classes=num_classes)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    recall_grid = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_grid:
        p = prec[rec >= r].max() if np.any(rec >= r) else 0.0
        ap += p / 101.0
    return float(ap)


def detection_metrics(preds: Sequence[Dict], gts: Sequence[Dict], num_classes: int) -> Dict[str, float]:
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    aps_by_thr = []
    for thr in iou_thresholds:
        aps = []
        for cls in range(1, num_classes + 1):
            cls_preds = []
            n_gt = 0
            for img_id, (pred, gt) in enumerate(zip(preds, gts)):
                p_mask = pred["labels"].numpy() == cls
                g_mask = gt["labels"].numpy() == cls
                p_boxes = pred["boxes"].numpy()[p_mask]
                p_scores = pred["scores"].numpy()[p_mask]
                g_boxes = gt["boxes"].numpy()[g_mask]
                n_gt += len(g_boxes)
                for box, score in zip(p_boxes, p_scores):
                    cls_preds.append((img_id, float(score), box))
            if n_gt == 0:
                continue
            cls_preds.sort(key=lambda x: x[1], reverse=True)
            matched = {i: np.zeros(len(gt["boxes"][gt["labels"] == cls]), dtype=bool) for i, gt in enumerate(gts)}
            tp, fp = [], []
            for img_id, _, box in cls_preds:
                gt = gts[img_id]
                g_mask = gt["labels"].numpy() == cls
                g_boxes = gt["boxes"].numpy()[g_mask]
                if len(g_boxes) == 0:
                    tp.append(0)
                    fp.append(1)
                    continue
                ious = box_iou_np(box[None, :], g_boxes)[0]
                best = int(ious.argmax())
                if ious[best] >= thr and not matched[img_id][best]:
                    matched[img_id][best] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            if not tp:
                aps.append(0.0)
                continue
            tp = np.cumsum(np.array(tp))
            fp = np.cumsum(np.array(fp))
            rec = tp / max(n_gt, 1)
            prec = tp / np.maximum(tp + fp, 1)
            aps.append(average_precision(rec, prec))
        aps_by_thr.append(float(np.mean(aps)) if aps else 0.0)

    precision, recall, f1 = prf_at_iou(preds, gts, num_classes=num_classes, iou_thr=0.5, score_thr=0.25)
    return {
        "map50": float(aps_by_thr[0]),
        "map50_95": float(np.mean(aps_by_thr)),
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
    }


def prf_at_iou(preds, gts, num_classes: int, iou_thr: float, score_thr: float):
    tp = fp = fn = 0
    for cls in range(1, num_classes + 1):
        for pred, gt in zip(preds, gts):
            p_mask = (pred["labels"].numpy() == cls) & (pred["scores"].numpy() >= score_thr)
            g_mask = gt["labels"].numpy() == cls
            p_boxes = pred["boxes"].numpy()[p_mask]
            p_scores = pred["scores"].numpy()[p_mask]
            g_boxes = gt["boxes"].numpy()[g_mask]
            order = np.argsort(-p_scores)
            p_boxes = p_boxes[order]
            matched = np.zeros(len(g_boxes), dtype=bool)
            for box in p_boxes:
                if len(g_boxes) == 0:
                    fp += 1
                    continue
                ious = box_iou_np(box[None, :], g_boxes)[0]
                best = int(ious.argmax())
                if ious[best] >= iou_thr and not matched[best]:
                    matched[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += int((~matched).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(precision), float(recall), float(f1)


# ----------------------------- 命令行 -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="VMambaPet detection experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        # 移除 ablation 参数，保留其他公共参数
        p.add_argument(
            "--dataset",
            choices=DATASET_ORDER,
            nargs="+",
            required=True,
            help="可选择一个、两个或三个数据集",
        )
        p.add_argument("--data-root", type=str, required=True)
        p.add_argument("--output-dir", type=str, default="runs_vmambapet")
        p.add_argument("--device", type=str, default="cuda:0")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--width", type=int, default=48)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument(
            "--banana-label-policy",
            choices=["two_class", "disease_only"],
            default="two_class",
        )

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument(
        "--ablation",
        choices=sorted(ABLATION_SPECS.keys()),
        default="full",
        help="消融实验方案；full 为完整模型",
    )
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=5e-4)
    p_train.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p_train.add_argument("--momentum", type=float, default=0.9)
    p_train.add_argument(
        "--scheduler",
        choices=["onecycle", "warmup_cosine", "plateau", "cosine", "none"],
        default="onecycle",
        help="学习率调度器；联合训练优先 onecycle 或 warmup_cosine，平台期可试 plateau",
    )
    p_train.add_argument("--warmup-epochs", type=int, default=5)
    p_train.add_argument("--min-lr-ratio", type=float, default=0.01)
    p_train.add_argument("--onecycle-pct-start", type=float, default=0.15)
    p_train.add_argument("--onecycle-div-factor", type=float, default=10.0)
    p_train.add_argument("--onecycle-final-div-factor", type=float, default=100.0)
    p_train.add_argument("--plateau-factor", type=float, default=0.5)
    p_train.add_argument("--plateau-patience", type=int, default=8)
    p_train.add_argument(
        "--allow-very-low-lr",
        action="store_true",
        help="允许低于 5e-5 的学习率；只建议在已充分收敛模型上微调时使用",
    )
    p_train.add_argument("--seed", type=int, default=2026)
    p_train.add_argument("--resume", type=str, default="", help="继续训练的权重路径；留空表示从头训练")
    p_train.add_argument(
        "--resume-mode",
        choices=["strict", "partial"],
        default="strict",
        help="strict 用于同结构断点续训；partial 用于从单数据集模型迁移到合并训练",
    )
    p_train.add_argument("--amp", action="store_true", help="A5000 建议开启，可降低显存占用")
    p_train.add_argument("--train-aug", action="store_true", help="开启训练增强与轻量鲁棒扰动")
    p_train.add_argument("--grad-clip", type=float, default=5.0)
    p_train.add_argument("--print-freq", type=int, default=50)
    p_train.add_argument(
        "--source-balanced",
        dest="source_balanced",
        action="store_true",
        default=True,
        help="合并训练时按数据源均衡采样；单数据集训练自动无效",
    )
    p_train.add_argument(
        "--no-source-balanced",
        dest="source_balanced",
        action="store_false",
        help="关闭合并训练的数据源均衡采样，用于对照实验",
    )
    p_train.add_argument(
        "--balanced-samples-per-epoch",
        type=int,
        default=0,
        help="均衡采样下每个 epoch 的样本数；0 表示使用合并数据集原始总图像数",
    )
    p_train.add_argument(
        "--source-sampling",
        type=str,
        default="",
        help="合并训练时各数据源的采样比例，例如 pdt:0.7,banana_rgb:0.3；留空表示各数据源等比例采样",
    )
    p_train.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="验证集 map50_95 连续多少轮不提升后停止；0 表示关闭早停",
    )
    p_train.add_argument(
        "--val-per-dataset",
        action="store_true",
        help="合并训练时每轮额外记录各子数据集验证指标；会增加验证耗时，排查负迁移时使用",
    )

    p_test = sub.add_parser("test")
    add_common(p_test)
    p_test.add_argument(
        "--ablation",
        choices=sorted(ABLATION_SPECS.keys()),
        default=None,                    
        help="消融方案；若未指定，则从权重路径自动推断（需保持标准目录结构）",
    )
    p_test.add_argument("--weights", type=str, required=True)
    p_test.add_argument("--split", choices=["val", "test"], default="test")
    p_test.add_argument(
        "--test-result-root",
        type=str,
        default="/home/lenovo/datasets/processed_dataset_VMambaPet/test_result",
        help="测试结果 PNG 保存根目录",
    )
    p_test.add_argument("--vis-score-thr", type=float, default=0.25)
    p_test.add_argument("--vis-box-width", type=int, default=2)
    p_test.add_argument("--save-result-images", dest="save_result_images", action="store_true", default=True)
    p_test.add_argument("--no-save-result-images", dest="save_result_images", action="store_false")

    p_robust = sub.add_parser("robust")
    add_common(p_robust)
    p_robust.add_argument(
        "--ablation",
        choices=sorted(ABLATION_SPECS.keys()),
        default=None,
        help="消融方案；若未指定，则从权重路径自动推断",
    )
    p_robust.add_argument("--weights", type=str, required=True)
    p_robust.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "test":
        test(args)
    elif args.cmd == "robust":
        robust(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()