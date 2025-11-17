"""
Adaptive-depth U-Net training script for ISIC-2017 lesion segmentation.

This script implements the two training protocols described in the MSCA-UNet
and D2HU-Net papers so that experiments stay comparable:

- Protocol A: α·Cross-Entropy + β·Dice (α=0.4, β=0.6), Adam with cosine
  annealing, 100 epochs with early stopping.
- Protocol B: 0.5·BCE + 1.0·Dice, Adam with fixed LR, batch size 16, 200 epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, mixed_precision
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import (
    BackupAndRestore,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.optimizers.schedules import CosineDecay

sys.path.append(str(Path(__file__).resolve().parents[2]))  # because Shared is two levels up

from dataset_paths import (  # noqa: E402
    LOG_ROOT,
    MODEL_ROOT,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
)

# Science cluster enables XLA globally; Resize ops lack an XLA kernel, so disable JIT.
tf.config.optimizer.set_jit(False)


# --------------------------------------------------------------------------- #
# Defaults & protocol definitions
# --------------------------------------------------------------------------- #

DEFAULT_IMAGE_SIZE = 256
DEFAULT_BASE_CHANNELS = 64
DEFAULT_DEPTH = 4
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #


def normalise_isic_key(path: Path) -> str:
    """Return a lower-case ISIC identifier without trailing segmentation tokens."""
    stem = path.stem.lower()
    stem = stem.replace("_segmentation", "")
    return stem


def collect_isic_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[str, str]]:
    """
    Align dermoscopic images with their segmentation masks.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or contains no valid files.
    ValueError
        If a corresponding mask cannot be found for any image.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    def valid_image(path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and "superpixels" not in path.stem.lower()
        )

    def valid_mask(path: Path) -> bool:
        stem = path.stem.lower()
        return path.is_file() and path.suffix.lower() in {".png", ".jpg"} and stem.endswith("_segmentation")

    image_paths = sorted([p for p in image_dir.iterdir() if valid_image(p)], key=lambda p: p.stem.lower())
    mask_paths = sorted([p for p in mask_dir.iterdir() if valid_mask(p)], key=lambda p: normalise_isic_key(p))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    mask_index = {normalise_isic_key(path): path for path in mask_paths}

    missing_masks: List[str] = []
    pairs: List[Tuple[str, str]] = []
    for image_path in image_paths:
        key = normalise_isic_key(image_path)
        mask_path = mask_index.get(key)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        pairs.append((str(image_path), str(mask_path)))

    if missing_masks:
        truncated = ", ".join(missing_masks[:5])
        suffix = "" if len(missing_masks) <= 5 else "…"
        raise ValueError(
            f"Missing {len(missing_masks)} segmentation masks in {mask_dir}; "
            f"examples: {truncated}{suffix}"
        )

    return pairs


def _coerce_manifest_pairs(split: str, entries: Iterable[Iterable[str]]) -> List[Tuple[str, str]]:
    """Validate and normalise manifest pairs for a particular split."""
    normalised: List[Tuple[str, str]] = []
    for idx, pair in enumerate(entries):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Malformed entry #{idx} in '{split}' manifest")
        image_path, mask_path = pair
        if not image_path or not mask_path:
            raise ValueError(f"Empty path in '{split}' manifest entry #{idx}")
        image = Path(str(image_path)).expanduser()
        mask = Path(str(mask_path)).expanduser()
        if not image.exists():
            raise FileNotFoundError(f"Manifest references missing image: {image}")
        if not mask.exists():
            raise FileNotFoundError(f"Manifest references missing mask: {mask}")
        normalised.append((str(image), str(mask)))
    return normalised


def load_pairs_manifest(path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Manifest at {path} must be a JSON object.")
    if "train" not in data or "val" not in data:
        raise ValueError(f"Manifest at {path} missing 'train' or 'val' keys.")
    train_pairs = _coerce_manifest_pairs("train", data["train"])
    val_pairs = _coerce_manifest_pairs("val", data["val"])
    return train_pairs, val_pairs


def write_pairs_manifest(
    path: Path,
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    *,
    metadata: Dict[str, str] | None = None,
) -> None:
    payload: Dict[str, object] = {
        "version": 1,
        "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "train": train_pairs,
        "val": val_pairs,
    }
    if metadata:
        payload["metadata"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_isic_image(path: tf.Tensor, size: int) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [size, size], method=tf.image.ResizeMethod.AREA)
    image.set_shape((size, size, 3))
    return image


def load_isic_mask(path: tf.Tensor, size: int) -> tf.Tensor:
    mask = tf.io.read_file(path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    mask.set_shape((size, size, 1))
    return mask


def apply_isic_augmentations(image: tf.Tensor, mask: tf.Tensor, size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random rotation, flipping, scaling, and cropping augmentations."""
    flip_lr = tf.random.uniform((), minval=0.0, maxval=1.0) > 0.5
    flip_ud = tf.random.uniform((), minval=0.0, maxval=1.0) > 0.5
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)

    def maybe_flip_lr(tensor: tf.Tensor) -> tf.Tensor:
        return tf.cond(flip_lr, lambda: tf.image.flip_left_right(tensor), lambda: tensor)

    def maybe_flip_ud(tensor: tf.Tensor) -> tf.Tensor:
        return tf.cond(flip_ud, lambda: tf.image.flip_up_down(tensor), lambda: tensor)

    def apply_transforms(tensor: tf.Tensor) -> tf.Tensor:
        tensor = tf.image.rot90(tensor, k)
        tensor = maybe_flip_lr(tensor)
        tensor = maybe_flip_ud(tensor)
        return tensor

    image = apply_transforms(image)
    mask = apply_transforms(mask)

    scale = tf.random.uniform((), minval=1.0, maxval=1.15)
    scaled_size = tf.cast(tf.round(scale * tf.cast(size, tf.float32)), tf.int32)
    image = tf.image.resize(image, [scaled_size, scaled_size], method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, [scaled_size, scaled_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    combined = tf.concat([image, mask], axis=-1)
    crop_size = tf.stack([size, size, tf.shape(combined)[-1]])
    combined = tf.image.random_crop(combined, crop_size)

    image = combined[..., :3]
    mask = combined[..., 3:]
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return image, mask


def build_isic_dataset(
    image_dir: Path,
    mask_dir: Path,
    batch_size: int,
    image_size: int,
    augment: bool,
    shuffle: bool,
    seed: int,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[tf.data.Dataset, int]:
    """Construct a tf.data pipeline for ISIC segmentation pairs."""
    if pairs is None:
        pairs = collect_isic_pairs(image_dir, mask_dir)
    if not pairs:
        raise ValueError(f"No image/mask pairs found for {image_dir} / {mask_dir}")

    image_paths, mask_paths = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)

    def load_pair(image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = load_isic_image(image_path, image_size)
        mask = load_isic_mask(mask_path, image_size)
        if augment:
            image, mask = apply_isic_augmentations(image, mask, image_size)
        return image, mask

    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(pairs)


def prepare_isic_train_val_datasets(
    train_image_dir: Path,
    train_mask_dir: Path,
    val_image_dir: Path,
    val_mask_dir: Path,
    image_size: int,
    train_batch_size: int,
    val_batch_size: int,
    seed: int,
    train_pairs: Optional[List[Tuple[str, str]]] = None,
    val_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build tf.data pipelines for the official ISIC-2017 train/validation splits."""
    train_ds, train_count = build_isic_dataset(
        train_image_dir,
        train_mask_dir,
        batch_size=train_batch_size,
        image_size=image_size,
        augment=True,
        shuffle=True,
        seed=seed,
        pairs=train_pairs,
    )
    val_ds, val_count = build_isic_dataset(
        val_image_dir,
        val_mask_dir,
        batch_size=val_batch_size,
        image_size=image_size,
        augment=False,
        shuffle=False,
        seed=seed,
        pairs=val_pairs,
    )
    return train_ds, val_ds, train_count, val_count


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def make_hybrid_ce_dice_loss(alpha: float, beta: float) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ce = bce(y_true, y_pred)
        dice_term = dice_loss(y_true, y_pred)
        return alpha * ce + beta * dice_term

    loss_fn.__name__ = "hybrid_ce_dice"
    return loss_fn


def make_bce_dice_loss(bce_weight: float, dice_weight: float) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ce = bce(y_true, y_pred)
        dice_term = dice_loss(y_true, y_pred)
        return bce_weight * ce + dice_weight * dice_term

    loss_fn.__name__ = "bce_dice"
    return loss_fn


def dice_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    value = dice_coefficient(y_true, y_pred)
    return value


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    value = iou_score(y_true, y_pred)
    return value


dice_metric.__name__ = "dice"
iou_metric.__name__ = "iou"


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #

def conv_block(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    x = L.Conv2D(filters, 3, padding="same", use_bias=True)(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Conv2D(filters, 3, padding="same", use_bias=True)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    return x


def build_adaptive_depth_unet(
    input_size: int,
    base_channels: int,
    depth: int,
) -> Model:
    inputs = Input(shape=(input_size, input_size, 3), name="isic_image")

    x = inputs
    skips: List[tf.Tensor] = []
    channel_progression: List[int] = []
    filters = base_channels

    for _ in range(depth):
        x = conv_block(x, filters)
        skips.append(x)
        channel_progression.append(filters)
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        filters *= 2

    x = conv_block(x, filters)

    for filters, skip in reversed(list(zip(channel_progression, skips))):
        x = L.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = L.Concatenate()([x, skip])
        x = conv_block(x, filters)

    outputs = L.Conv2D(1, 1, activation="sigmoid", name="lesion_mask")(x)
    return Model(inputs=inputs, outputs=outputs, name=f"adaptive_unet_depth{depth}_c{base_channels}")


# --------------------------------------------------------------------------- #
# Protocol configuration
# --------------------------------------------------------------------------- #


@dataclass
class ProtocolConfig:
    key: str
    description: str
    loss_builder: Callable[[], Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]
    initial_lr: float
    epochs: int
    batch_size: int
    cosine_schedule: bool
    early_stopping_patience: int | None


PROTOCOLS: Dict[str, ProtocolConfig] = {
    "A": ProtocolConfig(
        key="A",
        description="MSCA-UNet hybrid loss (0.4·CE + 0.6·Dice) with cosine annealing",
        loss_builder=lambda: make_hybrid_ce_dice_loss(alpha=0.4, beta=0.6),
        initial_lr=1e-3,
        epochs=100,
        batch_size=8,
        cosine_schedule=True,
        early_stopping_patience=15,
    ),
    "B": ProtocolConfig(
        key="B",
        description="D2HU-Net BCE+Dice loss (0.5·BCE + 1.0·Dice)",
        loss_builder=lambda: make_bce_dice_loss(bce_weight=0.5, dice_weight=1.0),
        initial_lr=3e-4,
        epochs=200,
        batch_size=16,
        cosine_schedule=False,
        early_stopping_patience=None,
    ),
}


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def prepare_callbacks(
    run_dir: Path,
    ckpt_path: Path,
    patience: int | None,
) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = []
    callbacks.append(
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_dice",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    )
    callbacks.append(
        BackupAndRestore(str(run_dir / "train_backup"))
    )
    callbacks.append(
        TensorBoard(
            log_dir=str(run_dir),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            profile_batch=0,
        )
    )
    if patience is not None and patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_dice",
                mode="max",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


def build_optimizer(protocol: ProtocolConfig, steps_per_epoch: int, epochs: int) -> tf.keras.optimizers.Optimizer:
    if protocol.cosine_schedule:
        decay_steps = epochs * max(steps_per_epoch, 1)
        schedule = CosineDecay(
            initial_learning_rate=protocol.initial_lr,
            decay_steps=decay_steps,
            alpha=0.0,
        )
        return tf.keras.optimizers.Adam(learning_rate=schedule)
    return tf.keras.optimizers.Adam(learning_rate=protocol.initial_lr)


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)

    protocol = PROTOCOLS[args.protocol]
    epochs = args.epochs or protocol.epochs
    batch_size = args.batch_size or protocol.batch_size
    image_size = args.image_size

    if args.mixed_precision:
        available_gpus = tf.config.list_physical_devices("GPU")
        if not available_gpus:
            print("[warn] Mixed precision requested but no GPU detected; running in float32.")
        else:
            mixed_precision.set_global_policy("mixed_float16")

    if args.refresh_pairs_manifest and not args.pairs_manifest:
        raise ValueError("--refresh_pairs_manifest requires --pairs_manifest")

    train_images = Path(args.train_images or TRAIN_IMAGE_DIR).expanduser()
    train_masks = Path(args.train_masks or TRAIN_MASK_DIR).expanduser()
    val_images = Path(args.val_images or VALID_IMAGE_DIR).expanduser()
    val_masks = Path(args.val_masks or VALID_MASK_DIR).expanduser()
    test_images = Path(args.test_images or TEST_IMAGE_DIR).expanduser()
    test_masks = Path(args.test_masks or TEST_MASK_DIR).expanduser()

    manifest_path: Optional[Path] = None
    train_pairs: Optional[List[Tuple[str, str]]] = None
    val_pairs: Optional[List[Tuple[str, str]]] = None
    if args.pairs_manifest:
        manifest_path = Path(args.pairs_manifest).expanduser()
        if manifest_path.exists() and not args.refresh_pairs_manifest:
            print(f"[data] Loading dataset manifest from {manifest_path}")
            train_pairs, val_pairs = load_pairs_manifest(manifest_path)
            print(f"[data] Manifest contains {len(train_pairs)} train / {len(val_pairs)} val pairs")
        else:
            if manifest_path.exists():
                print(f"[data] Refreshing dataset manifest at {manifest_path}")
            else:
                print(f"[data] Creating dataset manifest at {manifest_path}")
            train_pairs = collect_isic_pairs(train_images, train_masks)
            val_pairs = collect_isic_pairs(val_images, val_masks)
            print(f"[data] Collected {len(train_pairs)} train / {len(val_pairs)} val pairs")
            metadata = {
                "train_images": str(train_images),
                "train_masks": str(train_masks),
                "val_images": str(val_images),
                "val_masks": str(val_masks),
                "seed": args.seed,
            }
            write_pairs_manifest(manifest_path, train_pairs, val_pairs, metadata=metadata)

    train_ds, val_ds, train_count, val_count = prepare_isic_train_val_datasets(
        train_image_dir=train_images,
        train_mask_dir=train_masks,
        val_image_dir=val_images,
        val_mask_dir=val_masks,
        image_size=image_size,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        seed=args.seed,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
    )

    steps_per_epoch = math.ceil(train_count / batch_size)
    val_steps = math.ceil(val_count / batch_size)

    test_ds: Optional[tf.data.Dataset] = None
    test_steps = 0
    test_count = 0
    try:
        test_ds, test_count = build_isic_dataset(
            image_dir=test_images,
            mask_dir=test_masks,
            batch_size=batch_size,
            image_size=image_size,
            augment=False,
            shuffle=False,
            seed=args.seed,
        )
        test_steps = math.ceil(test_count / batch_size)
        print(f"[data] Test split contains {test_count} samples")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[warn] Skipping test evaluation: {exc}")

    model = build_adaptive_depth_unet(
        input_size=image_size,
        base_channels=args.base_channels,
        depth=args.depth,
    )

    loss_fn = protocol.loss_builder()
    metrics = [dice_metric, iou_metric]
    optimizer = build_optimizer(protocol, steps_per_epoch, epochs)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
        jit_compile=False,
    )

    summary_lines: List[str] = []
    model.summary(print_fn=summary_lines.append)
    if args.print_model_summary:
        for line in summary_lines:
            print(line)

    model_dir = Path(args.model_dir or MODEL_ROOT).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    log_root = Path(args.log_dir or LOG_ROOT).expanduser()
    log_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"protocol{protocol.key}_seed{args.seed}_{timestamp}"
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / f"{run_name}.keras"

    callbacks = prepare_callbacks(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        patience=args.patience if args.patience is not None else protocol.early_stopping_patience,
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=args.fit_verbose,
    )

    if ckpt_path.exists():
        print(f"[model] Loading best checkpoint from {ckpt_path} for evaluation")
        model.load_weights(str(ckpt_path))

    eval_metrics = model.evaluate(val_ds, return_dict=True, verbose=1)
    test_metrics: Optional[Dict[str, float]] = None
    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)

    config_payload = {
        "protocol": protocol.key,
        "description": protocol.description,
        "epochs_requested": epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "initial_lr": protocol.initial_lr,
        "batch_size": batch_size,
        "image_size": image_size,
        "train_samples": train_count,
        "val_samples": val_count,
        "train_steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "test_samples": test_count,
        "test_steps": test_steps,
        "seed": args.seed,
        "mixed_precision": bool(args.mixed_precision),
        "fit_verbose": args.fit_verbose,
        "print_model_summary": bool(args.print_model_summary),
        "threshold": DEFAULT_THRESHOLD,
        "model_checkpoint": str(ckpt_path),
        "train_images": str(train_images),
        "train_masks": str(train_masks),
        "val_images": str(val_images),
        "val_masks": str(val_masks),
        "test_images": str(test_images),
        "test_masks": str(test_masks),
        "pairs_manifest": str(manifest_path) if manifest_path else None,
        "metrics": eval_metrics,
        "test_metrics": test_metrics,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    (run_dir / "model_summary.txt").write_text("\n".join(summary_lines))

    print("Validation metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")

    if test_metrics:
        print("Test metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adaptive-Depth U-Net on ISIC-2017 segmentation.")
    parser.add_argument(
        "--protocol",
        type=str,
        choices=sorted(PROTOCOLS.keys()),
        default="A",
        help="Training protocol to follow.",
    )
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs (0 keeps protocol default).")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size (0 keeps protocol default).")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=None, help="Override patience (None uses protocol default).")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed_float16 policy.")
    parser.add_argument("--model_dir", type=str, default=str(MODEL_ROOT))
    parser.add_argument("--log_dir", type=str, default=str(LOG_ROOT))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--print_model_summary",
        action="store_true",
        help="Print the full Keras model summary to stdout (default: off).",
    )
    parser.add_argument(
        "--fit_verbose",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Verbosity level for model.fit (0=silent, 1=per-batch bar, 2=per-epoch).",
    )
    parser.add_argument("--train_images", type=str, default=None, help="Override training image directory.")
    parser.add_argument("--train_masks", type=str, default=None, help="Override training mask directory.")
    parser.add_argument("--val_images", type=str, default=None, help="Override validation image directory.")
    parser.add_argument("--val_masks", type=str, default=None, help="Override validation mask directory.")
    parser.add_argument("--test_images", type=str, default=None, help="Override test image directory.")
    parser.add_argument("--test_masks", type=str, default=None, help="Override test mask directory.")
    parser.add_argument(
        "--pairs_manifest",
        type=str,
        default=None,
        help="JSON file enumerating train/val pairs so experiments share identical data.",
    )
    parser.add_argument(
        "--refresh_pairs_manifest",
        action="store_true",
        help="Rebuild the manifest even if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
