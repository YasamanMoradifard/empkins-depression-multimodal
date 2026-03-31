"""
PyTorch Dataset for the d02 (EKSpression) audio-visual corpus.

The directory layout is assumed to be:

    <root>/
        EKSpression_labels_All.csv (or EKSpression_labels_<Condition>.csv)
        visual/<sample_id>_visual.npy   # [T_v, F_v]
        audio/<sample_id>.npy           # [T_a, F_a]

The labels CSV must contain at least:
    ID,label,Condition,fold,phase

Labels may use strings ("Depressed"/"Healthy") or integers (1/0). The
dataset concatenates visual and audio features along the feature axis,
truncating to the minimum time dimension when the lengths differ.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


def _load_feature(path: Path, required: bool = True) -> Optional[np.ndarray]:
    if path.exists():
        return np.load(path)
    if required:
        # Instead of raising an error, return None and let the caller skip the sample
        import warnings
        warnings.warn(f"Feature file not found (will skip sample): {path}", UserWarning)
    return None


class EKSpression(data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        fold: str = "train",
        gender: str = "both",
        condition: str = "all",
        phase: str = "all",
        transform=None,
        target_transform=None,
        aug: bool = False,
        min_aug_frames: int = 400,
        modalities: str = "av",
    ):
        """
        Args:
            root: Dataset root directory.
            fold: train / validation / test.
            gender: 'm' / 'f' / 'both' (ignored if gender column absent).
            condition: Filter rows to this condition (case-insensitive). Use
                'all' to keep every condition.
            phase: Filter rows to this phase. Use 'all' to keep every phase.
            transform: Optional callable applied to the concatenated feature.
            target_transform: Optional callable for labels.
            aug: Whether to apply random temporal cropping augmentation.
            min_aug_frames: Minimum length required for an augmented crop.
            modalities: 'av' (both), 'audio', or 'video'. When set to
                'audio' or 'video', only that modality is returned so you
                can fine-tune unimodal baselines easily.
        """
        self.root = Path(root)
        self.fold = fold
        self.gender = gender
        self.condition = condition.lower() if condition else "all"
        self.phase_filter = phase.lower() if phase else "all"
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug
        self.min_aug_frames = min_aug_frames
        self.modalities = (modalities or "av").lower()
        self.require_audio = "a" in self.modalities
        self.require_video = "v" in self.modalities

        # Determine which label file to use based on condition
        # If condition is "all", use EKSpression_labels_All.csv
        # Otherwise, use EKSpression_labels_<Condition>.csv (e.g., EKSpression_labels_CR.csv)
        if self.condition == "all":
            label_filename = "EKSpression_labels_All.csv"
        else:
            # Normalize condition to uppercase to match file naming (CR, ADK, CRADK, SHAM)
            condition_upper = self.condition.upper()
            label_filename = f"EKSpression_labels_{condition_upper}.csv"
        
        labels_path = self.root / label_filename
        if not labels_path.exists():
            expected_condition = condition_upper if self.condition != "all" else "All"
            raise FileNotFoundError(
                f"Label file not found: {labels_path}\n"
                f"Expected file for condition '{condition}' (normalized: '{expected_condition}')"
            )

        self.features = []
        self.labels = []

        with labels_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                idx = row.get("ID")  # Changed from "index" to "ID"
                if not idx:
                    continue
                label_token = row.get("label", "0")
                try:
                    # Try numeric first (0/1)
                    label = int(float(label_token))
                except Exception:
                    # Handle string labels: "Depressed"/"Healthy" or "depression"/"healthy"
                    label_str = str(label_token).lower().strip()
                    if label_str.startswith("dep") or label_str == "1":
                        label = 1
                    elif label_str.startswith("health") or label_str == "0":
                        label = 0
                    else:
                        # Default to 0 if unrecognized
                        label = 0
                gender_val = row.get("gender", "both")
                fold_val = row.get("fold", "train")
                condition_val = row.get("Condition", "all")  # Changed from "condition" to "Condition" (capitalized)
                phase_val = row.get("phase", "Unknown")

                if not self._match_sample(gender_val, fold_val, condition_val, phase_val):
                    continue

                # Get Aufgabe from CSV (task number)
                aufgabe = row.get("Aufgabe", "")
                
                # Convert label to lowercase for filename (Healthy -> healthy, Depressed -> depressed)
                label_lower = label_token.lower().strip()
                
                # Convert condition and phase to lowercase for filename
                condition_lower = condition_val.lower()
                phase_lower = phase_val.lower()
                
                # Remove leading zeros from ID to match actual file names
                idx_clean = str(int(idx)) if idx.isdigit() else idx
                
                # Construct filename pattern: ID_label_condition_phase_aufgabe_visual.npy
                visual_filename = f"{idx_clean}_{label_lower}_{condition_lower}_{phase_lower}_{aufgabe}_visual.npy"
                audio_filename = f"{idx_clean}_{label_lower}_{condition_lower}_{phase_lower}_{aufgabe}.npy"
                
                visual_path = self.root / "visual" / visual_filename
                audio_path = self.root / "audio" / audio_filename

                visual = _load_feature(visual_path, required=self.require_video)
                audio = _load_feature(audio_path, required=self.require_audio)

                # Skip this sample if required features are missing
                if self.require_video and visual is None:
                    continue  # Skip this sample
                if self.require_audio and audio is None:
                    continue  # Skip this sample

                feature = self._compose_feature(visual, audio)

                self.features.append(feature.astype(np.float32))
                self.labels.append(label)

                if self.aug and self.fold == "train":
                    self._augment(feature, label)

    def _match_sample(self, gender_val: str, fold_val: str, condition_val: str, phase_val: str) -> bool:
        gender_ok = (
            True
            if self.gender == "both" or gender_val == "unknown"
            else gender_val.lower() == self.gender.lower()
        )
        condition_ok = (
            True
            if self.condition == "all"
            else (condition_val or "").lower() == self.condition
        )
        phase_ok = (
            True
            if self.phase_filter == "all"
            else (phase_val or "").lower() == self.phase_filter
        )
        return gender_ok and condition_ok and phase_ok and fold_val == self.fold

    def _compose_feature(
        self,
        visual: Optional[np.ndarray],
        audio: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.modalities == "video" or (self.require_video and not self.require_audio):
            if visual is None:
                raise ValueError("Video modality requested but missing.")
            return visual
        if self.modalities == "audio" or (self.require_audio and not self.require_video):
            if audio is None:
                raise ValueError("Audio modality requested but missing.")
            return audio

        if visual is None or audio is None:
            raise ValueError("Both visual and audio features are required for 'av' mode.")

        t_v, t_a = visual.shape[0], audio.shape[0]
        if t_v == t_a:
            return np.concatenate([visual, audio], axis=1)

        t = min(t_v, t_a)
        return np.concatenate([visual[:t], audio[:t]], axis=1)

    def _augment(self, feature: np.ndarray, label: int) -> None:
        t_length = feature.shape[0]
        if t_length <= self.min_aug_frames:
            return
        # Simple random crop augmentation
        for _ in range(5):
            crop_len = np.random.randint(self.min_aug_frames, t_length + 1)
            if crop_len < self.min_aug_frames:
                continue
            start = np.random.randint(0, t_length - crop_len + 1)
            crop = feature[start : start + crop_len]
            self.features.append(crop.astype(np.float32))
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        feature = self.features[index]
        label = self.labels[index]
        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feature, label


def eks_collate_fn(batch):
    """
    Collate function that pads variable-length sequences.
    """
    features, labels = zip(*batch)
    tensors = [torch.from_numpy(f) for f in features]
    padded = pad_sequence(tensors, batch_first=True)
    mask = (padded.sum(dim=-1) != 0).long()
    labels_tensor = torch.tensor(labels)
    return padded, labels_tensor, mask


def get_eks_dataloader(
    root: Union[str, Path],
    fold: str = "train",
    batch_size: int = 8,
    gender: str = "both",
    condition: str = "all",
    phase: str = "all",
    transform=None,
    target_transform=None,
    aug: bool = False,
    modalities: str = "av",
    shuffle: Optional[bool] = None,
):
    dataset = EKSpression(
        root=root,
        fold=fold,
        gender=gender,
        condition=condition,
        phase=phase,
        transform=transform,
        target_transform=target_transform,
        aug=aug,
        modalities=modalities,
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(fold == "train") if shuffle is None else shuffle,
        collate_fn=eks_collate_fn,
    )
    return dataloader


__all__ = ["EKSpression", "get_eks_dataloader", "eks_collate_fn"]

