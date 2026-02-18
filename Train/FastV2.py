"""
Multi-Task Chest X-Ray — VERSION RAPIDE (< 1h sur GPU L4)
==========================================================
Même architecture de modèle, mais optimisé pour itération rapide :
  - Phase 1 : cache les features RadDino (encoder gelé) → entraîne sur features
  - Phase 2 : fine-tuning court avec dégel partiel
  - Heatmaps vectorisées sur GPU
  - Réduction d'epochs (10 + 3 = 13 total)
"""

from rad_dino import RadDino
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import math
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.amp import autocast, GradScaler

import time
from datetime import datetime
import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import json
import gc
import pickle


LIDC_DIR = "../nodulocc_dataset/lidc_png_16_bit"
NIH_DIR  = "../nodulocc_dataset/nih_filtered_images"

LOCALIZATION_CSV = "../nodulocc_dataset/localization_labels.csv"
CLASSIFICATION_CSV = "../nodulocc_dataset/classification_labels.csv"


# ============================================================================
# UTILITAIRES DE CHARGEMENT D'IMAGES (inchangés)
# ============================================================================

def _normalize_csv_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    return os.path.basename(p)

def _load_png(path: str) -> torch.Tensor:
    img = Image.open(path)
    if img.mode not in {"RGB", "L", "I;16", "I"}:
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        if arr.max() > 1.0:
            arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).permute(2, 0, 1)

def load_local_image(dataset: str, filename: str) -> torch.Tensor:
    base = LIDC_DIR if dataset.lower() == "lidc" else NIH_DIR
    img_path = os.path.join(base, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image introuvable: {img_path}")
    return _load_png(img_path).unsqueeze(0)

def _infer_filename_column(df: pd.DataFrame) -> str:
    candidates = [
        "file_name", "filename", "image", "image_name", "image_id",
        "path", "img_path", "image_path"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Colonne filename introuvable. Colonnes: {list(df.columns)}")

def _extract_coords_from_row(row: pd.Series, img_w: int = 224, img_h: int = 224) -> List[Tuple[float, float]]:
    cols = {k.lower(): k for k in row.index}
    for xk, yk in [("x", "y"), ("x_center", "y_center"), ("xc", "yc")]:
        if xk in cols and yk in cols:
            x, y = float(row[cols[xk]]), float(row[cols[yk]])
            if x > 1.5 or y > 1.5:
                x, y = x / max(img_w, 1), y / max(img_h, 1)
            return [(max(0., min(1., x)), max(0., min(1., y)))]
    bbox_keys = [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom")
    ]
    for a, b, c, d in bbox_keys:
        if a in cols and b in cols and c in cols and d in cols:
            xmin, ymin = float(row[cols[a]]), float(row[cols[b]])
            xmax, ymax = float(row[cols[c]]), float(row[cols[d]])
            if max(xmax, ymax) > 1.5:
                xmin, xmax = xmin / max(img_w, 1), xmax / max(img_w, 1)
                ymin, ymax = ymin / max(img_h, 1), ymax / max(img_h, 1)
            x = max(0., min(1., (xmin + xmax) / 2.))
            y = max(0., min(1., (ymin + ymax) / 2.))
            return [(x, y)]
    return []

def _resize_tensor_image(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Attendu (C,H,W), reçu: {tuple(x.shape)}")
    return F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


# ============================================================================
# DATASET UNIFIÉ LIDC + NIH (inchangé)
# ============================================================================

class LungMultiSourceDataset(Dataset):

    def __init__(
        self,
        lidc_dir: str = LIDC_DIR,
        nih_dir: str = NIH_DIR,
        localization_csv: str = LOCALIZATION_CSV,
        classification_csv: str = CLASSIFICATION_CSV,
        image_size: int = 224,
        augment: bool = False,
    ):
        super().__init__()
        self.lidc_dir = lidc_dir
        self.nih_dir = nih_dir
        self.image_size = image_size
        self.augment = augment

        df_loc = pd.read_csv(localization_csv)
        df_cls = pd.read_csv(classification_csv)

        fname_col_loc = _infer_filename_column(df_loc)
        fname_col_cls = _infer_filename_column(df_cls)
        df_loc["_filename"] = df_loc[fname_col_loc].apply(_normalize_csv_path)
        df_cls["_filename"] = df_cls[fname_col_cls].apply(_normalize_csv_path)

        label_candidates = ["Finding Labels", "label", "class", "target"]
        label_col = None
        for c in label_candidates:
            if c in df_cls.columns:
                label_col = c
                break
        if label_col is None:
            raise ValueError(f"Colonne de label introuvable. Colonnes: {list(df_cls.columns)}")

        self.samples = []
        for _, row in df_loc.iterrows():
            fname = row["_filename"]
            path = os.path.join(self.lidc_dir, fname)
            if os.path.exists(path):
                self.samples.append({"path": path, "filename": fname, "dataset": "lidc", "row": row})

        for _, row in df_cls.iterrows():
            fname = row["_filename"]
            path = os.path.join(self.nih_dir, fname)
            if os.path.exists(path):
                self.samples.append({"path": path, "filename": fname, "dataset": "nih", "row": row, "label_col": label_col})

        print(f"Dataset chargé: {len(self.samples)} images (LIDC + NIH)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        x = _load_png(sample["path"])
        h0, w0 = x.shape[1], x.shape[2]
        x = _resize_tensor_image(x, self.image_size)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() > 0.5:
                x = torch.clamp(x * (0.8 + torch.rand(1).item() * 0.4), 0, 1)

        if sample["dataset"] == "nih":
            y_raw = sample["row"][sample["label_col"]]
            if isinstance(y_raw, str):
                y = 0.0 if y_raw.strip().lower() in {"0", "false", "no", "no finding", "nofinding"} else 1.0
            else:
                y = 1.0 if float(y_raw) >= 0.5 else 0.0
            coords = []
        else:
            coords = _extract_coords_from_row(sample["row"], img_w=w0, img_h=h0)
            y = 1.0 if len(coords) > 0 else 0.0

        return {
            "image": x,
            "label": torch.tensor([y], dtype=torch.float32),
            "coords": coords,
            "dataset": sample["dataset"],
            "filename": sample["filename"],
            "path": sample["path"],
        }


def multitask_collate_fn(batch: List[Dict]):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    coords_batch = [b["coords"] for b in batch]
    meta = {
        "dataset": [b["dataset"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "path": [b["path"] for b in batch],
    }
    return images, labels, coords_batch, meta


def create_optimized_dataloaders(
    batch_size=64, num_workers=8, image_size=224,
    val_split=0.15, test_split=0.15, augment_train=True,
    data_fraction=0.2,
):
    full_ds = LungMultiSourceDataset(image_size=image_size, augment=False)

    # Sous-échantillonnage si data_fraction < 1.0
    n_total = len(full_ds)
    if data_fraction < 1.0:
        n_keep = max(10, int(n_total * data_fraction))
        g_sub = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_total, generator=g_sub)[:n_keep].tolist()
        full_ds = torch.utils.data.Subset(full_ds, indices)
        n_total = n_keep
        print(f"⚡ Sous-échantillonnage: {n_keep}/{len(full_ds.dataset)} images ({data_fraction*100:.0f}%)")

    n_test = max(1, int(n_total * test_split))
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val - n_test

    print(f"Split: train={n_train} | val={n_val} | test={n_test}")
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)

    if augment_train:
        # Gérer le cas Subset (data_fraction < 1.0)
        base_ds = full_ds.dataset if isinstance(full_ds, torch.utils.data.Subset) else full_ds
        base_ds.augment = True

    kw = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ============================================================================
# HEATMAP VECTORISÉE SUR GPU (remplace la boucle Python)
# ============================================================================

class GaussianHeatmapGenerator:
    """Génère des heatmaps gaussiennes — version vectorisée."""

    def __init__(self, image_size: int = 224, sigma: float = 10.0):
        self.image_size = image_size
        self.sigma = sigma
        self._grid_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_grid(self, device: torch.device):
        if device not in self._grid_cache:
            H = W = self.image_size
            y = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1)
            x = torch.arange(W, device=device, dtype=torch.float32).view(1, -1)
            self._grid_cache[device] = (y, x)
        return self._grid_cache[device]

    def generate(self, nodule_coords, device):
        H = W = self.image_size
        if len(nodule_coords) == 0:
            return torch.zeros((1, H, W), device=device)
        y_grid, x_grid = self._get_grid(device)
        heatmap = torch.zeros((H, W), device=device)
        for x_n, y_n in nodule_coords:
            g = torch.exp(-((x_grid - x_n * W) ** 2 + (y_grid - y_n * H) ** 2) / (2 * self.sigma ** 2))
            heatmap = torch.maximum(heatmap, g)
        return heatmap.unsqueeze(0)

    def batch_generate(self, batch_coords, device):
        return torch.stack([self.generate(c, device) for c in batch_coords])


# ============================================================================
# ARCHITECTURE DU MODÈLE (identique)
# ============================================================================

class SharedProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            return self.proj(x.reshape(B * N, C)).reshape(B, N, self.output_dim)
        return self.proj(x)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class LocalizationHead(nn.Module):
    def __init__(self, input_channels=512, input_spatial_size=37,
                 decoder_channels=[512, 256, 128, 64], output_size=224, dropout=0.3):
        super().__init__()
        self.input_channels = input_channels
        self.input_spatial_size = input_spatial_size
        self.output_size = output_size

        decoder_layers = []
        prev_ch = input_channels
        for i, out_ch in enumerate(decoder_channels):
            decoder_layers.extend([
                nn.Conv2d(prev_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ])
            if i < len(decoder_channels) - 1:
                decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            prev_ch = out_ch
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.final_upsample = nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=False)

    def forward(self, x):
        if x.shape[2] != self.input_spatial_size or x.shape[3] != self.input_spatial_size:
            x = F.interpolate(x, size=(self.input_spatial_size, self.input_spatial_size),
                              mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return self.final_upsample(self.final_conv(x))


class ChestXRayMultiTask(nn.Module):
    """Architecture multi-task complète (identique à l'original)."""

    def __init__(
        self, encoder, embed_dim=768, num_patches=1369, output_size=224,
        shared_hidden=512, classification_hidden=[256, 128],
        localization_channels=[512, 256, 128, 64], dropout=0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_frozen = True
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_grid_size = int(math.sqrt(num_patches))
        self.output_size = output_size

        self.shared_cls_proj = SharedProjector(embed_dim, shared_hidden, dropout)
        self.shared_patch_proj = SharedProjector(embed_dim, shared_hidden, dropout)
        self.classification_head = ClassificationHead(shared_hidden, classification_hidden, dropout)
        self.localization_head = LocalizationHead(
            shared_hidden, self.patch_grid_size, localization_channels, output_size, dropout
        )

    def forward(self, x):
        if self.encoder_frozen:
            with torch.no_grad():
                cls_emb, spatial = self.encoder(x)
            cls_emb = cls_emb.clone().detach().requires_grad_(True)
            spatial = spatial.clone().detach().requires_grad_(True)
        else:
            cls_emb, spatial = self.encoder(x)

        if spatial.dim() == 4:
            B, C, H, W = spatial.shape
            patch_emb = spatial.flatten(2).transpose(1, 2)
        else:
            patch_emb = spatial

        cls_shared = self.shared_cls_proj(cls_emb)
        patch_shared = self.shared_patch_proj(patch_emb)
        cls_logits = self.classification_head(cls_shared)
        patch_2d = patch_shared.transpose(1, 2).reshape(
            patch_shared.size(0), -1, H, W
        )
        heatmap = self.localization_head(patch_2d)
        return cls_logits, heatmap, cls_emb

    def forward_from_features(self, cls_emb, patch_emb):
        """Forward SANS encoder — utilise les features pré-cachées."""
        cls_shared = self.shared_cls_proj(cls_emb)
        patch_shared = self.shared_patch_proj(patch_emb)
        cls_logits = self.classification_head(cls_shared)
        patch_2d = patch_shared.transpose(1, 2).reshape(
            patch_shared.size(0), -1, self.patch_grid_size, self.patch_grid_size
        )
        heatmap = self.localization_head(patch_2d)
        return cls_logits, heatmap

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True

    def unfreeze_encoder(self, layers_to_unfreeze=None):
        if layers_to_unfreeze is None:
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            for name, p in self.encoder.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    p.requires_grad = True
        self.encoder_frozen = False


# ============================================================================
# LOSS (identique)
# ============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_loc=2.0, lambda_smooth=0.1,
                 use_focal_loss=True, focal_alpha=0.9, focal_gamma=2.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.lambda_smooth = lambda_smooth
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, cls_logits, pred_heatmaps, cls_targets, target_heatmaps):
        # Classification
        if self.use_focal_loss:
            bce = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction='none')
            pt = torch.where(cls_targets == 1, torch.sigmoid(cls_logits), 1 - torch.sigmoid(cls_logits))
            alpha = torch.where(cls_targets == 1, self.focal_alpha, 1 - self.focal_alpha)
            cls_loss = (alpha * (1 - pt) ** self.focal_gamma * bce).mean()
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)

        # Localisation (positifs seulement)
        pos_mask = (cls_targets > 0.5).squeeze(1)
        if pos_mask.sum() > 0:
            pred_pos, tgt_pos = pred_heatmaps[pos_mask], target_heatmaps[pos_mask]
            loc_loss = F.binary_cross_entropy_with_logits(pred_pos, tgt_pos, reduction='mean')
            smooth_loss = (
                torch.abs(pred_pos[:, :, 1:, :] - pred_pos[:, :, :-1, :]).mean() +
                torch.abs(pred_pos[:, :, :, 1:] - pred_pos[:, :, :, :-1]).mean()
            )
        else:
            loc_loss = torch.tensor(0.0, device=cls_logits.device)
            smooth_loss = torch.tensor(0.0, device=cls_logits.device)

        total = self.lambda_cls * cls_loss + self.lambda_loc * loc_loss + self.lambda_smooth * smooth_loss
        return total, {
            "total": total.item(), "classification": cls_loss.item(),
            "localization": loc_loss.item(), "smoothness": smooth_loss.item(),
        }


# ============================================================================
# EXTRACTION ET CACHE DES FEATURES — MEMORY-MAPPED SUR DISQUE (float16)
# ============================================================================
#
# Problème : 22K images × 1369 patches × 768 dims × 4 bytes = ~90 GB RAM → OOM
# Solution : écrire directement sur disque en float16 via numpy memmap
#            → ~45 GB sur disque, ~0 GB en RAM (lecture à la demande)
#

#

def extract_and_cache_features(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cache_dir: str,
    split_name: str = "train",
    embed_dim: int = 768,
    num_patches: int = 1369,
):
    """
    Extrait les features et les écrit directement sur disque en float16 (numpy memmap).
    Ne garde RIEN en RAM.
    
    Retourne le chemin du cache_dir pour construire le CachedFeatureDataset.
    """
    cls_path = os.path.join(cache_dir, f"{split_name}_cls.npy")
    patch_path = os.path.join(cache_dir, f"{split_name}_patches.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.npy")
    coords_path = os.path.join(cache_dir, f"{split_name}_coords.pkl")
    meta_path = os.path.join(cache_dir, f"{split_name}_meta.json")

    # Vérifier si cache existe déjà
    if all(os.path.exists(p) for p in [cls_path, patch_path, labels_path, coords_path, meta_path]):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        n = meta["num_samples"]
        size_gb = (os.path.getsize(cls_path) + os.path.getsize(patch_path)) / 1e9
        print(f"  ⚡ Cache trouvé: {n} samples, {size_gb:.1f} GB sur disque")
        return cache_dir, split_name, n

    # Compter le nombre total d'échantillons
    n_total = 0
    for images, labels, coords_batch, meta in dataloader:
        n_total += images.size(0)
    print(f"    {n_total} échantillons à extraire")

    # Pré-allouer les fichiers memmap
    cls_mmap = np.memmap(cls_path, dtype=np.float16, mode='w+', shape=(n_total, embed_dim))
    patch_mmap = np.memmap(patch_path, dtype=np.float16, mode='w+', shape=(n_total, num_patches, embed_dim))
    labels_mmap = np.memmap(labels_path, dtype=np.float32, mode='w+', shape=(n_total, 1))
    all_coords = []

    encoder.eval()
    offset = 0
    start = time.time()
    total_batches = len(dataloader)

    with torch.no_grad():
        for i, (images, labels, coords_batch, _meta) in enumerate(dataloader):
            B = images.size(0)
            images = images.to(device)

            with torch.amp.autocast('cuda'):
                cls_emb, spatial = encoder(images)

            if spatial.dim() == 4:
                patch_emb = spatial.flatten(2).transpose(1, 2)
            else:
                patch_emb = spatial

            # Écrire directement sur disque en float16
            cls_mmap[offset:offset+B] = cls_emb.cpu().half().numpy()
            patch_mmap[offset:offset+B] = patch_emb.cpu().half().numpy()
            labels_mmap[offset:offset+B] = labels.numpy()
            all_coords.extend(coords_batch)

            offset += B

            if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                elapsed = time.time() - start
                speed = offset / elapsed
                print(f"    [{i+1}/{total_batches}] {speed:.0f} img/s | {offset}/{n_total}")

    # Flush
    cls_mmap.flush()
    patch_mmap.flush()
    labels_mmap.flush()
    del cls_mmap, patch_mmap, labels_mmap

    # Sauvegarder coords (petit) et metadata
    with open(coords_path, 'wb') as f:
        pickle.dump(all_coords, f)
    with open(meta_path, 'w') as f:
        json.dump({
            "num_samples": n_total,
            "embed_dim": embed_dim,
            "num_patches": num_patches,
        }, f)

    elapsed = time.time() - start
    size_gb = (os.path.getsize(cls_path) + os.path.getsize(patch_path)) / 1e9
    print(f"  ✓ Features extraites: {n_total} images en {elapsed:.1f}s ({size_gb:.1f} GB sur disque)")

    return cache_dir, split_name, n_total


class CachedFeatureDataset(Dataset):
    """
    Dataset sur features pré-extraites stockées sur disque (numpy memmap).
    Charge chaque sample à la demande → quasi-zéro RAM.
    Lazy-open memmap pour compatibilité avec DataLoader num_workers > 0.
    """

    def __init__(self, cache_dir: str, split_name: str):
        meta_path = os.path.join(cache_dir, f"{split_name}_meta.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.n = meta["num_samples"]
        self.embed_dim = meta["embed_dim"]
        self.num_patches = meta["num_patches"]
        self.cache_dir = cache_dir
        self.split_name = split_name

        # Charger coords en mémoire (petit, quelques MB max)
        coords_path = os.path.join(cache_dir, f"{split_name}_coords.pkl")
        with open(coords_path, 'rb') as f:
            self.coords = pickle.load(f)

        # Les memmap seront ouverts lazily (compatibilité multiprocessing)
        self._cls_mmap = None
        self._patch_mmap = None
        self._labels_mmap = None

    def _open_mmaps(self):
        """Ouvre les memmap — appelé une seule fois par worker."""
        if self._cls_mmap is None:
            self._cls_mmap = np.memmap(
                os.path.join(self.cache_dir, f"{self.split_name}_cls.npy"),
                dtype=np.float16, mode='r', shape=(self.n, self.embed_dim)
            )
            self._patch_mmap = np.memmap(
                os.path.join(self.cache_dir, f"{self.split_name}_patches.npy"),
                dtype=np.float16, mode='r', shape=(self.n, self.num_patches, self.embed_dim)
            )
            self._labels_mmap = np.memmap(
                os.path.join(self.cache_dir, f"{self.split_name}_labels.npy"),
                dtype=np.float32, mode='r', shape=(self.n, 1)
            )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._open_mmaps()
        cls_e = torch.from_numpy(self._cls_mmap[idx].copy().astype(np.float32))
        patch_e = torch.from_numpy(self._patch_mmap[idx].copy().astype(np.float32))
        label = torch.from_numpy(self._labels_mmap[idx].copy())
        coord = self.coords[idx]
        return cls_e, patch_e, label, coord


def cached_collate_fn(batch):
    cls_emb = torch.stack([b[0] for b in batch])
    patch_emb = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    coords = [b[3] for b in batch]
    return cls_emb, patch_emb, labels, coords


# ============================================================================
# TRAINER RAPIDE
# ============================================================================

class FastTrainer:
    """Trainer optimisé : Phase 1 sur features cachées, Phase 2 end-to-end."""

    def __init__(self, model, criterion, heatmap_generator, device,
                 output_dir="checkpoints_multitask_fast", grad_clip_norm=1.0):
        self.model = model
        self.criterion = criterion
        self.heatmap_gen = heatmap_generator
        self.device = device
        self.output_dir = output_dir
        self.grad_clip_norm = grad_clip_norm
        os.makedirs(output_dir, exist_ok=True)

        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    # ------ PHASE 1 : Entraînement sur features cachées ------

    def train_phase1(
        self,
        train_loader: DataLoader,  # CachedFeatureDataset loader
        val_loader: DataLoader,
        optimizer,
        scheduler=None,
        num_epochs: int = 10,
        early_stopping_patience: int = 7,
    ):
        """Phase 1 : heads seulement, pas d'encoder dans la boucle."""
        print(f"\n{'='*70}")
        print(f"PHASE 1 — ENTRAÎNEMENT SUR FEATURES CACHÉES ({num_epochs} epochs)")
        print(f"{'='*70}\n")

        scaler = GradScaler()
        patience_ctr = 0

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # Train
            self.model.train()
            # Garder encoder en eval (pas utilisé mais au cas où)
            self.model.encoder.eval()

            train_loss, train_cls, train_loc, n_batches = 0, 0, 0, 0

            for cls_emb, patch_emb, labels, coords in train_loader:
                cls_emb = cls_emb.to(self.device)
                patch_emb = patch_emb.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast('cuda'):
                    cls_logits, heatmap = self.model.forward_from_features(cls_emb, patch_emb)
                    target_hm = self.heatmap_gen.batch_generate(coords, self.device)
                    loss, ld = self.criterion(cls_logits, heatmap, labels, target_hm)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                train_loss += ld["total"]
                train_cls += ld["classification"]
                train_loc += ld["localization"]
                n_batches += 1

            # Validation
            val_metrics = self._validate_cached(val_loader)

            avg_tl = train_loss / n_batches
            elapsed = time.time() - t0

            # Historique
            self.history["train_loss"].append(avg_tl)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_cls"].append(train_cls / n_batches)
            self.history["val_cls"].append(val_metrics["val_cls_loss"])
            self.history["train_loc"].append(train_loc / n_batches)
            self.history["val_loc"].append(val_metrics["val_loc_loss"])

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["val_loss"])
                else:
                    scheduler.step()

            # Best model
            improved = val_metrics["val_loss"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                patience_ctr = 0
                self._save("best_model.pt", epoch)
            else:
                patience_ctr += 1

            marker = " ★" if improved else ""
            print(f"  Epoch {epoch:2d}/{num_epochs} | "
                  f"Train {avg_tl:.4f} | Val {val_metrics['val_loss']:.4f} | "
                  f"{elapsed:.0f}s{marker}")

            if patience_ctr >= early_stopping_patience:
                print(f"  ⚠ Early stopping (patience={early_stopping_patience})")
                break

        print(f"\n  Best: epoch {self.best_epoch}, val_loss={self.best_val_loss:.4f}")

    @torch.no_grad()
    def _validate_cached(self, val_loader):
        self.model.eval()
        total, cls_l, loc_l, n = 0, 0, 0, 0
        for cls_emb, patch_emb, labels, coords in val_loader:
            cls_emb = cls_emb.to(self.device)
            patch_emb = patch_emb.to(self.device)
            labels = labels.to(self.device)
            cls_logits, heatmap = self.model.forward_from_features(cls_emb, patch_emb)
            target_hm = self.heatmap_gen.batch_generate(coords, self.device)
            _, ld = self.criterion(cls_logits, heatmap, labels, target_hm)
            total += ld["total"]; cls_l += ld["classification"]; loc_l += ld["localization"]
            n += 1
        return {"val_loss": total/n, "val_cls_loss": cls_l/n, "val_loc_loss": loc_l/n}

    # ------ PHASE 2 : Fine-tuning end-to-end ------

    def train_phase2(
        self,
        train_loader: DataLoader,  # DataLoader original (images)
        val_loader: DataLoader,
        optimizer,
        scheduler=None,
        num_epochs: int = 3,
        early_stopping_patience: int = 5,
        log_interval: int = 50,
    ):
        """Phase 2 : dégel partiel encoder, entraînement end-to-end."""
        print(f"\n{'='*70}")
        print(f"PHASE 2 — FINE-TUNING END-TO-END ({num_epochs} epochs)")
        print(f"{'='*70}\n")

        scaler = GradScaler()
        patience_ctr = 0
        phase2_best = self.best_val_loss

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            self.model.train()

            train_loss, n_batches = 0, 0

            for i, (images, labels, coords, meta) in enumerate(train_loader, 1):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast('cuda'):
                    cls_logits, heatmap, _ = self.model(images)
                    target_hm = self.heatmap_gen.batch_generate(coords, self.device)
                    loss, ld = self.criterion(cls_logits, heatmap, labels, target_hm)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                train_loss += ld["total"]
                n_batches += 1

                if i % log_interval == 0:
                    elapsed = time.time() - t0
                    print(f"    [{i}/{len(train_loader)}] loss={ld['total']:.4f} "
                          f"({i * images.size(0) / elapsed:.0f} img/s)")

            # Validation (end-to-end)
            val_m = self._validate_e2e(val_loader)
            avg_tl = train_loss / n_batches
            elapsed = time.time() - t0

            self.history["train_loss"].append(avg_tl)
            self.history["val_loss"].append(val_m["val_loss"])

            if scheduler:
                scheduler.step()

            improved = val_m["val_loss"] < phase2_best
            if improved:
                phase2_best = val_m["val_loss"]
                self.best_val_loss = phase2_best
                self.best_epoch = epoch + 100  # distinguer Phase 2
                patience_ctr = 0
                self._save("best_model.pt", epoch)
            else:
                patience_ctr += 1

            marker = " ★" if improved else ""
            print(f"  Epoch {epoch}/{num_epochs} | "
                  f"Train {avg_tl:.4f} | Val {val_m['val_loss']:.4f} | "
                  f"{elapsed:.0f}s{marker}")

            if patience_ctr >= early_stopping_patience:
                print(f"  ⚠ Early stopping")
                break

    @torch.no_grad()
    def _validate_e2e(self, val_loader):
        self.model.eval()
        total, cls_l, loc_l, n = 0, 0, 0, 0
        for images, labels, coords, meta in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            cls_logits, heatmap, _ = self.model(images)
            target_hm = self.heatmap_gen.batch_generate(coords, self.device)
            _, ld = self.criterion(cls_logits, heatmap, labels, target_hm)
            total += ld["total"]; cls_l += ld["classification"]; loc_l += ld["localization"]
            n += 1
        return {"val_loss": total/n, "val_cls_loss": cls_l/n, "val_loc_loss": loc_l/n}

    def _save(self, filename, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, os.path.join(self.output_dir, filename))

    def save_history(self):
        path = os.path.join(self.output_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"✓ Historique: {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHEST X-RAY MULTI-TASK — VERSION RAPIDE (< 1h)")
    print("="*70 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # ---- Configuration ----
    config = {
        "batch_size": 8,           # Plus gros batch (L4 a 24 GB)
        "num_workers": 2,
        "image_size": 224,
        "phase1_epochs": 10,         # Suffisant pour converger sur features
        "phase2_epochs": 3,          # Court fine-tuning
        "lr_head": 1e-3,
        "lr_backbone": 1e-5,
        "lambda_cls": 1.0,
        "lambda_loc": 2.0,
        "lambda_smooth": 0.1,
        "early_stopping": 7,
        "output_dir": "checkpoints_multitask_fast",
        "cache_dir": "feature_cache",
        "data_fraction": 0.1,       # ← 0.5 = 33K images, 0.25 = 16K, etc.
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # ---- Dataloaders (images) ----
    print("Création des dataloaders...")
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
        data_fraction=config["data_fraction"],
    )
    print()

    # ---- Modèle ----
    print("Création du modèle...")
    try:
        encoder = RadDino()
        embed_dim = 768
        num_patches = 37 * 37
        print("✓ RadDino chargé")
    except Exception as e:
        print(f"⚠ RadDino error: {e}, utilisation mock")
        class MockRadDino(nn.Module):
            def forward(self, x):
                B = x.size(0)
                return torch.randn(B, 768, device=x.device), torch.randn(B, 768, 37, 37, device=x.device)
        encoder = MockRadDino()
        embed_dim, num_patches = 768, 37 * 37

    model = ChestXRayMultiTask(
        encoder=encoder, embed_dim=embed_dim, num_patches=num_patches,
        output_size=config["image_size"], shared_hidden=512,
        classification_hidden=[256, 128],
        localization_channels=[512, 256, 128, 64], dropout=0.3,
    ).to(device)
    model.freeze_encoder()

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_total:,} total, {n_train:,} entraînables\n")

    # ---- Loss & Heatmap ----
    criterion = MultiTaskLoss(
        lambda_cls=config["lambda_cls"], lambda_loc=config["lambda_loc"],
        lambda_smooth=config["lambda_smooth"],
    )
    heatmap_gen = GaussianHeatmapGenerator(image_size=config["image_size"], sigma=10.0)

    # ================================================================
    # PHASE 1 : Extraction features + entraînement heads
    # ================================================================
    print("="*70)
    print("EXTRACTION DES FEATURES (une seule fois)")
    print("="*70 + "\n")

    os.makedirs(config["cache_dir"], exist_ok=True)
    frac_tag = f"_frac{config['data_fraction']:.2f}" if config["data_fraction"] < 1.0 else ""
    cache_subdir = os.path.join(config["cache_dir"], f"features{frac_tag}")
    os.makedirs(cache_subdir, exist_ok=True)

    print("  Train set:")
    extract_and_cache_features(
        encoder=model.encoder, dataloader=train_loader, device=device,
        cache_dir=cache_subdir, split_name="train",
    )
    print("  Val set:")
    extract_and_cache_features(
        encoder=model.encoder, dataloader=val_loader, device=device,
        cache_dir=cache_subdir, split_name="val",
    )

    # Libérer VRAM de l'extraction
    torch.cuda.empty_cache()
    gc.collect()

    # Dataloaders sur features (memmap = quasi-zéro RAM)
    train_feat_ds = CachedFeatureDataset(cache_subdir, "train")
    val_feat_ds = CachedFeatureDataset(cache_subdir, "val")

    feat_batch_size = config["batch_size"] * 4  # 384
    train_feat_loader = DataLoader(
        train_feat_ds, batch_size=feat_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=cached_collate_fn,
    )
    val_feat_loader = DataLoader(
        val_feat_ds, batch_size=feat_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=cached_collate_fn,
    )

    # Optimizer Phase 1
    optimizer_p1 = torch.optim.AdamW([
        {'params': model.shared_cls_proj.parameters(), 'lr': config["lr_head"]},
        {'params': model.shared_patch_proj.parameters(), 'lr': config["lr_head"]},
        {'params': model.classification_head.parameters(), 'lr': config["lr_head"]},
        {'params': model.localization_head.parameters(), 'lr': config["lr_head"]},
    ], weight_decay=0.01)

    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=config["phase1_epochs"], eta_min=1e-6
    )

    trainer = FastTrainer(model, criterion, heatmap_gen, device, config["output_dir"])

    trainer.train_phase1(
        train_loader=train_feat_loader,
        val_loader=val_feat_loader,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        num_epochs=config["phase1_epochs"],
        early_stopping_patience=config["early_stopping"],
    )

    # Libérer les dataloaders features
    del train_feat_ds, val_feat_ds, train_feat_loader, val_feat_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ================================================================
    # PHASE 2 : Fine-tuning end-to-end (court)
    # ================================================================

    model.unfreeze_encoder(layers_to_unfreeze=["blocks.10", "blocks.11"])

    optimizer_p2 = torch.optim.AdamW([
        {'params': model.shared_cls_proj.parameters(), 'lr': config["lr_head"] * 0.1},
        {'params': model.shared_patch_proj.parameters(), 'lr': config["lr_head"] * 0.1},
        {'params': model.classification_head.parameters(), 'lr': config["lr_head"] * 0.1},
        {'params': model.localization_head.parameters(), 'lr': config["lr_head"] * 0.1},
        {'params': [p for p in model.encoder.parameters() if p.requires_grad],
         'lr': config["lr_backbone"]},
    ], weight_decay=0.01)

    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=config["phase2_epochs"], eta_min=1e-7
    )

    trainer.train_phase2(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        num_epochs=config["phase2_epochs"],
        early_stopping_patience=5,
    )

    # ================================================================
    # TEST FINAL
    # ================================================================
    print(f"\n{'='*70}")
    print("ÉVALUATION FINALE")
    print(f"{'='*70}\n")

    test_metrics = trainer._validate_e2e(test_loader)
    print(f"  Test Loss:     {test_metrics['val_loss']:.4f}")
    print(f"  Test Cls Loss: {test_metrics['val_cls_loss']:.4f}")
    print(f"  Test Loc Loss: {test_metrics['val_loc_loss']:.4f}")

    # Sauvegarde
    trainer.save_history()
    results = {
        "config": config,
        "test_metrics": test_metrics,
        "best_val_loss": trainer.best_val_loss,
        "best_epoch": trainer.best_epoch,
    }
    with open(os.path.join(config["output_dir"], "final_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Terminé! Best val_loss={trainer.best_val_loss:.4f}")
    print(f"  Checkpoints: {config['output_dir']}/")
