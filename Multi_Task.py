from rad_dino import RadDino
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler

import time
from datetime import datetime
import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import json


LIDC_DIR = "../nodulocc_dataset/lidc_png_16_bit"
NIH_DIR  = "../nodulocc_dataset/nih_filtered_images"

LOCALIZATION_CSV = "../nodulocc_dataset/localization_labels.csv"
CLASSIFICATION_CSV = "../nodulocc_dataset/classification_labels.csv"

# ============================================================================
# UTILITAIRES DE CHARGEMENT D'IMAGES
# ============================================================================

def _normalize_csv_path(p: str) -> str:
    """Normalise une valeur de CSV qui peut être un chemin ou juste un nom de fichier."""
    p = str(p)
    p = p.strip().strip('"').strip("'")
    return os.path.basename(p)

def _load_png(path: str) -> torch.Tensor:
    """Charge un PNG et normalise correctement."""
    img = Image.open(path)
    
    # Conversion en RGB si nécessaire
    if img.mode in {"P", "RGBA", "LA", "CMYK", "YCbCr", "RGBa"}:
        img = img.convert("RGB")
    elif img.mode not in {"RGB", "L", "I;16", "I"}:
        img = img.convert("RGB")
    
    arr = np.array(img)
    
    # Normalisation appropriée
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        # Déjà en float, vérifier la plage
        if arr.max() > 1.0:
            arr = arr / 255.0
    
    # Assurer 3 canaux
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    
    # Clipping pour sécurité
    arr = np.clip(arr, 0.0, 1.0)
    
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def load_local_image(dataset: str, filename: str) -> torch.Tensor:
    """Retourne une image en tenseur (1,3,H,W) depuis LIDC ou NIH."""
    base = LIDC_DIR if dataset.lower() == "lidc" else NIH_DIR
    img_path = os.path.join(base, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image introuvable: {img_path}")
    x = _load_png(img_path).unsqueeze(0)
    return x

def _infer_filename_column(df: pd.DataFrame) -> str:
    """Trouve une colonne de nom de fichier dans un CSV."""
    candidates = [
        "file_name", "filename", "image", "image_name", "image_id",
        "path", "img_path", "image_path"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Impossible d'inférer la colonne contenant le nom de fichier. "
        f"Colonnes disponibles: {list(df.columns)}. "
        "Attendu une des colonnes: " + ", ".join(candidates)
    )

def _extract_coords_from_row(row: pd.Series, img_w: int = 224, img_h: int = 224) -> List[Tuple[float, float]]:
    """Extrait les coordonnées (x, y) normalisées [0,1] d'une ligne CSV."""
    cols = {k.lower(): k for k in row.index}
    
    # 1) centre directement (normalisé ou pixels)
    for xk, yk in [("x", "y"), ("x_center", "y_center"), ("xc", "yc")]:
        if xk in cols and yk in cols:
            x = float(row[cols[xk]])
            y = float(row[cols[yk]])
            if x > 1.5 or y > 1.5:
                x = x / max(img_w, 1)
                y = y / max(img_h, 1)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return [(x, y)]

    # 2) bbox
    bbox_keys = [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom")
    ]
    for a, b, c, d in bbox_keys:
        if a in cols and b in cols and c in cols and d in cols:
            xmin = float(row[cols[a]])
            ymin = float(row[cols[b]])
            xmax = float(row[cols[c]])
            ymax = float(row[cols[d]])

            if max(xmax, ymax) > 1.5:
                xmin = xmin / max(img_w, 1)
                xmax = xmax / max(img_w, 1)
                ymin = ymin / max(img_h, 1)
                ymax = ymax / max(img_h, 1)

            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return [(x, y)]

    return []

def _resize_tensor_image(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize bilinear d'un tenseur image (C,H,W) vers (C,size,size)."""
    if x.dim() != 3:
        raise ValueError(f"Attendu un tenseur (C,H,W), reçu: {tuple(x.shape)}")
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


# ============================================================================
# DATASET UNIFIÉ LIDC + NIH
# ============================================================================

class LungMultiSourceDataset(Dataset):
    """Dataset unifié LIDC (localization) + NIH (classification)."""

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

        # Chargement CSV
        df_loc = pd.read_csv(localization_csv)
        df_cls = pd.read_csv(classification_csv)

        # Normalisation noms de fichiers
        fname_col_loc = _infer_filename_column(df_loc)
        fname_col_cls = _infer_filename_column(df_cls)

        df_loc["_filename"] = df_loc[fname_col_loc].apply(_normalize_csv_path)
        df_cls["_filename"] = df_cls[fname_col_cls].apply(_normalize_csv_path)

        # Label column pour NIH
        label_candidates = ["Finding Labels", "label", "class", "target"]
        label_col = None
        for c in label_candidates:
            if c in df_cls.columns:
                label_col = c
                break
        if label_col is None:
            raise ValueError(f"Colonne de label introuvable dans {classification_csv}. Colonnes: {list(df_cls.columns)}")

        # Construction de la liste unifiée
        self.samples = []

        # LIDC
        for _, row in df_loc.iterrows():
            fname = row["_filename"]
            path = os.path.join(self.lidc_dir, fname)
            if not os.path.exists(path):
                continue
            self.samples.append({
                "path": path,
                "filename": fname,
                "dataset": "lidc",
                "row": row,
            })

        # NIH
        for _, row in df_cls.iterrows():
            fname = row["_filename"]
            path = os.path.join(self.nih_dir, fname)
            if not os.path.exists(path):
                continue
            self.samples.append({
                "path": path,
                "filename": fname,
                "dataset": "nih",
                "row": row,
                "label_col": label_col,
            })

        print(f"Dataset chargé: {len(self.samples)} images (LIDC + NIH)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        path = sample["path"]
        filename = sample["filename"]
        dataset = sample["dataset"]
        row = sample["row"]

        # Chargement image
        x = _load_png(path)
        h0, w0 = x.shape[1], x.shape[2]
        x = _resize_tensor_image(x, self.image_size)

        # Augmentation (optionnelle)
        if self.augment:
            x = self._apply_augmentation(x)

        # Labels et coords selon dataset
        if dataset == "nih":
            label_col = sample["label_col"]
            y_raw = row[label_col]
            if isinstance(y_raw, str):
                y_str = y_raw.strip().lower()
                y = 0.0 if y_str in {"0", "false", "no", "no finding", "nofinding"} else 1.0
            else:
                y = float(y_raw)
                y = 1.0 if y >= 0.5 else 0.0
            coords = []
        else:
            # LIDC
            coords = _extract_coords_from_row(row, img_w=w0, img_h=h0)
            y = 1.0 if len(coords) > 0 else 0.0

        return {
            "image": x,
            "label": torch.tensor([y], dtype=torch.float32),
            "coords": coords,
            "dataset": dataset,
            "filename": filename,
            "path": path,
        }

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Augmentation simple."""
        # Flip horizontal aléatoire
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
        
        # Ajustement luminosité/contraste
        if torch.rand(1).item() > 0.5:
            factor = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            x = torch.clamp(x * factor, 0, 1)
        
        return x


def multitask_collate_fn(batch: List[Dict]):
    """Collate pour batch avec coords de longueur variable."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    coords_batch = [b["coords"] for b in batch]
    meta = {
        "dataset": [b["dataset"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "path": [b["path"] for b in batch],
    }
    return images, labels, coords_batch, meta


def create_optimized_dataloaders(
    batch_size: int = 64,
    num_workers: int = 8,
    image_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    lidc_dir: str = LIDC_DIR,
    nih_dir: str = NIH_DIR,
    localization_csv: str = LOCALIZATION_CSV,
    classification_csv: str = CLASSIFICATION_CSV,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crée les dataloaders train/val/test optimisés."""
    
    full_ds = LungMultiSourceDataset(
        lidc_dir=lidc_dir,
        nih_dir=nih_dir,
        localization_csv=localization_csv,
        classification_csv=classification_csv,
        image_size=image_size,
        augment=False,
    )

    n_total = len(full_ds)
    n_test = max(1, int(n_total * test_split))
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val - n_test

    print(f"Split dataset: train={n_train} | val={n_val} | test={n_test}")

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)

    # Augmentation seulement sur train
    if augment_train:
        train_ds.dataset.augment = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# GÉNÉRATEUR DE HEATMAPS GAUSSIENNES
# ============================================================================

class GaussianHeatmapGenerator:
    """Génère des heatmaps gaussiennes à partir de coordonnées de nodules."""
    
    def __init__(self, image_size: int = 224, sigma: float = 10.0):
        self.image_size = image_size
        self.sigma = sigma
    
    def generate(
        self, 
        nodule_coords: List[Tuple[float, float]], 
        device: torch.device
    ) -> torch.Tensor:
        """
        Génère une heatmap pour une liste de coordonnées.
        
        Args:
            nodule_coords: liste de (x, y) normalisés [0,1]
            device: dispositif PyTorch
        Returns:
            heatmap: (1, H, W)
        """
        H, W = self.image_size, self.image_size
        heatmap = torch.zeros((H, W), device=device)
        
        if len(nodule_coords) == 0:
            return heatmap.unsqueeze(0)
        
        # Grille de coordonnées
        y_coords = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)
        x_coords = torch.arange(0, W, device=device).view(1, -1).expand(H, W)
        
        for x_norm, y_norm in nodule_coords:
            x_pixel = x_norm * W
            y_pixel = y_norm * H
            
            # Gaussienne 2D
            gaussian = torch.exp(
                -((x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2) / (2 * self.sigma ** 2)
            )
            
            heatmap = torch.maximum(heatmap, gaussian)
        
        return heatmap.unsqueeze(0)
    
    def batch_generate(
        self, 
        batch_nodule_coords: List[List[Tuple[float, float]]], 
        device: torch.device
    ) -> torch.Tensor:
        """
        Génère des heatmaps pour un batch.
        
        Args:
            batch_nodule_coords: Liste de listes de coordonnées
            device: dispositif PyTorch
        Returns:
            heatmaps: (B, 1, H, W)
        """
        heatmaps = []
        for nodule_coords in batch_nodule_coords:
            heatmap = self.generate(nodule_coords, device)
            heatmaps.append(heatmap)
        
        return torch.stack(heatmaps, dim=0)


# ============================================================================
# ARCHITECTURE DU MODÈLE MULTI-TASK
# ============================================================================

class SharedProjector(nn.Module):
    """
    Projecteur partagé pour réduire la dimensionnalité des features.
    Architecture: Linear -> BN -> ReLU -> Dropout -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.3
    ):
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
        
        print(f"    SharedProjector: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) ou (B, N, input_dim)
            
        Returns:
            (B, output_dim) ou (B, N, output_dim)
        """
        # Vérification de la dimension d'entrée
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {x.size(-1)}. "
                f"Input shape: {x.shape}"
            )
        
        # Si 3D (batch, num_patches, dim)
        if x.dim() == 3:
            B, N, C = x.shape
            # Reshape: (B, N, C) -> (B*N, C)
            x_flat = x.reshape(B * N, C)
            
            # Projection
            x_proj = self.proj(x_flat)
            
            # Reshape back: (B*N, output_dim) -> (B, N, output_dim)
            x_out = x_proj.reshape(B, N, self.output_dim)
            
        # Si 2D (batch, dim)
        elif x.dim() == 2:
            x_out = self.proj(x)
        
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D: {x.shape}")
        
        return x_out



class ClassificationHead(nn.Module):
    """
    Head de classification binaire (sain vs pathologique).
    Architecture: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,  # <-- AJOUT du paramètre manquant
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Dimension d'entrée (depuis shared projector)
            hidden_dims: Liste des dimensions cachées (ex: [256, 128])
            dropout: Taux de dropout
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Couches cachées
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
        
        # Couche de sortie (logit unique pour classification binaire)
        layers.append(nn.Linear(dims[-1], 1))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"  ClassificationHead:")
        print(f"    Input: {input_dim}")
        print(f"    Hidden: {hidden_dims}")
        print(f"    Output: 1 (logit binaire)")
        print(f"    Dropout: {dropout}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) - Features depuis shared projector
            
        Returns:
            logits: (B, 1) - Logits de classification (non sigmoidés)
        """
        return self.classifier(x)


class LocalizationHead(nn.Module):
    """
    Head de localisation avec décodeur CNN pour générer des heatmaps.
    Prend des features spatiales et les upsampling vers la résolution cible.
    """
    
    def __init__(
        self,
        input_channels: int = 512,
        input_spatial_size: int = 37,  # Taille spatiale d'entrée (37x37 pour RadDino)
        decoder_channels: List[int] = [512, 256, 128, 64],
        output_size: int = 224,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_spatial_size = input_spatial_size
        self.output_size = output_size
        
        print(f"  LocalizationHead:")
        print(f"    Input: {input_channels} x {input_spatial_size} x {input_spatial_size}")
        print(f"    Decoder channels: {decoder_channels}")
        print(f"    Output: 1 x {output_size} x {output_size}")
        print(f"    Dropout: {dropout}")
        
        # Calculer le facteur d'upsampling nécessaire
        self.upsample_factor = output_size / input_spatial_size
        
        # Construire le décodeur
        decoder_layers = []
        prev_channels = input_channels
        
        for i, out_channels in enumerate(decoder_channels):
            # Bloc de décodage
            decoder_layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ])
            
            # Upsampling progressif (sauf dernière couche)
            if i < len(decoder_channels) - 1:
                decoder_layers.append(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            
            prev_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Couche finale pour produire la heatmap
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Heatmap entre 0 et 1
        )
        
        # Upsampling final pour atteindre la taille exacte
        self.final_upsample = nn.Upsample(
            size=(output_size, output_size), 
            mode='bilinear', 
            align_corners=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features spatiales (B, input_channels, H, W)
               où H=W=input_spatial_size
        
        Returns:
            heatmap: (B, 1, output_size, output_size)
        """
        # Vérification de la taille d'entrée
        if x.shape[2] != self.input_spatial_size or x.shape[3] != self.input_spatial_size:
            # Si la taille ne correspond pas, on interpole
            x = F.interpolate(
                x, 
                size=(self.input_spatial_size, self.input_spatial_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Décodage
        x = self.decoder(x)
        
        # Convolution finale
        heatmap = self.final_conv(x)
        
        # Upsampling final vers la taille cible
        heatmap = self.final_upsample(heatmap)
        
        return heatmap

class RadDinoWrapper(nn.Module):
    """Wrapper pour extraire les embeddings de RadDino"""
    
    def __init__(self, model_name: str = "microsoft/rad-dino"):
        super().__init__()
        from rad_dino import RadDino
        
        self.model = RadDino()
        self.embed_dim = 768
        self.spatial_size = 37  # 224/6 ≈ 37 (pas de patch de ~6x6)
        self.num_patches = self.spatial_size * self.spatial_size  # 1369
        
        print(f"✓ RadDino chargé")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Spatial size: {self.spatial_size}x{self.spatial_size}")
        print(f"  Num patches: {self.num_patches}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Images (B, 3, 224, 224) normalisées [0, 1]
        
        Returns:
            cls_embeddings: (B, 768) - embeddings globaux
            patch_embeddings: (B, 1369, 768) - embeddings des patches
        """
        # RadDino retourne (cls_token, spatial_features)
        cls_token, spatial_features = self.model(x)
        
        # cls_token: (B, 768) - déjà prêt
        # spatial_features: (B, 768, 37, 37) - à convertir en séquence
        
        B, C, H, W = spatial_features.shape
        
        # Reshape en séquence de patches: (B, 768, 37, 37) → (B, 1369, 768)
        patch_embeddings = spatial_features.flatten(2).transpose(1, 2)
        
        return cls_token, patch_embeddings
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias pour compatibilité"""
        return self.forward(x)

    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Version robuste qui gère les sorties 2D et 4D.
        """
        B = x.size(0)
        
        # Preprocessing
        x = self.preprocess(x)
        
        # Forward pass
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        
        # Tentative 1: last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            print(f"[DEBUG] last_hidden_state shape: {hidden.shape}")
            
            if hidden.dim() == 3:  # (B, 1+N, C)
                cls_embeddings = hidden[:, 0, :]
                patch_embeddings = hidden[:, 1:, :]
            
            elif hidden.dim() == 4:  # (B, C, H, W)
                # Pas de CLS token explicite, utiliser pooling
                cls_embeddings = hidden.mean(dim=[2, 3])  # (B, C)
                
                # Convertir patches en séquence
                B, C, H, W = hidden.shape
                patch_embeddings = hidden.flatten(2, 3).permute(0, 2, 1)  # (B, H*W, C)
            
            else:
                raise ValueError(f"Unexpected hidden state dim: {hidden.dim()}")
        
        # Tentative 2: pooler_output (pour CLS) + hidden_states (pour patches)
        elif hasattr(outputs, 'pooler_output') and hasattr(outputs, 'hidden_states'):
            cls_embeddings = outputs.pooler_output
            
            # Prendre le dernier hidden state pour les patches
            last_hidden = outputs.hidden_states[-1]
            
            if last_hidden.dim() == 4:
                B, C, H, W = last_hidden.shape
                patch_embeddings = last_hidden.flatten(2, 3).permute(0, 2, 1)
            else:
                patch_embeddings = last_hidden[:, 1:, :]  # Ignorer CLS
        
        else:
            raise ValueError(f"Unsupported output format: {outputs.keys()}")
        
        print(f"[DEBUG] Final shapes:")
        print(f"  cls_embeddings: {cls_embeddings.shape}")
        print(f"  patch_embeddings: {patch_embeddings.shape}\n")
        
        return cls_embeddings, patch_embeddings

class ChestXRayMultiTask(nn.Module):
    """Architecture multi-task complète."""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        num_patches: int = 1369,  # 37x37 pour RadDino
        output_size: int = 224,
        shared_hidden: int = 512,
        classification_hidden: List[int] = [256, 128],
        localization_channels: List[int] = [512, 256, 128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()

        print(f"\n{'='*80}")
        print(f"CONSTRUCTION DU MODÈLE MULTI-TASK")
        print(f"{'='*80}")

        # Encoder
        self.encoder = encoder
        self.encoder_frozen = True
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_grid_size = int(math.sqrt(num_patches))  # 37 pour RadDino
        self.output_size = output_size

        print(f"\n1. ENCODER:")
        print(f"   Type: {type(encoder).__name__}")
        print(f"   Embed dim: {embed_dim}")
        print(f"   Num patches: {num_patches} ({self.patch_grid_size}x{self.patch_grid_size})")

        # Projecteurs partagés
        print(f"\n2. PROJECTEURS PARTAGÉS:")
        self.shared_cls_proj = SharedProjector(
            input_dim=embed_dim,
            output_dim=shared_hidden,
            dropout=dropout
        )
        self.shared_patch_proj = SharedProjector(
            input_dim=embed_dim,
            output_dim=shared_hidden,
            dropout=dropout
        )

        print(f"    SharedProjector: {embed_dim} -> {shared_hidden}")
        print(f"    SharedProjector: {embed_dim} -> {shared_hidden}")

        # Classification Head
        print(f"\n3. CLASSIFICATION HEAD:")
        self.classification_head = ClassificationHead(
            input_dim=shared_hidden,
            hidden_dims=classification_hidden,
            dropout=dropout
        )

        # Localization Head
        print(f"\n4. LOCALIZATION HEAD:")
        self.localization_head = LocalizationHead(
            input_channels=shared_hidden,
            input_spatial_size=self.patch_grid_size,
            decoder_channels=localization_channels,
            output_size=output_size,
            dropout=dropout
        )

        print(f"\n{'='*80}")
        print(f"✓ MODÈLE MULTI-TASK CONSTRUIT")
        print(f"{'='*80}\n")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass avec gestion du mode frozen.

        Args:
            x: (B, 3, 224, 224) - Images d'entrée

        Returns:
            cls_logits: (B, 1) - Logits de classification
            heatmap: (B, 1, 224, 224) - Heatmap de localisation
            cls_embeddings: (B, 768) - Features CLS (pour analyse)
        """
        B = x.size(0)

        # Extraction des features de l'encodeur
        if self.encoder_frozen:
            # Mode gelé : no_grad + clone pour permettre le backward dans les heads
            with torch.no_grad():
                cls_embeddings, spatial_features = self.encoder(x)
            
            # Clone et détache pour créer des tenseurs normaux avec requires_grad=True
            cls_embeddings = cls_embeddings.clone().detach().requires_grad_(True)
            spatial_features = spatial_features.clone().detach().requires_grad_(True)
        else:
            # Mode dégelé : gradient actif partout
            cls_embeddings, spatial_features = self.encoder(x)
        
        # Convertir spatial_features au bon format si nécessaire
        if spatial_features.dim() == 4:
            # Format (B, C, H, W) → (B, N, C)
            B, C, H, W = spatial_features.shape
            patch_embeddings = spatial_features.flatten(2).transpose(1, 2)
        else:
            # Déjà au format (B, N, C)
            patch_embeddings = spatial_features
        
        # Projection partagée
        cls_shared = self.shared_cls_proj(cls_embeddings)  # (B, 512)
        patch_shared = self.shared_patch_proj(patch_embeddings)  # (B, N, 512)
        
        # Classification (utilise cls_token)
        cls_logits = self.classification_head(cls_shared)  # (B, 1)
        
        # Localisation (utilise patch embeddings)
        patch_2d = self._reshape_patches(patch_shared)  # (B, 512, H, W)
        heatmap = self.localization_head(patch_2d)  # (B, 1, 224, 224)
        
        return cls_logits, heatmap, cls_embeddings


    def _reshape_patches(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reshape patches 1D -> 2D.
        
        Args:
            patch_embeddings: (B, N, C) où N = H*W
        
        Returns:
            patch_2d: (B, C, H, W)
        """
        B, N, C = patch_embeddings.shape
        H = W = self.patch_grid_size  # 37
        assert H * W == N, f"num_patches={N} != {H}x{W}={H*W}"
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        return patch_embeddings.transpose(1, 2).reshape(B, C, H, W)

    def freeze_encoder(self):
        """Gèle l'encodeur."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        print("✓ Encoder gelé (Phase 1)")

    def unfreeze_encoder(self, layers_to_unfreeze: Optional[List[str]] = None):
        """
        Dégèle l'encodeur (totalement ou partiellement).
        
        Args:
            layers_to_unfreeze: Liste de noms de couches à dégeler.
                               Si None, dégèle tout.
        """
        if layers_to_unfreeze is None:
            # Dégel complet
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("✓ Encoder complètement dégelé")
        else:
            # Dégel partiel
            for name, param in self.encoder.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
            print(f"✓ Encoder partiellement dégelé: {layers_to_unfreeze}")
        
        self.encoder_frozen = False

# ============================================================================
# LOSS HIÉRARCHIQUE
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Loss hiérarchique: Classification + Localisation conditionnelle."""
    
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_loc: float = 2.0,
        lambda_smooth: float = 0.1,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.lambda_smooth = lambda_smooth
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss pour classification déséquilibrée."""
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        alpha = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        loss = alpha * focal_weight * bce_loss
        
        return loss.mean()
    
    def smoothness_loss(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Pénalise les variations brusques dans la heatmap."""
        diff_y = torch.abs(heatmap[:, :, 1:, :] - heatmap[:, :, :-1, :])
        diff_x = torch.abs(heatmap[:, :, :, 1:] - heatmap[:, :, :, :-1])
        return diff_y.mean() + diff_x.mean()
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        cls_targets: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_logits: (B, 1)
            pred_heatmaps: (B, 1, H, W)
            cls_targets: (B, 1)
            target_heatmaps: (B, 1, H, W)
        Returns:
            total_loss, loss_dict
        """
        # Classification Loss
        if self.use_focal_loss:
            cls_loss = self.focal_loss(cls_logits, cls_targets)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
        
        # Localisation Loss (seulement sur positifs)
        positive_mask = (cls_targets > 0.5).squeeze(1)
        
        if positive_mask.sum() > 0:
            pred_pos = pred_heatmaps[positive_mask]
            target_pos = target_heatmaps[positive_mask]
            loc_loss = F.mse_loss(pred_pos, target_pos)
            
            # Smoothness
            smooth_loss = self.smoothness_loss(pred_pos)
        else:
            loc_loss = torch.tensor(0.0, device=cls_logits.device)
            smooth_loss = torch.tensor(0.0, device=cls_logits.device)
        
        # Loss totale
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_loc * loc_loss +
            self.lambda_smooth * smooth_loss
        )
        
        loss_dict = {
            "total": total_loss.item(),
            "classification": cls_loss.item(),
            "localization": loc_loss.item(),
            "smoothness": smooth_loss.item(),
        }
        
        return total_loss, loss_dict


# ============================================================================
# TRAINER
# ============================================================================

class ModelTrainer:
    """Trainer optimisé pour gros dataset (66K images)."""
    
    def __init__(
        self,
        model: ChestXRayMultiTask,
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        heatmap_generator: GaussianHeatmapGenerator,
        device: torch.device,
        output_dir: str = "checkpoints_multitask_66k",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        grad_clip_norm: float = 1.0,
        log_interval: int = 50,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.heatmap_generator = heatmap_generator
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.scaler = GradScaler() if use_amp else None
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_cls_loss": [],
            "val_cls_loss": [],
            "train_loc_loss": [],
            "val_loc_loss": [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10,
        checkpoint_frequency: int = 5,
    ):
        """Entraînement complet."""
        print(f"\n{'='*80}")
        print(f"DÉBUT DE L'ENTRAÎNEMENT - {num_epochs} EPOCHS")
        print(f"{'='*80}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*80}\n")
            
            # Train
            train_metrics = self.train_one_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Historique
            self.history["train_loss"].append(train_metrics["avg_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_cls_loss"].append(train_metrics["avg_cls_loss"])
            self.history["val_cls_loss"].append(val_metrics["val_cls_loss"])
            self.history["train_loc_loss"].append(train_metrics["avg_loc_loss"])
            self.history["val_loc_loss"].append(val_metrics["val_loc_loss"])
            
            # Scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["val_loss"])
                else:
                    scheduler.step()
            
            # Sauvegarde meilleur modèle
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                patience_counter = 0
                
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Nouveau meilleur modèle! Val Loss: {self.best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Checkpoint régulier
            if epoch % checkpoint_frequency == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping déclenché (patience={early_stopping_patience})")
                break
            
            # Résumé epoch
            print(f"\nRésumé Epoch {epoch}:")
            print(f"  Train Loss: {train_metrics['avg_loss']:.6f} | Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"  Best Val Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})")
            print(f"  Patience: {patience_counter}/{early_stopping_patience}")
        
        # Sauvegarde historique
        self.save_history()
        
        print(f"\n{'='*80}")
        print(f"ENTRAÎNEMENT TERMINÉ")
        print(f"{'='*80}")
        print(f"Meilleur modèle: Epoch {self.best_epoch} | Val Loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints: {self.output_dir}\n")
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Une epoch d'entraînement."""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels, coords_batch, meta) in enumerate(train_loader, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                cls_logits, pred_heatmaps, _ = self.model(images)
                target_heatmaps = self.heatmap_generator.batch_generate(coords_batch, self.device)
                
                loss, loss_dict = self.criterion(cls_logits, pred_heatmaps, labels, target_heatmaps)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if batch_idx % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulation
            total_loss += loss_dict["total"]
            total_cls_loss += loss_dict["classification"]
            total_loc_loss += loss_dict["localization"]
            num_batches += 1
            
            # Log
            if batch_idx % self.log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx * images.size(0)) / elapsed
                
                print(f"  [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss_dict['total']:.6f} "
                      f"(Cls: {loss_dict['classification']:.6f}, "
                      f"Loc: {loss_dict['localization']:.6f}) | "
                      f"{samples_per_sec:.1f} samples/s")
        
        return {
            "avg_loss": total_loss / num_batches,
            "avg_cls_loss": total_cls_loss / num_batches,
            "avg_loc_loss": total_loc_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validation."""
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0
        num_batches = 0
        
        for images, labels, coords_batch, meta in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            cls_logits, pred_heatmaps, _ = self.model(images)
            target_heatmaps = self.heatmap_generator.batch_generate(coords_batch, self.device)
            
            loss, loss_dict = self.criterion(cls_logits, pred_heatmaps, labels, target_heatmaps)
            
            total_loss += loss_dict["total"]
            total_cls_loss += loss_dict["classification"]
            total_loc_loss += loss_dict["localization"]
            num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "val_cls_loss": total_cls_loss / num_batches,
            "val_loc_loss": total_loc_loss / num_batches,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }
        
        if is_best:
            path = os.path.join(self.output_dir, "best_model.pt")
        else:
            path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Sauvegarde l'historique d'entraînement."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Historique sauvegardé: {history_path}")


# ============================================================================
# CRÉATION DU MODÈLE ET COMPOSANTS
# ============================================================================

def create_model_and_components(
    device: torch.device,
    image_size: int = 224,
    learning_rate_head: float = 1e-3,
    learning_rate_backbone: float = 1e-5,
    lambda_cls: float = 1.0,
    lambda_loc: float = 2.0,
    lambda_smooth: float = 0.1,
    freeze_encoder: bool = True,
):
    """Crée le modèle et les composants d'entraînement."""
    
    print("Création du modèle...")
    
    # Encodeur RadDino
    try:
        encoder = RadDino()
        print("✓ RadDino chargé")
        embed_dim = 768
        # RadDino produit des features spatiales de 37x37
        num_patches = 37 * 37  # 1369
    except Exception as e:
        print(f"⚠ Erreur lors du chargement de RadDino: {e}")
        print("Utilisation d'un mock")
        class MockRadDino(nn.Module):
            def forward(self, x):
                B = x.size(0)
                cls_embeddings = torch.randn(B, 768, device=x.device)
                # Retourne des features spatiales (B, 768, 37, 37)
                spatial_features = torch.randn(B, 768, 37, 37, device=x.device)
                return cls_embeddings, spatial_features
        encoder = MockRadDino()
        embed_dim = 768
        num_patches = 37 * 37
    
    # Modèle
    model = ChestXRayMultiTask(
        encoder=encoder,
        embed_dim=embed_dim,
        num_patches=num_patches,
        output_size=image_size,
        shared_hidden=512,
        classification_hidden=[256, 128],
        localization_channels=[512, 256, 128, 64],
        dropout=0.3
    ).to(device)
    
    if freeze_encoder:
        model.freeze_encoder()
    
    print(f"✓ Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")
    print(f"  Entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
   
    # Loss
    criterion = MultiTaskLoss(
        lambda_cls=lambda_cls,
        lambda_loc=lambda_loc,
        lambda_smooth=lambda_smooth,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    print("✓ Loss configurée")
    
    # Optimiseur avec learning rates différenciés
    param_groups = [
        {
            'params': model.shared_cls_proj.parameters(),
            'lr': learning_rate_head,
            'name': 'shared_cls_proj'
        },
        {
            'params': model.shared_patch_proj.parameters(),
            'lr': learning_rate_head,
            'name': 'shared_patch_proj'
        },
        {
            'params': model.classification_head.parameters(),
            'lr': learning_rate_head,
            'name': 'classification_head'
        },
        {
            'params': model.localization_head.parameters(),
            'lr': learning_rate_head,
            'name': 'localization_head'
        },
        {
            'params': model.encoder.parameters(),
            'lr': learning_rate_backbone,
            'name': 'encoder'
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    print("✓ Optimizer AdamW configuré")
    
    # Générateur de heatmaps
    heatmap_generator = GaussianHeatmapGenerator(image_size=image_size, sigma=10.0)
    print("✓ Heatmap generator créé")
    
    return model, criterion, optimizer, heatmap_generator


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHEST X-RAY MULTI-TASK LEARNING - 66K IMAGES")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    


    # Configuration
    config = {
        "batch_size": 64,
        "num_workers": 8,
        "image_size": 224,
        "num_epochs": 50,
        "learning_rate_head": 1e-3,
        "learning_rate_backbone": 1e-5,
        "lambda_cls": 1.0,
        "lambda_loc": 2.0,
        "lambda_smooth": 0.1,
        "freeze_encoder": True,
        "use_amp": torch.cuda.is_available(),
        "gradient_accumulation_steps": 1,
        "grad_clip_norm": 1.0,
        "val_split": 0.15,
        "test_split": 0.15,
        "scheduler_type": "cosine",
        "warmup_epochs": 5,
        "early_stopping_patience": 15,
        "checkpoint_frequency": 5,
        "log_interval": 50,
        "output_dir": "checkpoints_multitask_66k",
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    
    
    # Dataloaders
    print("Création des dataloaders...")
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        augment_train=True,
    )
    print()
    
    # Modèle et composants
    model, criterion, optimizer, heatmap_generator = create_model_and_components(
        device=device,
        image_size=config["image_size"],
        learning_rate_head=config["learning_rate_head"],
        learning_rate_backbone=config["learning_rate_backbone"],
        lambda_cls=config["lambda_cls"],
        lambda_loc=config["lambda_loc"],
        lambda_smooth=config["lambda_smooth"],
        freeze_encoder=config["freeze_encoder"],
    )
    print()
    
    # Scheduler
    print("Configuration du scheduler...")
    if config["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],
            eta_min=1e-6
        )
        print("✓ CosineAnnealingLR scheduler créé")
    elif config["scheduler_type"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config["learning_rate_head"]] * 4 + [config["learning_rate_backbone"]],
            epochs=config["num_epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=config["warmup_epochs"] / config["num_epochs"],
        )
        print("✓ OneCycleLR scheduler créé")
    else:
        scheduler = None
        print("Pas de scheduler")
    
    print()
    
    # Trainer
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        heatmap_generator=heatmap_generator,
        device=device,
        output_dir=config["output_dir"],
        use_amp=config["use_amp"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        grad_clip_norm=config["grad_clip_norm"],
        log_interval=config["log_interval"],
    )
    
    # PHASE 1: Fine-tuning des heads (encoder gelé)
    print("\n" + "="*80)
    print("PHASE 1: FINE-TUNING DES HEADS (ENCODER GELÉ)")
    print("="*80 + "\n")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"] // 2,
        scheduler=scheduler,
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_frequency=config["checkpoint_frequency"],
    )
    
    # PHASE 2: Fine-tuning complet (dégel progressif)
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING COMPLET (DÉGEL DU BACKBONE)")
    print("="*80 + "\n")
    
    # Dégel des dernières couches
    model.unfreeze_encoder(layers_to_unfreeze=["blocks.10", "blocks.11"])
    
    # Réduction LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    
    print("Learning rates ajustés (x0.1)")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Group {i} ({param_group.get('name', 'unnamed')}): {param_group['lr']:.6f}")
    print()
    
    # Nouveau scheduler
    if config["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"] // 2,
            eta_min=1e-7
        )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"] // 2,
        scheduler=scheduler,
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_frequency=config["checkpoint_frequency"],
    )
    
    # Évaluation finale
    print("\n" + "="*80)
    print("ÉVALUATION FINALE SUR TEST SET")
    print("="*80 + "\n")
    
    test_metrics = trainer.validate(test_loader, epoch=-1)
    
    print("Résultats finaux:")
    print(f"  Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"  Test Cls Loss: {test_metrics['val_cls_loss']:.4f}")
    print(f"  Test Loc Loss: {test_metrics['val_loc_loss']:.4f}")
    
    # Sauvegarde résultats
    results = {
        "config": config,
        "test_metrics": test_metrics,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
    }
    
    results_path = os.path.join(config["output_dir"], "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {results_path}")
