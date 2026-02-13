from rad_dino import RadDino
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import time
from datetime import datetime
import os
import pandas as pd
import numpy as np
from PIL import Image


LIDC_DIR = "../nodulocc_dataset/lidc_png_16_bit"
NIH_DIR  = "../nodulocc_dataset/nih_filtered_images"

LOCALIZATION_CSV = "../nodulocc_dataset/localization_labels.csv"
CLASSIFICATION_CSV = "../nodulocc_dataset/classification_labels.csv"

def _normalize_csv_path(p: str) -> str:
    """Normalise une valeur de CSV qui peut être un chemin ou juste un nom de fichier."""
    p = str(p)
    p = p.strip().strip('"').strip("'")
    return os.path.basename(p)

def _load_png(path: str) -> torch.Tensor:
    """Charge un PNG (8-bit ou 16-bit) en tenseur float32 (C,H,W) dans [0,1].

    Normalise aussi le nombre de canaux à 3 (RGB) car l'encodeur attend 3 canaux.
    Certains PNG peuvent être RGBA (4 canaux) ou palettisés.
    """
    img = Image.open(path)

    # Assure un format exploitable et stable côté canaux.
    # - Conserve les images 16-bit grayscale (souvent mode "I;16")
    # - Convertit les formats palettisés/alpha vers RGB (3 canaux)
    if img.mode in {"P", "RGBA", "LA", "CMYK", "YCbCr", "RGBa"}:
        img = img.convert("RGB")
    elif img.mode not in {"RGB", "L", "I;16", "I"}:
        # fallback défensif
        img = img.convert("RGB")

    arr = np.array(img)

    # Normalisation selon profondeur
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0

    # Harmonise les canaux -> 3 canaux
    if arr.ndim == 2:
        # grayscale
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 2:
            # LA (au cas où) -> garde L
            arr = np.repeat(arr[:, :, :1], 3, axis=2)
        elif arr.shape[2] >= 4:
            # RGBA / autres -> drop alpha / canaux supplémentaires
            arr = arr[:, :, :3]

    # (H,W,C) -> (C,H,W)
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t

def load_local_image(dataset: str, filename: str) -> torch.Tensor:
    """Retourne une image en tenseur (1,3,H,W) depuis LIDC ou NIH."""
    base = LIDC_DIR if dataset.lower() == "lidc" else NIH_DIR
    img_path = os.path.join(base, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image introuvable: {img_path}")
    x = _load_png(img_path).unsqueeze(0)  # (1,C,H,W)
    return x

# ---------------------------------------------------------------------------
# Dataset + DataLoader (LIDC + NIH) à partir des CSV
# ---------------------------------------------------------------------------

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


def _infer_binary_label_column(df: pd.DataFrame) -> str:
    """Trouve une colonne de label binaire (0/1) dans un CSV."""
    candidates = [
        "label", "target", "y", "has_nodule", "nodule", "presence", "class",
        "No Finding"  # parfois utilisé (à convertir ensuite)
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: si une unique colonne numérique (hors id) semble être le label
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "if" and c.lower() not in {"id", "image_id"}]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(
        "Impossible d'inférer la colonne de label binaire. "
        f"Colonnes disponibles: {list(df.columns)}."
    )


def _extract_coords_from_row(row: pd.Series, img_w: int, img_h: int) -> List[Tuple[float, float]]:
    """Extrait une liste de coordonnées (x,y) normalisées depuis une ligne de CSV.

    Supporte plusieurs formats courants:
    - centre déjà normalisé: (x, y) ou (x_center, y_center)
    - bbox en pixels: (xmin, ymin, xmax, ymax) ou (x_min, y_min, x_max, y_max)
    - bbox normalisée: idem mais valeurs dans [0,1]
    """
    cols = {c.lower(): c for c in row.index}

    # 1) centre directement (normalisé ou pixels)
    for xk, yk in [("x", "y"), ("x_center", "y_center"), ("xc", "yc")]:
        if xk in cols and yk in cols:
            x = float(row[cols[xk]])
            y = float(row[cols[yk]])
            # si > 1.5 on suppose pixels
            if x > 1.5 or y > 1.5:
                x = x / max(img_w, 1)
                y = y / max(img_h, 1)
            # clamp
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

            # si bbox en pixels
            if max(xmax, ymax) > 1.5:
                xmin = xmin / max(img_w, 1)
                xmax = xmax / max(img_w, 1)
                ymin = ymin / max(img_h, 1)
                ymax = ymax / max(img_h, 1)

            # centre bbox
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return [(x, y)]

    # 3) pas d'annotation exploitable
    return []


def _resize_tensor_image(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize bilinear d'un tenseur image (C,H,W) vers (C,size,size)."""
    if x.dim() != 3:
        raise ValueError(f"Attendu un tenseur (C,H,W), reçu: {tuple(x.shape)}")
    x = x.unsqueeze(0)  # (1,C,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


class LungMultiSourceDataset(Dataset):
    """Dataset unifié LIDC (localization) + NIH (classification).

    - LIDC: renvoie (image, cls_label=1, coords) si coords trouvées, sinon label peut être 0.
    - NIH: renvoie (image, cls_label binaire, coords=[])

    Remarque: on sort `coords` (liste de tuples) plutôt qu'une heatmap.
    La heatmap est générée ensuite via `GaussianHeatmapGenerator`.
    """

    def __init__(
        self,
        lidc_dir: str = LIDC_DIR,
        nih_dir: str = NIH_DIR,
        localization_csv: str = LOCALIZATION_CSV,
        classification_csv: str = CLASSIFICATION_CSV,
        image_size: int = 224,
    ):
        super().__init__()
        self.lidc_dir = lidc_dir
        self.nih_dir = nih_dir
        self.image_size = image_size

        # --- LIDC (localization)
        df_loc = pd.read_csv(localization_csv)
        loc_file_col = _infer_filename_column(df_loc)
        df_loc = df_loc.copy()
        df_loc["_dataset"] = "lidc"
        df_loc["_filename"] = df_loc[loc_file_col].apply(_normalize_csv_path).astype(str)
        df_loc["_abs_path"] = df_loc["_filename"].apply(lambda fn: os.path.join(self.lidc_dir, fn))

        # --- NIH (classification)
        df_cls = pd.read_csv(classification_csv)
        cls_file_col = _infer_filename_column(df_cls)
        cls_label_col = _infer_binary_label_column(df_cls)
        df_cls = df_cls.copy()
        df_cls["_dataset"] = "nih"
        df_cls["_filename"] = df_cls[cls_file_col].apply(_normalize_csv_path).astype(str)
        df_cls["_label_col"] = cls_label_col
        df_cls["_abs_path"] = df_cls["_filename"].apply(lambda fn: os.path.join(self.nih_dir, fn))

        # Concat
        df_all = pd.concat([df_loc, df_cls], ignore_index=True)

        # Filtre: garde uniquement les fichiers présents sur disque
        exists_mask = df_all["_abs_path"].apply(lambda p: os.path.exists(p))
        n_missing = int((~exists_mask).sum())
        if n_missing > 0:
            missing_examples = df_all.loc[~exists_mask, ["_dataset", "_filename", "_abs_path"]].head(10)
            print(f"[Dataset] WARNING: {n_missing} images manquantes -> elles seront ignorées.")
            print("[Dataset] Exemples manquants:\n", missing_examples.to_string(index=False))

        self.df = df_all.loc[exists_mask].reset_index(drop=True)
        print(f"[Dataset] Images valides: {len(self.df)} / {len(df_all)}")

    def __len__(self) -> int:
        return len(self.df)

    def _abs_path(self, dataset: str, filename: str) -> str:
        base = self.lidc_dir if dataset == "lidc" else self.nih_dir
        return os.path.join(base, _normalize_csv_path(filename))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dataset = str(row["_dataset"]).lower()
        filename = str(row["_filename"])

        # Le dataset est déjà filtré sur l'existence des fichiers,
        # mais on garde une vérif defensive.
        path = row.get("_abs_path", None)
        if path is None:
            path = self._abs_path(dataset, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image introuvable (après filtrage): {path}")

        # Charge image (C,H,W) puis resize
        x = _load_png(path)
        # pour extraire coords en pixels on a besoin de la taille originale
        _, h0, w0 = x.shape
        x = _resize_tensor_image(x, self.image_size)

        # Labels
        if dataset == "nih":
            label_col = row.get("_label_col", None)
            y_raw = row[label_col] if label_col is not None else row.get("label", 0)

            # Convertit en 0/1 le plus robustement possible
            if isinstance(y_raw, str):
                y_str = y_raw.strip().lower()
                # Heuristique: "no finding" => 0, sinon 1
                y = 0.0 if y_str in {"0", "false", "no", "no finding", "nofinding"} else 1.0
            else:
                y = float(y_raw)
                y = 1.0 if y >= 0.5 else 0.0

            coords: List[Tuple[float, float]] = []

        else:
            # LIDC: si coords dispo -> positif, sinon négatif
            coords = _extract_coords_from_row(row, img_w=w0, img_h=h0)
            y = 1.0 if len(coords) > 0 else 0.0

        return {
            "image": x,  # (3, H, W)
            "label": torch.tensor([y], dtype=torch.float32),  # (1,)
            "coords": coords,
            "dataset": dataset,
            "filename": filename,
            "path": path,
        }


def multitask_collate_fn(batch: List[Dict]):
    """Collate pour batch avec coords de longueur variable."""
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,H,W)
    labels = torch.stack([b["label"] for b in batch], dim=0)  # (B,1)
    coords_batch = [b["coords"] for b in batch]
    meta = {
        "dataset": [b["dataset"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "path": [b["path"] for b in batch],
    }
    return images, labels, coords_batch, meta


def create_dataloader(
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    image_size: int = 224,
    lidc_dir: str = LIDC_DIR,
    nih_dir: str = NIH_DIR,
    localization_csv: str = LOCALIZATION_CSV,
    classification_csv: str = CLASSIFICATION_CSV,
) -> DataLoader:
    """Crée un DataLoader unifié (LIDC+NIH)."""
    ds = LungMultiSourceDataset(
        lidc_dir=lidc_dir,
        nih_dir=nih_dir,
        localization_csv=localization_csv,
        classification_csv=classification_csv,
        image_size=image_size,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )

# ---------------------------------------------------------------------------
# Exemple de chargement local + features + CSV (optionnel)
# ---------------------------------------------------------------------------

def _debug_local_loading_example() -> None:
    """Petit smoke-test optionnel pour vérifier le chargement local + RadDino."""
    encoder = RadDino()

    # Exemple: extraction de features sur une image locale LIDC
    image = load_local_image("lidc", "0001.png")
    cls_embeddings, patch_embeddings = encoder.extract_features(image)
    print("LIDC 0001.png ->", cls_embeddings.shape, patch_embeddings.shape)

    # Exemple: extraction de features sur une image locale NIH
    image2 = load_local_image("nih", "00000002_000.png")
    cls_embeddings2, patch_embeddings2 = encoder.extract_features(image2)
    print("NIH 00000002_000.png ->", cls_embeddings2.shape, patch_embeddings2.shape)

    # Chargement des CSV labels (pour intégration ultérieure dans un Dataset/DataLoader)
    try:
        df_loc = pd.read_csv(LOCALIZATION_CSV)
        df_cls = pd.read_csv(CLASSIFICATION_CSV)
        print("localization_labels.csv:", df_loc.shape, "| classification_labels.csv:", df_cls.shape)
    except Exception as e:
        print("Impossible de lire les CSV de labels:", e)

class ClassificationHead(nn.Module):
    """Head MLP pour la classification binaire (présence/absence de nodules)."""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embeddings: (B, embed_dim)
        Returns:
            logits: (B, 1) - logits avant sigmoid
        """
        return self.mlp(cls_embeddings)


class LocalizationHead(nn.Module):
    """Head convolutionnel pour générer une heatmap de localisation."""
    
    def __init__(
        self, 
        embed_dim: int, 
        patch_size: int = 16,
        output_size: int = 224,
        hidden_channels: List[int] = [256, 128, 64, 32]
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_size = output_size
        
        # Projection initiale pour réduire la dimension
        self.proj = nn.Conv2d(embed_dim, hidden_channels[0], kernel_size=1)
        
        # Décodeur avec upsampling progressif
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.extend([
                nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ])
        
        self.decoder = nn.Sequential(*layers)
        
        # Couche finale pour la heatmap
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
    
    def forward(self, patch_embeddings: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (B, N, embed_dim) où N = num_patches²
            num_patches: nombre de patches par dimension (ex: 14 pour 14x14)
        Returns:
            heatmap: (B, 1, H, W) - heatmap de localisation
        """
        B, N, C = patch_embeddings.shape
        
        # Reshape en grille spatiale 2D
        patch_grid = patch_embeddings.transpose(1, 2).reshape(B, C, num_patches, num_patches)
        
        # Décodeur convolutionnel
        x = self.proj(patch_grid)
        x = self.decoder(x)
        
        # Interpolation finale pour atteindre la résolution cible
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        
        # Heatmap finale
        heatmap = self.final_conv(x)
        
        return heatmap


class ChestXRayMultiTask(nn.Module):
    """Modèle multi-task pour détection et localisation de nodules pulmonaires."""
    
    def __init__(
        self,
        encoder,  
        embed_dim: int = 768,
        num_patches: int = 14,
        output_size: int = 224,
        classification_hidden: int = 512,
        localization_channels: List[int] = [256, 128, 64, 32],
        dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = encoder
        self.num_patches = num_patches
        
        # Heads pour les deux tâches
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=classification_hidden,
            dropout=dropout
        )
        
        self.localization_head = LocalizationHead(
            embed_dim=embed_dim,
            patch_size=16,
            output_size=output_size,
            hidden_channels=localization_channels
        )
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W)
        Returns:
            cls_logits: (B, 1) - logits de classification
            heatmap: (B, 1, H, W) - heatmap de localisation
        """
        # Extraction des features avec RadDino
        cls_embeddings, patch_embeddings = self.encoder.extract_features(images)
        
        # Classification
        cls_logits = self.classification_head(cls_embeddings)
        
        # Localisation
        heatmap = self.localization_head(patch_embeddings, self.num_patches)
        
        return cls_logits, heatmap
    
    def freeze_encoder(self):
        """Gèle complètement l'encodeur RadDino."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder_partial(self, num_blocks: int = 2):
        """
        Dégèle les derniers blocs de l'encodeur pour fine-tuning.
        Args:
            num_blocks: nombre de blocs à dégeler depuis la fin
        """
        # Gèle d'abord tout
        self.freeze_encoder()
        
        # Dégèle les derniers blocs (adapter selon la structure de RadDino)
        if hasattr(self.encoder, 'blocks'):
            blocks = list(self.encoder.blocks)[-num_blocks:]
            for block in blocks:
                for param in block.parameters():
                    param.requires_grad = True


class GaussianHeatmapGenerator:
    """Générateur de heatmaps Gaussiennes pour les annotations de nodules."""
    
    def __init__(self, image_size: int = 224, sigma: float = 10.0):
        self.image_size = image_size
        self.sigma = sigma
    
    def generate(
        self, 
        nodule_coords: List[Tuple[float, float]], 
        device: torch.device
    ) -> torch.Tensor:
        """
        Génère une heatmap Gaussienne pour plusieurs nodules.
        
        Args:
            nodule_coords: Liste de (x, y) en coordonnées normalisées [0, 1]
            device: dispositif PyTorch
        Returns:
            heatmap: (1, H, W) - heatmap avec Gaussiennes centrées sur les nodules
        """
        heatmap = torch.zeros((self.image_size, self.image_size), device=device)
        
        if len(nodule_coords) == 0:
            return heatmap.unsqueeze(0)
        
        # Grille de coordonnées
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.image_size, device=device),
            torch.arange(self.image_size, device=device),
            indexing='ij'
        )
        
        # Ajoute une Gaussienne pour chaque nodule
        for x_norm, y_norm in nodule_coords:
            # Convertir coordonnées normalisées en pixels
            x_pixel = x_norm * self.image_size
            y_pixel = y_norm * self.image_size
            
            # Gaussienne 2D
            gaussian = torch.exp(
                -((x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2) / (2 * self.sigma ** 2)
            )
            
            # Superposition (max pour éviter dilution avec plusieurs nodules)
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
            batch_nodule_coords: Liste de listes de coordonnées, une par image
            device: dispositif PyTorch
        Returns:
            heatmaps: (B, 1, H, W)
        """
        heatmaps = []
        for nodule_coords in batch_nodule_coords:
            heatmap = self.generate(nodule_coords, device)
            heatmaps.append(heatmap)
        
        return torch.stack(heatmaps, dim=0)


class MultiTaskLoss(nn.Module):
    """Loss combinée pour classification et localisation."""
    
    def __init__(
        self, 
        lambda_cls: float = 1.0, 
        lambda_loc: float = 1.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if not use_focal_loss:
            self.cls_criterion = nn.BCEWithLogitsLoss()
        
        self.loc_criterion = nn.MSELoss()
    
    def focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal Loss pour gérer le déséquilibre de classes."""
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        if self.focal_alpha is not None:
            alpha_weight = targets * self.focal_alpha + (1 - targets) * (1 - self.focal_alpha)
            focal_weight = alpha_weight * focal_weight
        
        return (focal_weight * bce_loss).mean()
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        pred_heatmap: torch.Tensor,
        cls_targets: torch.Tensor,
        target_heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_logits: (B, 1) - logits de classification
            pred_heatmap: (B, 1, H, W) - heatmap prédite
            cls_targets: (B, 1) - labels binaires
            target_heatmap: (B, 1, H, W) - heatmap ground truth
        Returns:
            total_loss: loss totale
            loss_dict: détails des losses
        """
        # Classification loss
        if self.use_focal_loss:
            loss_cls = self.focal_loss(cls_logits, cls_targets)
        else:
            loss_cls = self.cls_criterion(cls_logits, cls_targets)
        
        # Localisation loss (seulement pour images positives)
        has_nodule = cls_targets.squeeze(-1) > 0.5
        if has_nodule.sum() > 0:
            loss_loc = self.loc_criterion(
                pred_heatmap[has_nodule],
                target_heatmap[has_nodule]
            )
        else:
            loss_loc = torch.tensor(0.0, device=cls_logits.device)
        
        # Loss totale
        total_loss = self.lambda_cls * loss_cls + self.lambda_loc * loss_loc
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': loss_cls.item(),
            'localization': loss_loc.item()
        }
        
        return total_loss, loss_dict


def training_step_example(
    model: ChestXRayMultiTask,
    images: torch.Tensor,
    cls_labels: torch.Tensor,
    nodule_coords_batch: List[List[Tuple[float, float]]],
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    heatmap_generator: GaussianHeatmapGenerator,
    device: torch.device
) -> Dict[str, float]:
    """
    Exemple de training step complet.
    
    Args:
        model: modèle multi-task
        images: (B, C, H, W)
        cls_labels: (B,) - labels binaires
        nodule_coords_batch: liste de coordonnées pour chaque image
        criterion: fonction de loss
        optimizer: optimiseur
        heatmap_generator: générateur de heatmaps
        device: dispositif PyTorch
    
    Returns:
        loss_dict: dictionnaire des losses
    """
    model.train()
    
    # Transfert vers device
    images = images.to(device)
    cls_labels = cls_labels.to(device).unsqueeze(1).float()
    
    # Génération des heatmaps ground truth
    target_heatmaps = heatmap_generator.batch_generate(nodule_coords_batch, device)
    
    # Forward pass
    cls_logits, pred_heatmaps = model(images)
    
    # Calcul de la loss
    loss, loss_dict = criterion(cls_logits, pred_heatmaps, cls_labels, target_heatmaps)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss_dict


def create_model_and_training_components(
    encoder,
    device: torch.device,
    learning_rate_head: float = 1e-3,
    learning_rate_backbone: float = 1e-5,
    lambda_cls: float = 1.0,
    lambda_loc: float = 1.0,
    use_focal_loss: bool = True
):
    """
    Crée le modèle et les composants d'entraînement.
    
    Returns:
        model, criterion, optimizer, heatmap_generator
    """
    # Modèle
    model = ChestXRayMultiTask(
        encoder=encoder,
        embed_dim=768,  # Adapter selon RadDino
        num_patches=14,  # Adapter selon taille image et patch size
        output_size=224
    ).to(device)
    
    # Phase 1: Gèle l'encodeur
    model.freeze_encoder()
    
    # Loss
    criterion = MultiTaskLoss(
        lambda_cls=lambda_cls,
        lambda_loc=lambda_loc,
        use_focal_loss=use_focal_loss
    )
    
    # Optimiseur avec LR différenciés
    optimizer = torch.optim.AdamW([
        {'params': model.classification_head.parameters(), 'lr': learning_rate_head},
        {'params': model.localization_head.parameters(), 'lr': learning_rate_head},
        {'params': model.encoder.parameters(), 'lr': learning_rate_backbone}
    ], weight_decay=0.01)
    
    # Générateur de heatmaps
    heatmap_generator = GaussianHeatmapGenerator(image_size=224, sigma=10.0)
    
    return model, criterion, optimizer, heatmap_generator


# ============================================================================
# EXEMPLE D'UTILISATION COMPLET
# ============================================================================

if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.backends.cuda.is_available() else 'cpu')
    batch_size = 32

    # Smoke-test DataLoader unifié (LIDC + NIH)
    try:
        dl = create_dataloader(batch_size=batch_size, shuffle=True, num_workers=0, image_size=224)
        images_dl, labels_dl, coords_dl, meta_dl = next(iter(dl))
        print("\nDataLoader batch:")
        print("  images:", images_dl.shape, "| labels:", labels_dl.shape)
        print("  coords lens:", [len(c) for c in coords_dl])
        print("  datasets:", meta_dl["dataset"])
    except Exception as e:
        print("\nDataLoader smoke-test échoué:", e)

    class MockRadDino(nn.Module):
        def extract_features(self, x):
            B = x.size(0)
            cls_embeddings = torch.randn(B, 768, device=x.device)
            patch_embeddings = torch.randn(B, 196, 768, device=x.device)  # 14x14 patches
            return cls_embeddings, patch_embeddings

    encoder = MockRadDino()

    # Créer le modèle et les composants
    model, criterion, optimizer, heatmap_generator = create_model_and_training_components(
        encoder=encoder,
        device=device,
        learning_rate_head=1e-3,
        learning_rate_backbone=1e-5,
        lambda_cls=1.0,
        lambda_loc=2.0,
        use_focal_loss=True
    )

    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    print(f"Paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ENTRAINEMENT REEL (Classification + Localisation) sur LIDC + NIH
    # --------------------------------------------------------------------
    # Dataset + split train/val
    # --------------------------------------------------------------------
    full_ds = LungMultiSourceDataset(
        lidc_dir=LIDC_DIR,
        nih_dir=NIH_DIR,
        localization_csv=LOCALIZATION_CSV,
        classification_csv=CLASSIFICATION_CSV,
        image_size=224,
    )

    val_ratio = 0.1  # 10% validation
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    # Split déterministe
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    print(f"Dataset total: {n_total} | train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )

    # Scheduler et autres hyperparams
    num_epochs = 30
    use_amp = torch.cuda.is_available()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --------------------------------------------------------------------
    # Logs (max d'infos)
    # --------------------------------------------------------------------
    log_every = 10            # log toutes les N itérations (batches)
    grad_clip_norm = 1.0      # 0.0 pour désactiver
    track_grad_norm = True
    track_cuda_mem = torch.cuda.is_available()

    def _now() -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _cuda_mem_str() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return f"cuda_mem(MB): alloc={alloc:.0f} reserved={reserved:.0f} max_alloc={max_alloc:.0f}"

    def _grad_norm(model_: nn.Module) -> float:
        total = 0.0
        for p in model_.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total += float(param_norm.item()) ** 2
        return float(total ** 0.5)

    def _compute_subset_losses(
        cls_logits: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        cls_targets: torch.Tensor,
        target_heatmaps: torch.Tensor,
        mask: torch.Tensor,
        criterion_: MultiTaskLoss,
    ) -> Dict[str, float]:
        """Recalcule les composantes de loss (cls/loc/total) sur un sous-ensemble du batch."""
        if mask.sum() == 0:
            return {"total": 0.0, "classification": 0.0, "localization": 0.0, "n": 0}

        logits_m = cls_logits[mask]
        y_m = cls_targets[mask]
        pred_m = pred_heatmaps[mask]
        tgt_m = target_heatmaps[mask]

        # cls
        if criterion_.use_focal_loss:
            loss_cls = criterion_.focal_loss(logits_m, y_m)
        else:
            loss_cls = criterion_.cls_criterion(logits_m, y_m)

        # loc (positifs uniquement)
        has_nodule = y_m.squeeze(-1) > 0.5
        if has_nodule.sum() > 0:
            loss_loc = criterion_.loc_criterion(pred_m[has_nodule], tgt_m[has_nodule])
        else:
            loss_loc = torch.tensor(0.0, device=logits_m.device)

        total = criterion_.lambda_cls * loss_cls + criterion_.lambda_loc * loss_loc
        return {
            "total": float(total.item()),
            "classification": float(loss_cls.item()),
            "localization": float(loss_loc.item()),
            "n": int(mask.sum().item()),
        }

    def train_one_epoch(epoch: int) -> dict:
        model.train()
        t0 = time.time()

        total_loss = 0.0
        total_cls = 0.0
        total_loc = 0.0
        n_batches = 0
        n_samples = 0

        # breakdown
        lidc_tot = lidc_cls = lidc_loc = 0.0
        nih_tot = nih_cls = nih_loc = 0.0
        n_lidc = 0
        n_nih = 0

        n_pos = 0
        n_neg = 0

        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        for step, (images, labels, coords_batch, meta) in enumerate(train_loader, start=1):
            step_t0 = time.time()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()  # (B,1)

            # Comptage pos/neg
            with torch.no_grad():
                n_pos += int((labels.squeeze(-1) > 0.5).sum().item())
                n_neg += int((labels.squeeze(-1) <= 0.5).sum().item())

            # Heatmaps GT
            target_heatmaps = heatmap_generator.batch_generate(coords_batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                cls_logits, pred_heatmaps = model(images)
                loss, loss_dict = criterion(cls_logits, pred_heatmaps, labels, target_heatmaps)

            scaler.scale(loss).backward()

            # grad clip + grad norm
            gn = None
            if grad_clip_norm and grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            if track_grad_norm:
                scaler.unscale_(optimizer)
                gn = _grad_norm(model)

            scaler.step(optimizer)
            scaler.update()

            bsz = images.size(0)
            n_samples += bsz

            # Totaux
            total_loss += float(loss.item())
            total_cls += float(loss_dict["classification"])
            total_loc += float(loss_dict["localization"])
            n_batches += 1

            # breakdown par dataset (LIDC vs NIH)
            with torch.no_grad():
                ds_list = meta.get("dataset", [])
                if isinstance(ds_list, list) and len(ds_list) == bsz:
                    ds_tensor = torch.tensor([1 if d == "lidc" else 0 for d in ds_list], device=device)
                    lidc_mask = ds_tensor == 1
                    nih_mask = ds_tensor == 0

                    lidc_losses = _compute_subset_losses(cls_logits, pred_heatmaps, labels, target_heatmaps, lidc_mask, criterion)
                    nih_losses = _compute_subset_losses(cls_logits, pred_heatmaps, labels, target_heatmaps, nih_mask, criterion)

                    lidc_tot += lidc_losses["total"]
                    lidc_cls += lidc_losses["classification"]
                    lidc_loc += lidc_losses["localization"]
                    n_lidc += lidc_losses["n"]

                    nih_tot += nih_losses["total"]
                    nih_cls += nih_losses["classification"]
                    nih_loc += nih_losses["localization"]
                    n_nih += nih_losses["n"]

            # Logs batch
            if step == 1 or step % log_every == 0:
                lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
                mem = _cuda_mem_str() if track_cuda_mem else ""
                dt = time.time() - step_t0

                # infos positives pour loc
                has_nodule = (labels.squeeze(-1) > 0.5)
                b_pos = int(has_nodule.sum().item())
                b_neg = int((~has_nodule).sum().item())

                msg = (
                    f"[{_now()}] TRAIN e{epoch:03d} step {step:05d}/{len(train_loader)} | "
                    f"bs={bsz} | pos={b_pos} neg={b_neg} | "
                    f"loss={loss.item():.4f} cls={loss_dict['classification']:.4f} loc={loss_dict['localization']:.4f} | "
                    f"lr={lrs}"
                )
                if gn is not None:
                    msg += f" | grad_norm={gn:.2f}"
                if mem:
                    msg += f" | {mem}"
                msg += f" | step_time={dt:.2f}s"
                print(msg)

        epoch_time = time.time() - t0

        # Moyennes globales
        out = {
            "loss": total_loss / max(1, n_batches),
            "loss_cls": total_cls / max(1, n_batches),
            "loss_loc": total_loc / max(1, n_batches),
            "epoch_time_s": epoch_time,
            "n_samples": n_samples,
            "pos": n_pos,
            "neg": n_neg,
        }

        # breakdown moyenné par sample (plus parlant que par batch)
        if n_lidc > 0:
            out.update({
                "lidc_loss": lidc_tot / max(1, n_lidc),
                "lidc_cls": lidc_cls / max(1, n_lidc),
                "lidc_loc": lidc_loc / max(1, n_lidc),
                "n_lidc": n_lidc,
            })
        else:
            out.update({"lidc_loss": 0.0, "lidc_cls": 0.0, "lidc_loc": 0.0, "n_lidc": 0})

        if n_nih > 0:
            out.update({
                "nih_loss": nih_tot / max(1, n_nih),
                "nih_cls": nih_cls / max(1, n_nih),
                "nih_loc": nih_loc / max(1, n_nih),
                "n_nih": n_nih,
            })
        else:
            out.update({"nih_loss": 0.0, "nih_cls": 0.0, "nih_loc": 0.0, "n_nih": 0})

        return out

    @torch.no_grad()
    def validate_one_epoch(epoch: int) -> dict:
        model.eval()
        t0 = time.time()

        total_loss = 0.0
        total_cls = 0.0
        total_loc = 0.0
        n_batches = 0
        n_samples = 0

        # breakdown
        lidc_tot = lidc_cls = lidc_loc = 0.0
        nih_tot = nih_cls = nih_loc = 0.0
        n_lidc = 0
        n_nih = 0

        n_pos = 0
        n_neg = 0

        for step, (images, labels, coords_batch, meta) in enumerate(val_loader, start=1):
            step_t0 = time.time()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            target_heatmaps = heatmap_generator.batch_generate(coords_batch, device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                cls_logits, pred_heatmaps = model(images)
                loss, loss_dict = criterion(cls_logits, pred_heatmaps, labels, target_heatmaps)

            bsz = images.size(0)
            n_samples += bsz

            # pos/neg
            n_pos += int((labels.squeeze(-1) > 0.5).sum().item())
            n_neg += int((labels.squeeze(-1) <= 0.5).sum().item())

            total_loss += float(loss.item())
            total_cls += float(loss_dict["classification"])
            total_loc += float(loss_dict["localization"])
            n_batches += 1

            # breakdown per dataset
            ds_list = meta.get("dataset", [])
            if isinstance(ds_list, list) and len(ds_list) == bsz:
                ds_tensor = torch.tensor([1 if d == "lidc" else 0 for d in ds_list], device=device)
                lidc_mask = ds_tensor == 1
                nih_mask = ds_tensor == 0

                lidc_losses = _compute_subset_losses(cls_logits, pred_heatmaps, labels, target_heatmaps, lidc_mask, criterion)
                nih_losses = _compute_subset_losses(cls_logits, pred_heatmaps, labels, target_heatmaps, nih_mask, criterion)

                lidc_tot += lidc_losses["total"]
                lidc_cls += lidc_losses["classification"]
                lidc_loc += lidc_losses["localization"]
                n_lidc += lidc_losses["n"]

                nih_tot += nih_losses["total"]
                nih_cls += nih_losses["classification"]
                nih_loc += nih_losses["localization"]
                n_nih += nih_losses["n"]

            # Logs val batch
            if step == 1 or step % max(1, (log_every * 2)) == 0:
                dt = time.time() - step_t0
                has_nodule = (labels.squeeze(-1) > 0.5)
                b_pos = int(has_nodule.sum().item())
                b_neg = int((~has_nodule).sum().item())

                print(
                    f"[{_now()}] VAL   e{epoch:03d} step {step:05d}/{len(val_loader)} | "
                    f"bs={bsz} | pos={b_pos} neg={b_neg} | "
                    f"loss={loss.item():.4f} cls={loss_dict['classification']:.4f} loc={loss_dict['localization']:.4f} | "
                    f"step_time={dt:.2f}s"
                )

        epoch_time = time.time() - t0

        out = {
            "val_loss": total_loss / max(1, n_batches),
            "val_loss_cls": total_cls / max(1, n_batches),
            "val_loss_loc": total_loc / max(1, n_batches),
            "val_epoch_time_s": epoch_time,
            "val_n_samples": n_samples,
            "val_pos": n_pos,
            "val_neg": n_neg,
        }

        if n_lidc > 0:
            out.update({
                "val_lidc_loss": lidc_tot / max(1, n_lidc),
                "val_lidc_cls": lidc_cls / max(1, n_lidc),
                "val_lidc_loc": lidc_loc / max(1, n_lidc),
                "val_n_lidc": n_lidc,
            })
        else:
            out.update({"val_lidc_loss": 0.0, "val_lidc_cls": 0.0, "val_lidc_loc": 0.0, "val_n_lidc": 0})

        if n_nih > 0:
            out.update({
                "val_nih_loss": nih_tot / max(1, n_nih),
                "val_nih_cls": nih_cls / max(1, n_nih),
                "val_nih_loc": nih_loc / max(1, n_nih),
                "val_n_nih": n_nih,
            })
        else:
            out.update({"val_nih_loss": 0.0, "val_nih_cls": 0.0, "val_nih_loc": 0.0, "val_n_nih": 0})

        return out

    print("===== Début entraînement =====")
    print("Epochs:", num_epochs, "| batch_size:", batch_size)
    # --------------------------------------------------------------------
    # Early stopping fort + best checkpoint (sur val_loss)
    # --------------------------------------------------------------------
    best_val = float("inf")
    patience = 4
    bad_epochs = 0
    best_ckpt_path = "best_checkpoint.pt"

    for epoch in range(1, num_epochs + 1):
        metrics = train_one_epoch(epoch)
        scheduler.step()
        val_metrics = validate_one_epoch(epoch)

        # Best checkpoint + early stopping
        cur_val = val_metrics["val_loss"]
        if cur_val < best_val:
            best_val = cur_val
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_val": best_val,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                best_ckpt_path,
            )
            print(f"Best val improved -> {best_val:.4f} | saved: {best_ckpt_path}")
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0:
            lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
            print(
                f"\n[{_now()}] EPOCH {epoch:03d}/{num_epochs} SUMMARY"\
                f"\n  TRAIN: loss={metrics['loss']:.4f} cls={metrics['loss_cls']:.4f} loc={metrics['loss_loc']:.4f}"\
                f" | pos={metrics.get('pos', 0)} neg={metrics.get('neg', 0)}"\
                f" | lidc(n={metrics.get('n_lidc',0)}): loss={metrics.get('lidc_loss',0):.4f} cls={metrics.get('lidc_cls',0):.4f} loc={metrics.get('lidc_loc',0):.4f}"\
                f" | nih(n={metrics.get('n_nih',0)}): loss={metrics.get('nih_loss',0):.4f} cls={metrics.get('nih_cls',0):.4f} loc={metrics.get('nih_loc',0):.4f}"\
                f"\n  VAL:   loss={val_metrics['val_loss']:.4f} cls={val_metrics['val_loss_cls']:.4f} loc={val_metrics['val_loss_loc']:.4f}"\
                f" | pos={val_metrics.get('val_pos', 0)} neg={val_metrics.get('val_neg', 0)}"\
                f" | lidc(n={val_metrics.get('val_n_lidc',0)}): loss={val_metrics.get('val_lidc_loss',0):.4f} cls={val_metrics.get('val_lidc_cls',0):.4f} loc={val_metrics.get('val_lidc_loc',0):.4f}"\
                f" | nih(n={val_metrics.get('val_n_nih',0)}): loss={val_metrics.get('val_nih_loss',0):.4f} cls={val_metrics.get('val_nih_cls',0):.4f} loc={val_metrics.get('val_nih_loc',0):.4f}"\
                f"\n  LR: {lrs} | bad_epochs={bad_epochs}/{patience}"\
                f"\n  Time: train={metrics.get('epoch_time_s',0):.1f}s val={val_metrics.get('val_epoch_time_s',0):.1f}s | {_cuda_mem_str()}\n"
            )

        if bad_epochs >= patience:
            print(f"\nEarly stopping déclenché à l'epoch {epoch} (patience={patience}). Best val={best_val:.4f}.")
            break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                f"checkpoint_epoch_{epoch:03d}.pt",
            )

# if __name__ == "__main__":
#     # Décommente si tu veux tester le chargement local rapidement.
#     # _debug_local_loading_example()
#     pass