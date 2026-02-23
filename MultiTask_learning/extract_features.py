"""
Extraction features RadDino → disque (memmap float16, .bin natif)
À lancer UNE SEULE FOIS avant l'entraînement.

NIH  : cls_token seulement (pas de localisation → ~77MB train)
LIDC : cls_token + spatial + heatmap GT + coords (localisation complète)

Arborescence produite :
  feature_cache/features/
  ├── train_cls.bin          NIH train  cls_token  (N, 768)
  ├── train_labels.bin       NIH train  labels     (N,)
  ├── train_meta.json
  ├── val_cls.bin
  ├── val_labels.bin
  ├── val_meta.json
  ├── test_cls.bin
  ├── test_labels.bin
  ├── test_meta.json
  ├── lidc_train_cls.bin     LIDC train cls_token  (N_aug, 768)
  ├── lidc_train_spatial.bin LIDC train patches    (N_aug, 1369, 768)
  ├── lidc_train_heatmaps.bin LIDC train GT heatmap (N_aug, 14, 14)
  ├── lidc_train_labels.bin  LIDC train labels     (N_aug,)
  ├── lidc_train_coords.json LIDC train coords     liste de listes
  ├── lidc_train_meta.json
  ├── lidc_val_cls.bin
  ├── lidc_val_spatial.bin
  ├── lidc_val_heatmaps.bin
  ├── lidc_val_labels.bin
  ├── lidc_val_coords.json
  └── lidc_val_meta.json
"""

import torch
import numpy as np
import os
import json
import gc
from tqdm import tqdm
import pandas as pd

from rad_dino import RadDino
from FastV2 import (
    create_optimized_dataloaders,
    Augmentor,
    GaussianHeatmapGenerator,
    LIDC_DIR,
    LOCALIZATION_CSV,
    _load_png, 
    _extract_coords_from_row
)


# ============================================================================
# UTILITAIRES MEMMAP
# ============================================================================

def _create_memmap(path: str, dtype: str, shape: tuple) -> np.memmap:
    """Crée un fichier memmap en écriture."""
    return np.memmap(path, dtype=dtype, mode='w+', shape=shape)


def _save_meta(cache_dir: str, split_name: str, meta: dict) -> None:
    path = os.path.join(cache_dir, f"{split_name}_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta → {path}")


# ============================================================================
# EXTRACTION NIH  (cls_token uniquement — pas de localisation sur NIH)
# ============================================================================

def extract_nih_features(
    encoder,
    dataloader,
    device: torch.device,
    cache_dir: str,
    split_name: str,
) -> None:
    """
    Extrait cls_token + label pour chaque image NIH.
    spatial NON stocké : NIH n'a pas d'annotations nodules,
    la tête de localisation n'est pas entraînée sur NIH.

    Taille : 50k × 768 × 2 bytes ≈ 77 MB pour train.
    """
    N     = len(dataloader.dataset)
    D_cls = 768

    cls_path    = os.path.join(cache_dir, f"{split_name}_cls.bin")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.bin")

    cls_mm    = _create_memmap(cls_path,    'float16', (N, D_cls))
    labels_mm = _create_memmap(labels_path, 'float32', (N,))

    idx = 0
    encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  NIH {split_name}"):
            # Le dataloader NIH retourne (images, labels)
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images = batch["image"]
                labels = batch["label"]

            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == 'cuda')):
                cls_tok, _ = encoder(images)   # on ignore spatial

            B = cls_tok.shape[0]
            cls_mm[idx:idx+B] = (
                cls_tok.float().cpu().numpy().astype('float16')
            )
            labels_mm[idx:idx+B] = (
                labels.numpy().astype('float32')
                if isinstance(labels, torch.Tensor)
                else np.array(labels, dtype='float32')
            )
            idx += B

    # Flush mémoire → disque
    cls_mm.flush()
    labels_mm.flush()
    del cls_mm, labels_mm

    meta = {
        "N":      idx,
        "D_cls":  D_cls,
        "source": "NIH",
        "split":  split_name,
        "has_spatial":  False,
        "has_heatmap":  False,
    }
    _save_meta(cache_dir, split_name, meta)
    print(f"  ✓ NIH {split_name} : {idx} samples sauvegardés")


# ============================================================================
# EXTRACTION LIDC  (cls_token + spatial + heatmap GT + coords)
# ============================================================================

def extract_lidc_features(
    encoder,
    device: torch.device,
    cache_dir: str,
    lidc_dir: str,
    localization_csv: str,
    image_size: int  = 224,
    repeat_factor: int = 15,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Extrait les features LIDC avec augmentation pour le split train.
    Chaque image originale est augmentée repeat_factor fois (train seulement).
    Le split val utilise les images originales sans augmentation.

    Taille estimée train (260 × 15 = 3900 samples) :
      cls     : 3900 × 768     × 2 bytes ≈   6 MB
      spatial : 3900 × 1369    × 768 × 2 ≈ 8.2 GB
      heatmap : 3900 × 14 × 14 × 2 bytes ≈   2 MB
    """
    D_cls   = 768
    N_patch = 37 * 37   # = 1369  (ViT-B/14 sur image 518px → projeté 37×37)
    D_patch = 768
    H_hm    = image_size // 16   # taille heatmap GT (14 pour 224px)
    W_hm    = image_size // 16

    augmentor   = Augmentor(image_size=image_size)
    heatmap_gen = GaussianHeatmapGenerator(image_size=image_size)

    # ── Lecture CSV et split par patient ────────────────────────────────
    df = pd.read_csv(localization_csv)

    # Détecter la colonne patient_id (adapter selon ton CSV)
    if "patient_id" in df.columns:
        id_col = "patient_id"
    elif "patientid" in df.columns:
        id_col = "patientid"
    else:
        # Fallback : utiliser la première colonne
        id_col = df.columns[0]
        print(f"  ⚠ Colonne patient_id non trouvée, utilisation de '{id_col}'")

    patients = df[id_col].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patients))
    patients = patients[perm]

    n_val = max(1, int(len(patients) * val_fraction))
    val_patients   = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    print(f"  Split LIDC : {len(train_patients)} patients train, "
          f"{len(val_patients)} patients val")

    splits_def = {
        "train": (train_patients, repeat_factor),
        "val":   (val_patients,   1),            # pas d'augmentation en val
    }

    for split_name, (patient_set, reps) in splits_def.items():

        split_df = df[df[id_col].isin(patient_set)].reset_index(drop=True)
        n_images = len(split_df)
        N_total  = n_images * reps

        print(f"\n  LIDC {split_name} : {n_images} images × {reps} = {N_total} samples")

        # ── Créer les memmaps ────────────────────────────────────────────
        prefix = os.path.join(cache_dir, f"lidc_{split_name}")

        cls_mm = _create_memmap(f"{prefix}_cls.bin",
                                'float16', (N_total, D_cls))
        spa_mm = _create_memmap(f"{prefix}_spatial.bin",
                                'float16', (N_total, N_patch, D_patch))
        hm_mm  = _create_memmap(f"{prefix}_heatmaps.bin",
                                'float16', (N_total, H_hm, W_hm))
        lab_mm = _create_memmap(f"{prefix}_labels.bin",
                                'float32', (N_total,))

        coords_list = []   # sauvegardé en JSON (léger)
        idx = 0

        encoder.eval()

        for row_idx, row in tqdm(split_df.iterrows(),
                                  total=n_images,
                                  desc=f"  LIDC {split_name}"):

            # ── Charger image ────────────────────────────────────────────
            filename = (
                row.get("filename")
                or row.get("file_name")
                or row.get("image_path")
                or row.iloc[0]
            )
            img_path = os.path.join(lidc_dir, str(filename))

            if not os.path.exists(img_path):
                print(f"  ⚠ Fichier manquant : {img_path}")
                continue

            image  = _load_png(img_path, image_size)      # Tensor (1, H, W)
            coords = _extract_coords_from_row(row)        # [(x,y), ...]
            label  = 1.0 if len(coords) > 0 else 0.0

            # ── Boucle augmentation ──────────────────────────────────────
            for rep in range(reps):

                if split_name == "train" and rep > 0:
                    # Augmentation aléatoire (différente à chaque rep)
                    aug_img, aug_coords = augmentor(
                        image.clone(), [c for c in coords]
                    )
                    aug_label = 1.0 if len(aug_coords) > 0 else 0.0
                else:
                    # Rep 0 (ou val) : image originale sans augmentation
                    aug_img    = image
                    aug_coords = coords
                    aug_label  = label

                # ── Heatmap GT ──────────────────────────────────────────
                heatmap = heatmap_gen.generate(aug_coords)
                # heatmap : Tensor (1, H_hm, W_hm)

                # ── Encoder ─────────────────────────────────────────────
                inp = aug_img.unsqueeze(0).to(device, non_blocking=True)

                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=(device.type == 'cuda')
                    ):
                        cls_tok, spatial = encoder(inp)

                # Normaliser spatial → (1, N_patch, D_patch)
                if spatial.dim() == 4:
                    # (1, C, H, W) → (1, H*W, C)
                    spatial = spatial.flatten(2).transpose(1, 2)

                # Vérifier cohérence N_patch
                actual_n = spatial.shape[1]
                if actual_n != N_patch:
                    # Cas rare si l'encodeur retourne une grille différente
                    # On redimensionne en interpolation bilinéaire
                    side = int(actual_n ** 0.5)
                    spatial_4d = spatial.transpose(1, 2).reshape(
                        1, D_patch, side, side
                    )
                    spatial_4d = torch.nn.functional.interpolate(
                        spatial_4d, size=(37, 37), mode='bilinear',
                        align_corners=False
                    )
                    spatial = spatial_4d.flatten(2).transpose(1, 2)

                # Vérifier cohérence heatmap
                hm_np = heatmap[0].float().cpu().numpy()
                if hm_np.shape != (H_hm, W_hm):
                    import cv2
                    hm_np = cv2.resize(hm_np, (W_hm, H_hm))

                # ── Écriture memmap ──────────────────────────────────────
                cls_mm[idx] = (
                    cls_tok[0].float().cpu().numpy().astype('float16')
                )
                spa_mm[idx] = (
                    spatial[0].float().cpu().numpy().astype('float16')
                )
                hm_mm[idx]  = hm_np.astype('float16')
                lab_mm[idx] = aug_label
                coords_list.append(aug_coords)
                idx += 1

        # ── Flush → disque ───────────────────────────────────────────────
        cls_mm.flush();  del cls_mm
        spa_mm.flush();  del spa_mm
        hm_mm.flush();   del hm_mm
        lab_mm.flush();  del lab_mm
        gc.collect()

        # ── Métadonnées ──────────────────────────────────────────────────
        meta = {
            "N":          idx,
            "D_cls":      D_cls,
            "N_patches":  N_patch,
            "D_patch":    D_patch,
            "H_heatmap":  H_hm,
            "W_heatmap":  W_hm,
            "source":     "LIDC",
            "split":      split_name,
            "has_spatial": True,
            "has_heatmap": True,
            "repeat_factor": reps,
        }
        _save_meta(cache_dir, f"lidc_{split_name}", meta)

        with open(f"{prefix}_coords.json", "w") as f:
            json.dump(coords_list[:idx], f)

        print(f"  ✓ LIDC {split_name} : {idx} samples sauvegardés")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Configuration (tout ici, pas d'import externe) ──────────────────
    extraction_config = {
        "batch_size":     8,
        "num_workers":    2,
        "image_size":     224,
        "data_fraction":  1.0,
        "cache_dir":      "feature_cache/features",
        "lidc_repeat":    15,
        "val_fraction":   0.2,
        "seed":           42,
    }

    os.makedirs(extraction_config["cache_dir"], exist_ok=True)

    # ── Encoder (frozen, eval) ───────────────────────────────────────────
    encoder = RadDino().to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder RadDino chargé ({sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params)")

    # ── NIH ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXTRACTION NIH")
    print("="*60)

    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=extraction_config["batch_size"],
        num_workers=extraction_config["num_workers"],
        image_size=extraction_config["image_size"],
        data_fraction=extraction_config["data_fraction"],
    )

    for name, loader in [
        ("train", train_loader),
        ("val",   val_loader),
        ("test",  test_loader),
    ]:
        extract_nih_features(
            encoder=encoder,
            dataloader=loader,
            device=device,
            cache_dir=extraction_config["cache_dir"],
            split_name=name,
        )

    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ── LIDC ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXTRACTION LIDC (avec augmentation × repeat_factor)")
    print("="*60)

    extract_lidc_features(
        encoder=encoder,
        device=device,
        cache_dir=extraction_config["cache_dir"],
        lidc_dir=LIDC_DIR,
        localization_csv=LOCALIZATION_CSV,
        image_size=extraction_config["image_size"],
        repeat_factor=extraction_config["lidc_repeat"],
        val_fraction=extraction_config["val_fraction"],
        seed=extraction_config["seed"],
    )

    # ── Nettoyage final ──────────────────────────────────────────────────
    del encoder
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*60)
    print("✓ Extraction terminée.")
    print(f"  Cache → {extraction_config['cache_dir']}")
    print("  Lance feature_extraction.py une seule fois,")
    print("  puis train.py à chaque entraînement.")
    print("="*60)
