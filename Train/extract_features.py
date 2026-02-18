"""
Script d'extraction des features RadDino → disque (memmap float16)
À lancer UNE SEULE FOIS avant l'entraînement.
"""
from rad_dino import RadDino
import torch
import os
import json
import time
import gc

# Réutiliser tes fonctions existantes
from FastV2 import (
    create_optimized_dataloaders,
    extract_and_cache_features,
)
from config import config
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = {
        "batch_size": 8,
        "num_workers": 2,
        "image_size": 224,
        "data_fraction": 1.0,  # 1.0 = 100% des données, 0.5 = 50%, etc.
        "cache_dir": "feature_cache",
    }

    # Dataloaders images
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
        data_fraction=config["data_fraction"],
    )

    # Encoder seul (pas besoin des heads)
    encoder = RadDino()
    encoder = encoder.to(device)
    encoder.eval()

    frac_tag = f"_frac{config['data_fraction']:.2f}" if config["data_fraction"] < 1.0 else ""
    cache_subdir = os.path.join(config["cache_dir"], f"features{frac_tag}")
    os.makedirs(cache_subdir, exist_ok=True)

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"\nExtraction {name}...")
        extract_and_cache_features(
            encoder=encoder,
            dataloader=loader,
            device=device,
            cache_dir=cache_subdir,
            split_name=name,
        )

    # On libère TOUT
    del encoder, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✓ Features extraites. Tu peux lancer train.py maintenant.")
