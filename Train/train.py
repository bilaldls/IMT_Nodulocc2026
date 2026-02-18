"""
Entraînement multi-task sur features pré-extraites.
Phase 1 : heads seulement (pas d'encoder en mémoire)
Phase 2 : fine-tuning end-to-end (encoder chargé à ce moment)
"""
from rad_dino import RadDino
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import gc

from FastV2 import (
    ChestXRayMultiTask,
    MultiTaskLoss,
    GaussianHeatmapGenerator,
    FastTrainer,
    CachedFeatureDataset,
    cached_collate_fn,
    create_optimized_dataloaders,
)

from config import config

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        "batch_size": 8,
        "num_workers": 2,
        "image_size": 224,
        "phase1_epochs": 10,
        "phase2_epochs": 3,
        "lr_head": 1e-3,
        "lr_backbone": 1e-5,
        "data_fraction": 1.0,
        "cache_dir": "feature_cache",
        "output_dir": "checkpoints_multitask_fast_V3",
    }

    frac_tag = f"_frac{config['data_fraction']:.2f}" if config["data_fraction"] < 1.0 else ""
    cache_subdir = os.path.join(config["cache_dir"], f"features{frac_tag}")

    # ================================================================
    # PHASE 1 : Heads seulement — PAS D'ENCODER EN MÉMOIRE
    # ================================================================

    # Encoder placeholder (jamais utilisé en Phase 1)
    class DummyEncoder(nn.Module):
        def forward(self, x):
            raise RuntimeError("Encoder non chargé — Phase 1 only")

    model = ChestXRayMultiTask(
        encoder=DummyEncoder(),
        embed_dim=768,
        num_patches=37*37,
        output_size=config["image_size"],
    ).to(device)
    model.freeze_encoder()

    # Dataloaders features (memmap → quasi-zéro RAM)
    train_feat_ds = CachedFeatureDataset(cache_subdir, "train")
    val_feat_ds = CachedFeatureDataset(cache_subdir, "val")

    feat_bs = config["batch_size"] * 4
    train_feat_loader = DataLoader(
        train_feat_ds, batch_size=feat_bs, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=cached_collate_fn,
    )
    val_feat_loader = DataLoader(
        val_feat_ds, batch_size=feat_bs, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=cached_collate_fn,
    )

    criterion = MultiTaskLoss()
    heatmap_gen = GaussianHeatmapGenerator(image_size=config["image_size"])

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
        train_feat_loader, val_feat_loader,
        optimizer_p1, scheduler_p1,
        num_epochs=config["phase1_epochs"],
    )

    # Libérer features
    del train_feat_ds, val_feat_ds, train_feat_loader, val_feat_loader
    torch.cuda.empty_cache()
    gc.collect()
    trainer.save_history()

'''
    # ================================================================
    # PHASE 2 : Charger le VRAI encoder maintenant
    # ================================================================

    # Sauvegarder les poids des heads entraînées
    head_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("encoder.")
    }

    # Reconstruire avec le vrai encoder
    del model
    torch.cuda.empty_cache()

    encoder = RadDino()
    model = ChestXRayMultiTask(
        encoder=encoder, embed_dim=768, num_patches=37*37,
        output_size=config["image_size"],
    ).to(device)

    # Restaurer les heads entraînées en Phase 1
    model.load_state_dict(head_state, strict=False)

    model.unfreeze_encoder(layers_to_unfreeze=["blocks.10", "blocks.11"])

    # Dataloaders images pour Phase 2
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
        data_fraction=config["data_fraction"],
    )

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

    trainer.model = model  # mettre à jour le trainer avec le vrai modèle
    trainer.train_phase2(
        train_loader, val_loader,
        optimizer_p2, scheduler_p2,
        num_epochs=config["phase2_epochs"],
    )

    # Test final
    test_metrics = trainer._validate_e2e(test_loader)
    print(f"\nTest Loss: {test_metrics['val_loss']:.4f}")
   
'''