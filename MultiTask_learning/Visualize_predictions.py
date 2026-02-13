"""
Script de visualisation des prédictions du modèle.
Génère des images avec overlays des heatmaps et labels.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
import argparse
from typing import List, Tuple
from tqdm import tqdm

from MultiTask_evolved import (
    ChestXRayMultiTask,
    create_optimized_dataloaders,
    GaussianHeatmapGenerator,
    RadDino
)


class PredictionVisualizer:
    """Visualiseur de prédictions."""
    
    def __init__(
        self,
        model: ChestXRayMultiTask,
        device: torch.device,
        output_dir: str = "visualizations"
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.eval()
    
    @torch.no_grad()
    def visualize_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        coords_batch: List[List[Tuple[float, float]]],
        filenames: List[str],
        num_samples: int = 8
    ):
        """Visualise un batch de prédictions."""
        
        images = images.to(self.device)
        
        # Prédictions
        cls_logits, pred_heatmaps, _ = self.model(images)
        cls_probs = torch.sigmoid(cls_logits)
        
        # Sélection d'exemples
        num_samples = min(num_samples, images.size(0))
        
        for i in range(num_samples):
            self._visualize_single(
                image=images[i],
                label=labels[i].item(),
                coords=coords_batch[i],
                cls_prob=cls_probs[i].item(),
                heatmap=pred_heatmaps[i],
                filename=filenames[i]
            )
    
    def _visualize_single(
        self,
        image: torch.Tensor,
        label: float,
        coords: List[Tuple[float, float]],
        cls_prob: float,
        heatmap: torch.Tensor,
        filename: str
    ):
        """Visualise une seule prédiction."""
        
        # Conversion image
        img_np = image.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Conversion heatmap
        heatmap_np = heatmap.squeeze().cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        # Figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Image originale
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f'Image Originale\nLabel: {"Positif" if label > 0.5 else "Négatif"}')
        axes[0].axis('off')
        
        # Annotations GT
        H, W = img_np.shape[:2]
        for x_norm, y_norm in coords:
            x_px = x_norm * W
            y_px = y_norm * H
            circle = patches.Circle((x_px, y_px), radius=15, 
                                   fill=False, edgecolor='green', linewidth=2)
            axes[0].add_patch(circle)
            axes[0].plot(x_px, y_px, 'g+', markersize=15, markeredgewidth=2)
        
        # Heatmap prédite
        axes[1].imshow(img_np, cmap='gray')
        axes[1].imshow(heatmap_np, cmap='jet', alpha=0.5)
        axes[1].set_title(f'Heatmap Prédite\nProb: {cls_prob:.3f}')
        axes[1].axis('off')
        
        # Peak de la heatmap
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        axes[1].plot(peak_x, peak_y, 'r+', markersize=20, markeredgewidth=3)
        
        # Overlay combiné
        axes[2].imshow(img_np, cmap='gray')
        axes[2].imshow(heatmap_np, cmap='jet', alpha=0.4)
        
        # GT en vert
        for x_norm, y_norm in coords:
            x_px = x_norm * W
            y_px = y_norm * H
            circle = patches.Circle((x_px, y_px), radius=15, 
                                   fill=False, edgecolor='green', linewidth=2)
            axes[2].add_patch(circle)
        
        # Prédiction en rouge
        axes[2].plot(peak_x, peak_y, 'r+', markersize=20, markeredgewidth=3)
        
        pred_label = "Positif" if cls_prob > 0.5 else "Négatif"
        correct = (pred_label == "Positif") == (label > 0.5)
        title_color = 'green' if correct else 'red'
        
        axes[2].set_title(f'Overlay (GT: vert, Pred: rouge)\nPred: {pred_label} | {"✓" if correct else "✗"}',
                         color=title_color, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarde
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(self.output_dir, f"pred_{safe_filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_dataloader(
        self,
        dataloader,
        num_batches: int = 5,
        samples_per_batch: int = 4
    ):
        """Visualise plusieurs batchs."""
        
        print(f"Visualisation de {num_batches} batchs...")
        
        for batch_idx, (images, labels, coords_batch, meta) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            print(f"Batch {batch_idx + 1}/{num_batches}")
            
            self.visualize_batch(
                images=images,
                labels=labels,
                coords_batch=coords_batch,
                filenames=meta['filename'],
                num_samples=samples_per_batch
            )
        
        print(f"\n✓ Visualisations sauvegardées dans: {self.output_dir}")


def load_model(checkpoint_path: str, device: torch.device) -> ChestXRayMultiTask:
    """Charge un modèle depuis un checkpoint."""
    
    # Encoder
    try:
        encoder = RadDino()
    except:
        class MockRadDino(torch.nn.Module):
            def extract_features(self, x):
                B = x.size(0)
                return (torch.randn(B, 768, device=x.device),
                       torch.randn(B, 196, 768, device=x.device))
        encoder = MockRadDino()
    
    # Modèle
    model = ChestXRayMultiTask(
        encoder=encoder,
        embed_dim=768,
        num_patches=14,
        output_size=224,
        shared_hidden=512,
        classification_hidden=[256, 128],
        localization_channels=[512, 256, 128, 64],
        dropout=0.3
    )
    
    # Chargement
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Visualisation des prédictions")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Chemin vers le checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Split à visualiser'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=5,
        help='Nombre de batchs à visualiser'
    )
    parser.add_argument(
        '--samples_per_batch',
        type=int,
        default=4,
        help='Nombre d\'exemples par batch'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations',
        help='Dossier de sortie'
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Modèle
    print(f"Chargement du modèle depuis: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("✓ Modèle chargé\n")
    
    # Dataloaders
    print("Création des dataloaders...")
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=16,
        num_workers=2,
        image_size=224,
        augment_train=False,
    )
    
    # Sélection du split
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    print(f"Split sélectionné: {args.split}\n")
    
    # Visualiseur
    visualizer = PredictionVisualizer(model, device, args.output_dir)
    
    # Visualisation
    visualizer.visualize_dataloader(
        dataloader,
        num_batches=args.num_batches,
        samples_per_batch=args.samples_per_batch
    )


if __name__ == "__main__":
    main()
