"""
Script d'inférence sur une image unique.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from MultiTask_evolved import ChestXRayMultiTask, _load_png, _resize_tensor_image, RadDino


def load_model(checkpoint_path: str, device: torch.device) -> ChestXRayMultiTask:
    """Charge le modèle."""
    try:
        encoder = RadDino()
    except:
        class MockRadDino(torch.nn.Module):
            def extract_features(self, x):
                B = x.size(0)
                return (torch.randn(B, 768, device=x.device),
                       torch.randn(B, 196, 768, device=x.device))
        encoder = MockRadDino()
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def predict_image(
    model: ChestXRayMultiTask,
    image_path: str,
    device: torch.device,
    threshold: float = 0.5
):
    """Prédiction sur une image."""
    
    # Chargement
    img_tensor = _load_png(image_path)
    img_tensor = _resize_tensor_image(img_tensor, size=224)
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # Prédiction
    cls_logits, heatmap, _ = model(img_batch)
    cls_prob = torch.sigmoid(cls_logits).item()
    
    # Résultats
    is_positive = cls_prob > threshold
    
    # Peak heatmap
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    peak_x_norm = peak_x / heatmap_np.shape[1]
    peak_y_norm = peak_y / heatmap_np.shape[0]
    
    results = {
        'classification': 'POSITIF (nodules détectés)' if is_positive else 'NÉGATIF (pas de nodules)',
        'probability': cls_prob,
        'nodule_location_normalized': (peak_x_norm, peak_y_norm) if is_positive else None,
        'nodule_location_pixels': (peak_x, peak_y) if is_positive else None,
        'heatmap': heatmap_np,
        'image': img_tensor.permute(1, 2, 0).numpy()
    }
    
    return results


def visualize_results(results: dict, output_path: str = None):
    """Visualise les résultats."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    img = results['image']
    img = np.clip(img, 0, 1)
    
    # Image originale
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Image Originale', fontsize=14)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(results['heatmap'], cmap='jet', alpha=0.5)
    axes[1].set_title(f'Heatmap de Localisation\nProb: {results["probability"]:.3f}', fontsize=14)
    axes[1].axis('off')
    
    if results['nodule_location_pixels']:
        x, y = results['nodule_location_pixels']
        axes[1].plot(x, y, 'r+', markersize=25, markeredgewidth=3)
    
    # Overlay
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(results['heatmap'], cmap='jet', alpha=0.4)
    
    if results['nodule_location_pixels']:
        x, y = results['nodule_location_pixels']
        axes[2].plot(x, y, 'r+', markersize=25, markeredgewidth=3)
        circle = patches.Circle((x, y), radius=20, fill=False, 
                               edgecolor='red', linewidth=2)
        axes[2].add_patch(circle)
    
    color = 'red' if 'POSITIF' in results['classification'] else 'green'
    axes[2].set_title(f'Résultat: {results["classification"]}', 
                     fontsize=14, color=color, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualisation sauvegardée: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inférence sur une image")
    parser.add_argument('image', type=str, help='Chemin vers l\'image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint du modèle')
    parser.add_argument('--threshold', type=float, default=0.5, help='Seuil de classification')
    parser.add_argument('--output', type=str, default=None, help='Fichier de sortie (optionnel)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Chargement modèle
    print(f"Chargement du modèle depuis: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("✓ Modèle chargé\n")
    
    # Prédiction
    print(f"Analyse de l'image: {args.image}")
    results = predict_image(model, args.image, device, args.threshold)
    
    # Affichage résultats
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ANALYSE")
    print("="*60)
    print(f"Classification: {results['classification']}")
    print(f"Probabilité:    {results['probability']:.4f}")
    
    if results['nodule_location_normalized']:
        x, y = results['nodule_location_normalized']
        print(f"Localisation:   x={x:.3f}, y={y:.3f} (normalisé)")
        px_x, px_y = results['nodule_location_pixels']
        print(f"                x={px_x}px, y={px_y}px")
    
    print("="*60 + "\n")
    
    # Visualisation
    output_path = args.output if args.output else f"prediction_{args.image.split('/')[-1]}"
    visualize_results(results, output_path)


if __name__ == "__main__":
    main()
