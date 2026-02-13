"""
Script d'évaluation détaillée du modèle multi-task.
Calcule des métriques complètes pour classification et localisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import os
import json
import argparse
from tqdm import tqdm

from MultiTask_evolved import (
    ChestXRayMultiTask,
    create_optimized_dataloaders,
    GaussianHeatmapGenerator,
    RadDino
)


class ModelEvaluator:
    """Évaluateur complet pour le modèle multi-task."""
    
    def __init__(
        self,
        model: ChestXRayMultiTask,
        device: torch.device,
        heatmap_generator: GaussianHeatmapGenerator,
    ):
        self.model = model
        self.device = device
        self.heatmap_generator = heatmap_generator
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Évaluation complète sur un dataloader."""
        
        # Collecte des prédictions et labels
        all_cls_preds = []
        all_cls_probs = []
        all_cls_labels = []
        
        all_heatmaps_pred = []
        all_heatmaps_gt = []
        
        # Métriques par dataset
        lidc_preds = []
        lidc_labels = []
        nih_preds = []
        nih_labels = []
        
        # Métriques de localisation
        localization_errors = []
        
        print("Évaluation en cours...")
        for images, labels, coords_batch, meta in tqdm(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            cls_logits, pred_heatmaps, _ = self.model(images)
            cls_probs = torch.sigmoid(cls_logits)
            cls_preds = (cls_probs > 0.5).float()
            
            # Heatmaps GT
            target_heatmaps = self.heatmap_generator.batch_generate(coords_batch, self.device)
            
            # Stockage
            all_cls_preds.extend(cls_preds.cpu().numpy().flatten())
            all_cls_probs.extend(cls_probs.cpu().numpy().flatten())
            all_cls_labels.extend(labels.cpu().numpy().flatten())
            
            all_heatmaps_pred.append(pred_heatmaps.cpu())
            all_heatmaps_gt.append(target_heatmaps.cpu())
            
            # Par dataset
            for i, ds in enumerate(meta['dataset']):
                pred = cls_preds[i].item()
                label = labels[i].item()
                
                if ds == 'lidc':
                    lidc_preds.append(pred)
                    lidc_labels.append(label)
                else:
                    nih_preds.append(pred)
                    nih_labels.append(label)
            
            # Erreur de localisation (sur positifs uniquement)
            for i, coords in enumerate(coords_batch):
                if len(coords) > 0 and cls_preds[i].item() > 0.5:
                    error = self._compute_localization_error(
                        pred_heatmaps[i],
                        coords
                    )
                    localization_errors.append(error)
        
        # Conversion en arrays
        all_cls_preds = np.array(all_cls_preds)
        all_cls_probs = np.array(all_cls_probs)
        all_cls_labels = np.array(all_cls_labels)
        
        # Calcul des métriques
        metrics = self._compute_metrics(
            all_cls_preds,
            all_cls_probs,
            all_cls_labels,
            lidc_preds,
            lidc_labels,
            nih_preds,
            nih_labels,
            localization_errors
        )
        
        return metrics
    
    def _compute_localization_error(
        self,
        pred_heatmap: torch.Tensor,
        gt_coords: List[Tuple[float, float]]
    ) -> float:
        """Calcule l'erreur de localisation (distance euclidienne en pixels)."""
        # Trouve le maximum de la heatmap prédite
        heatmap = pred_heatmap.squeeze().cpu().numpy()
        H, W = heatmap.shape
        
        pred_y, pred_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        pred_x_norm = pred_x / W
        pred_y_norm = pred_y / H
        
        # Distance au nodule GT le plus proche
        min_dist = float('inf')
        for gt_x, gt_y in gt_coords:
            dist = np.sqrt((pred_x_norm - gt_x)**2 + (pred_y_norm - gt_y)**2)
            min_dist = min(min_dist, dist)
        
        # Conversion en pixels
        return min_dist * max(H, W)
    
    def _compute_metrics(
        self,
        preds: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
        lidc_preds: List,
        lidc_labels: List,
        nih_preds: List,
        nih_labels: List,
        loc_errors: List[float]
    ) -> Dict:
        """Calcule toutes les métriques."""
        
        metrics = {}
        
        # Métriques globales de classification
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, zero_division=0)
        metrics['recall'] = recall_score(labels, preds, zero_division=0)
        metrics['f1'] = f1_score(labels, preds, zero_division=0)
        
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probs)
        except:
            metrics['auc_roc'] = 0.0
        
        # Matrice de confusion
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Spécificité et sensibilité
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Métriques par dataset
        if len(lidc_preds) > 0:
            metrics['lidc_accuracy'] = accuracy_score(lidc_labels, lidc_preds)
            metrics['lidc_f1'] = f1_score(lidc_labels, lidc_preds, zero_division=0)
            metrics['lidc_n_samples'] = len(lidc_preds)
        
        if len(nih_preds) > 0:
            metrics['nih_accuracy'] = accuracy_score(nih_labels, nih_preds)
            metrics['nih_f1'] = f1_score(nih_labels, nih_preds, zero_division=0)
            metrics['nih_n_samples'] = len(nih_preds)
        
        # Métriques de localisation
        if len(loc_errors) > 0:
            metrics['localization_mean_error_px'] = float(np.mean(loc_errors))
            metrics['localization_median_error_px'] = float(np.median(loc_errors))
            metrics['localization_std_error_px'] = float(np.std(loc_errors))
            metrics['localization_n_samples'] = len(loc_errors)
            
            # Pourcentage avec erreur < seuil
            for threshold in [5, 10, 20, 30]:
                pct = (np.array(loc_errors) < threshold).mean() * 100
                metrics[f'localization_acc_{threshold}px'] = float(pct)
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Affiche les métriques de façon lisible."""
        
        print("\n" + "="*80)
        print("MÉTRIQUES DE CLASSIFICATION")
        print("="*80)
        
        print(f"\nMétriques Globales:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        
        print(f"\nMatrice de Confusion:")
        print(f"  TN: {metrics['true_negatives']:<6} FP: {metrics['false_positives']}")
        print(f"  FN: {metrics['false_negatives']:<6} TP: {metrics['true_positives']}")
        
        if 'lidc_accuracy' in metrics:
            print(f"\nMétriques LIDC (n={metrics['lidc_n_samples']}):")
            print(f"  Accuracy: {metrics['lidc_accuracy']:.4f}")
            print(f"  F1-Score: {metrics['lidc_f1']:.4f}")
        
        if 'nih_accuracy' in metrics:
            print(f"\nMétriques NIH (n={metrics['nih_n_samples']}):")
            print(f"  Accuracy: {metrics['nih_accuracy']:.4f}")
            print(f"  F1-Score: {metrics['nih_f1']:.4f}")
        
        if 'localization_mean_error_px' in metrics:
            print("\n" + "="*80)
            print("MÉTRIQUES DE LOCALISATION")
            print("="*80)
            
            print(f"\nErreur de Distance (n={metrics['localization_n_samples']}):")
            print(f"  Mean:   {metrics['localization_mean_error_px']:.2f} px")
            print(f"  Median: {metrics['localization_median_error_px']:.2f} px")
            print(f"  Std:    {metrics['localization_std_error_px']:.2f} px")
            
            print(f"\nAccuracy par seuil:")
            for threshold in [5, 10, 20, 30]:
                key = f'localization_acc_{threshold}px'
                if key in metrics:
                    print(f"  < {threshold:2d}px: {metrics[key]:.2f}%")
        
        print("\n" + "="*80)


def load_model(checkpoint_path: str, device: torch.device) -> ChestXRayMultiTask:
    """Charge un modèle depuis un checkpoint."""
    
    print(f"Chargement du modèle depuis: {checkpoint_path}")
    
    # Encoder
    try:
        encoder = RadDino()
    except:
        print("⚠ RadDino non disponible, utilisation d'un mock")
        class MockRadDino(nn.Module):
            def extract_features(self, x):
                B = x.size(0)
                cls_embeddings = torch.randn(B, 768, device=x.device)
                patch_embeddings = torch.randn(B, 196, 768, device=x.device)
                return cls_embeddings, patch_embeddings
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
    
    # Chargement des poids
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Modèle chargé (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle multi-task")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Chemin vers le checkpoint du modèle'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Split à évaluer'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Taille des batchs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Fichier de sortie pour les résultats'
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Chargement du modèle
    model = load_model(args.checkpoint, device)
    
    # Dataloaders
    print("Création des dataloaders...")
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
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
    
    print(f"Évaluation sur le split: {args.split}")
    print(f"Nombre de batchs: {len(dataloader)}\n")
    
    # Heatmap generator
    heatmap_generator = GaussianHeatmapGenerator(image_size=224, sigma=10.0)
    
    # Évaluateur
    evaluator = ModelEvaluator(model, device, heatmap_generator)
    
    # Évaluation
    metrics = evaluator.evaluate(dataloader)
    
    # Affichage
    evaluator.print_metrics(metrics)
    
    # Sauvegarde
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {args.output}\n")


if __name__ == "__main__":
    main()
