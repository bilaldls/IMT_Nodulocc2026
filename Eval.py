import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import torch.nn.functional as F


class ModelEvaluator:
    """Évaluateur pour les métriques de classification et localisation."""
    
    def __init__(self, iou_threshold: float = 0.5, distance_threshold: float = 20.0):
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold  # en pixels
        self.reset()
    
    def reset(self):
        """Réinitialise les accumulateurs de métriques."""
        self.all_cls_preds = []
        self.all_cls_labels = []
        self.all_distances = []
        self.detection_tp = 0
        self.detection_fp = 0
        self.detection_fn = 0
    
    def update(
        self,
        cls_logits: torch.Tensor,
        pred_heatmap: torch.Tensor,
        cls_labels: torch.Tensor,
        gt_coords: List[List[Tuple[float, float]]],
        image_size: int = 224
    ):
        """
        Met à jour les métriques avec un batch.
        
        Args:
            cls_logits: (B, 1) - logits de classification
            pred_heatmap: (B, 1, H, W) - heatmap prédite
            cls_labels: (B, 1) - labels binaires
            gt_coords: coordonnées ground truth par image
            image_size: taille de l'image
        """
        # Métriques de classification
        cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
        cls_labels_np = cls_labels.cpu().numpy()
        
        self.all_cls_preds.extend(cls_probs.flatten().tolist())
        self.all_cls_labels.extend(cls_labels_np.flatten().tolist())
        
        # Métriques de localisation
        for i in range(len(gt_coords)):
            if len(gt_coords[i]) == 0:
                continue
            
            # Extraire les pics de la heatmap
            heatmap = pred_heatmap[i, 0].cpu().numpy()
            pred_coords = self._extract_peaks(heatmap, threshold=0.5)
            
            # Convertir gt_coords en pixels
            gt_coords_pixels = [
                (x * image_size, y * image_size) 
                for x, y in gt_coords[i]
            ]
            
            # Matching entre prédictions et ground truth
            matched_gt = set()
            for pred_x, pred_y in pred_coords:
                # Trouver le GT le plus proche
                min_dist = float('inf')
                closest_gt_idx = -1
                
                for j, (gt_x, gt_y) in enumerate(gt_coords_pixels):
                    if j in matched_gt:
                        continue
                    dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_gt_idx = j
                
                # True positive si distance < threshold
                if min_dist < self.distance_threshold:
                    self.detection_tp += 1
                    matched_gt.add(closest_gt_idx)
                    self.all_distances.append(min_dist)
                else:
                    self.detection_fp += 1
            
            # False negatives = GT non matchés
            self.detection_fn += len(gt_coords_pixels) - len(matched_gt)
    
    def _extract_peaks(
        self, 
        heatmap: np.ndarray, 
        threshold: float = 0.5,
        min_distance: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Extrait les pics locaux d'une heatmap.
        
        Args:
            heatmap: (H, W) - heatmap 2D
            threshold: seuil minimal
            min_distance: distance minimale entre pics
        Returns:
            Liste de coordonnées (x, y)
        """
        from scipy.ndimage import maximum_filter
        
        # Normaliser
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Détection de maxima locaux
        local_max = maximum_filter(heatmap, size=min_distance) == heatmap
        peaks = heatmap > threshold
        peaks = peaks & local_max
        
        # Extraire coordonnées
        y_coords, x_coords = np.where(peaks)
        coords = list(zip(x_coords, y_coords))
        
        return coords
    
    def compute_metrics(self) -> dict:
        """Calcule toutes les métriques."""
        metrics = {}
        
        # Métriques de classification
        if len(self.all_cls_preds) > 0:
            metrics['cls_auroc'] = roc_auc_score(self.all_cls_labels, self.all_cls_preds)
            metrics['cls_auprc'] = average_precision_score(self.all_cls_labels, self.all_cls_preds)
            
            # Accuracy à seuil 0.5
            preds_binary = np.array(self.all_cls_preds) > 0.5
            cm = confusion_matrix(self.all_cls_labels, preds_binary)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                metrics['cls_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
                metrics['cls_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['cls_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Métriques de localisation
        if self.detection_tp + self.detection_fp + self.detection_fn > 0:
            metrics['loc_precision'] = self.detection_tp / (self.detection_tp + self.detection_fp) \
                if (self.detection_tp + self.detection_fp) > 0 else 0
            metrics['loc_recall'] = self.detection_tp / (self.detection_tp + self.detection_fn) \
                if (self.detection_tp + self.detection_fn) > 0 else 0
            
            if metrics['loc_precision'] + metrics['loc_recall'] > 0:
                metrics['loc_f1'] = 2 * metrics['loc_precision'] * metrics['loc_recall'] / \
                    (metrics['loc_precision'] + metrics['loc_recall'])
            else:
                metrics['loc_f1'] = 0
        
        # Distance moyenne pour les détections correctes
        if len(self.all_distances) > 0:
            metrics['loc_mean_distance'] = np.mean(self.all_distances)
            metrics['loc_median_distance'] = np.median(self.all_distances)
        
        return metrics


def visualize_predictions(
    image: torch.Tensor,
    pred_heatmap: torch.Tensor,
    gt_heatmap: Optional[torch.Tensor] = None,
    cls_prob: Optional[float] = None,
    gt_label: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualise une image avec sa heatmap prédite et optionnellement la GT.
    
    Args:
        image: (C, H, W) - image d'entrée
        pred_heatmap: (1, H, W) - heatmap prédite
        gt_heatmap: (1, H, W) - heatmap ground truth (optionnel)
        cls_prob: probabilité de classification
        gt_label: label ground truth
        save_path: chemin pour sauvegarder la figure
    """
    num_plots = 3 if gt_heatmap is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Image originale
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    
    title = "Image originale"
    if gt_label is not None:
        title += f"\nGT: {'Positif' if gt_label == 1 else 'Négatif'}"
    if cls_prob is not None:
        title += f"\nPréd: {cls_prob:.3f}"
    axes[0].set_title(title)
    axes[0].axis('off')
    
    # Heatmap prédite
    pred_hm = pred_heatmap[0].cpu().numpy()
    im1 = axes[1].imshow(img_np, alpha=0.6)
    im2 = axes[1].imshow(pred_hm, cmap='hot', alpha=0.4)
    axes[1].set_title("Heatmap prédite")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Heatmap GT si disponible
    if gt_heatmap is not None:
        gt_hm = gt_heatmap[0].cpu().numpy()
        axes[2].imshow(img_np, alpha=0.6)
        im3 = axes[2].imshow(gt_hm, cmap='hot', alpha=0.4)
        axes[2].set_title("Heatmap ground truth")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def validate_epoch(
    model,
    dataloader,
    criterion,
    heatmap_generator,
    device: torch.device
) -> Tuple[dict, ModelEvaluator]:
    """
    Validation complète sur une époque.
    
    Args:
        model: modèle multi-task
        dataloader: dataloader de validation
        criterion: fonction de loss
        heatmap_generator: générateur de heatmaps
        device: dispositif PyTorch
    
    Returns:
        avg_losses: dictionnaire des losses moyennes
        evaluator: évaluateur avec toutes les métriques
    """
    model.eval()
    evaluator = ModelEvaluator()
    
    total_losses = {'total': 0, 'classification': 0, 'localization': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            cls_labels = batch['label'].to(device).unsqueeze(1).float()
            nodule_coords = batch['nodule_coords']
            
            # Génération heatmaps GT
            target_heatmaps = heatmap_generator.batch_generate(nodule_coords, device)
            
            # Forward
            cls_logits, pred_heatmaps = model(images)
            
            # Loss
            _, loss_dict = criterion(cls_logits, pred_heatmaps, cls_labels, target_heatmaps)
            
            for key in total_losses.keys():
                total_losses[key] += loss_dict[key]
            num_batches += 1
            
            # Métriques
            evaluator.update(cls_logits, pred_heatmaps, cls_labels, nodule_coords)
    
    # Moyennes
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses, evaluator


class LearningRateScheduler:
    """Scheduler de learning rate avec warmup et decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Met à jour le learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup linéaire
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    heatmap_generator,
    device: torch.device,
    epoch: int,
    print_freq: int = 50
) -> dict:
    """
    Entraînement sur une époque complète.
    
    Args:
        model: modèle multi-task
        dataloader: dataloader d'entraînement
        criterion: fonction de loss
        optimizer: optimiseur
        heatmap_generator: générateur de heatmaps
        device: dispositif PyTorch
        epoch: numéro d'époque
        print_freq: fréquence d'affichage
    
    Returns:
        avg_losses: dictionnaire des losses moyennes
    """
    model.train()
    
    total_losses = {'total': 0, 'classification': 0, 'localization': 0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        cls_labels = batch['label'].to(device).unsqueeze(1).float()
        nodule_coords = batch['nodule_coords']
        
        # Training step
        from MultiTask import training_step_example
        loss_dict = training_step_example(
            model=model,
            images=images,
            cls_labels=cls_labels.squeeze(1),
            nodule_coords_batch=nodule_coords,
            criterion=criterion,
            optimizer=optimizer,
            heatmap_generator=heatmap_generator,
            device=device
        )
        
        for key in total_losses.keys():
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Affichage
        if (batch_idx + 1) % print_freq == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(Cls: {loss_dict['classification']:.4f}, "
                  f"Loc: {loss_dict['localization']:.4f})")
    
    # Moyennes
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


# ============================================================================
# EXEMPLE COMPLET D'ENTRAÎNEMENT
# ============================================================================

def main_training_loop():
    """Exemple de boucle d'entraînement complète."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50
    
    # Modèle (utiliser votre RadDino réel ici)
    from MultiTask import MockRadDino, create_model_and_training_components
    encoder = MockRadDino()
    
    model, criterion, optimizer, heatmap_generator = create_model_and_training_components(
        encoder=encoder,
        device=device,
        lambda_cls=1.0,
        lambda_loc=2.0,
        use_focal_loss=True
    )
    
    # Scheduler
    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=num_epochs,
        initial_lr=1e-3
    )
    
    # Historique
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auroc': [],
        'val_f1': []
    }
    
    print("Début de l'entraînement...")
    print(f"Phase 1: Backbone gelé")
    
    # Phase 1: Backbone gelé (20 époques)
    for epoch in range(20):
        lr = scheduler.step(epoch)
        print(f"\n=== Epoch {epoch+1}/20 | LR: {lr:.6f} ===")
        
        # Train
        # train_losses = train_one_epoch(...)
        # val_losses, evaluator = validate_epoch(...)
        # metrics = evaluator.compute_metrics()
        
        # Ici: code d'entraînement réel avec vos dataloaders
        
    # Phase 2: Fine-tuning partiel
    print("\n\n=== Phase 2: Fine-tuning des derniers blocs ===")
    model.unfreeze_encoder_partial(num_blocks=2)
    
    # Réduire LR pour stabilité
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    
    for epoch in range(20, num_epochs):
        lr = scheduler.step(epoch)
        print(f"\n=== Epoch {epoch+1}/{num_epochs} | LR: {lr:.6f} ===")
        
        # Train et validation
        # ...
    
    print("\nEntraînement terminé!")
    
    return model, history


if __name__ == "__main__":
    print("Module d'évaluation et visualisation chargé.")
    print("\nFonctions disponibles:")
    print("  - ModelEvaluator: calcul des métriques")
    print("  - visualize_predictions: visualisation des résultats")
    print("  - validate_epoch: validation complète")
    print("  - train_one_epoch: entraînement d'une époque")
    print("  - LearningRateScheduler: gestion du learning rate")