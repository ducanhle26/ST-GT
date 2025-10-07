#!/usr/bin/env python3
"""
Main Training Script for Spatio-Temporal Graph Transformer
"""
import os
import sys
import yaml
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processor import (
    TopSectionsDataProcessor, 
    create_multi_section_sequences,
    create_adjacency_matrix
)
from src.data.augmentation import (
    safe_smote_augmentation,
    augment_sequence_data,
    validate_array_consistency
)
from src.models.graph_transformer import SpatioTemporalGraphTransformer
from src.training.losses import ImbalancedFocalLoss
from src.evaluation.metrics import evaluate_binary_model_with_bootstrap, StatisticalAnalyzer
from src.utils.feature_selection import EnhancedFeatureSelector

from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_model(config: dict):
    """Main training function"""
    
    # Setup
    set_random_seeds(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_gpu'] else "cpu")
    print(f"💻 Device: {device}")
    
    # Create output directory
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("🔍 Loading data...")
    processor = TopSectionsDataProcessor(top_N=config['data']['top_n_sections'])
    df, top_sections, analysis_df = processor.load_and_analyze_top_sections(
        config['data']['file_path']
    )
    
    feature_combinations = processor.define_enhanced_features(df)
    print(f"📊 Feature combinations defined:")
    for key, features in feature_combinations.items():
        print(f"  {key}: {len(features)} features")
    
    # Create sequences
    print("🔄 Creating sequences...")
    X_seq, X_static, y, sections, node_indices, section_to_node = create_multi_section_sequences(
        df, top_sections, feature_combinations, seq_len=config['data']['sequence_length']
    )
    
    # Feature selection
    static_feature_names = [f for f in feature_combinations['all_static'] 
                           if f not in feature_combinations['temporal_sequence']]
    
    if config['feature_selection']['enabled'] and X_static.shape[1] > 1:
        print("🔍 Performing feature selection...")
        feature_selector = EnhancedFeatureSelector(
            n_bootstrap=config['feature_selection']['n_bootstrap']
        )
        
        stability_stats = feature_selector.compute_feature_stability(
            X_static, y, static_feature_names, method=config['feature_selection']['method']
        )
        
        selected_features = feature_selector.select_stable_features(
            stability_stats, 
            top_k=config['feature_selection']['top_k'],
            stability_threshold=config['feature_selection']['stability_threshold']
        )
        
        selected_indices = [static_feature_names.index(f) for f in selected_features 
                          if f in static_feature_names]
        X_static = X_static[:, selected_indices] if len(selected_indices) > 0 else X_static
        print(f"✅ Selected {len(selected_features)} stable features")
    
    # Preprocessing
    print("⚙️ Preprocessing data...")
    seq_scaler = RobustScaler()
    static_scaler = StandardScaler()
    
    B, T, F_seq = X_seq.shape
    X_seq_scaled = seq_scaler.fit_transform(X_seq.reshape(-1, F_seq)).reshape(B, T, F_seq).astype(np.float32)
    
    if X_static.size > 0:
        X_static_scaled = static_scaler.fit_transform(X_static).astype(np.float32)
    else:
        X_static_scaled = np.zeros((X_seq.shape[0], 1), dtype=np.float32)
    
    y_scaled = y.astype(np.float32)
    
    # Validate consistency
    X_seq_scaled, X_static_scaled, y_scaled, sections, node_indices = validate_array_consistency(
        X_seq_scaled, X_static_scaled, y_scaled, sections, node_indices, "after preprocessing"
    )
    
    # Apply SMOTE
    if config['augmentation']['use_smote'] and X_static.size > 0:
        print("🔄 Applying SMOTE...")
        X_static_scaled, y_scaled = safe_smote_augmentation(
            X_static_scaled, y_scaled, 
            random_state=config['seed'],
            k_neighbors=config['augmentation']['smote_k_neighbors']
        )
        
        # Adjust other arrays
        current_size = len(X_seq_scaled)
        new_size = len(X_static_scaled)
        
        if new_size != current_size:
            if new_size > current_size:
                replication_factor = new_size // current_size
                remainder = new_size % current_size
                
                X_seq_scaled = np.concatenate([
                    np.repeat(X_seq_scaled, replication_factor, axis=0),
                    X_seq_scaled[:remainder]
                ], axis=0)
                sections = np.concatenate([
                    np.repeat(sections, replication_factor, axis=0),
                    sections[:remainder]
                ], axis=0)
                node_indices = np.concatenate([
                    np.repeat(node_indices, replication_factor, axis=0),
                    node_indices[:remainder]
                ], axis=0)
    
    # Sequence augmentation
    pos_indices = np.where(y_scaled == 1)[0]
    if len(pos_indices) > 0:
        print(f"🔄 Augmenting {len(pos_indices)} positive samples...")
        X_seq_scaled, X_static_scaled, y_scaled, sections, node_indices = augment_sequence_data(
            X_seq_scaled, X_static_scaled, y_scaled, sections, node_indices, pos_indices,
            augmentation_factor=config['augmentation']['sequence_augmentation_factor'],
            noise_std=config['augmentation']['noise_std']
        )
    
    # Final validation
    X_seq_scaled, X_static_scaled, y_scaled, sections, node_indices = validate_array_consistency(
        X_seq_scaled, X_static_scaled, y_scaled, sections, node_indices, "after augmentation"
    )
    
    pos_ratio = y_scaled.mean()
    pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
    
    print(f"📊 Dataset: {len(X_seq_scaled)} samples, positive ratio: {pos_ratio:.4f}")
    
    # Create adjacency matrix
    adj_matrix = create_adjacency_matrix(
        df, top_sections, section_to_node,
        threshold_km=config['spatial']['adjacency_threshold_km']
    )
    
    # Cross-validation
    unique_sections = np.unique(sections)
    np.random.shuffle(unique_sections)
    k_folds = config['training']['k_folds']
    fold_size = len(unique_sections) // k_folds
    
    fold_metrics = []
    
    print(f"\n🔬 Starting {k_folds}-fold cross-validation")
    
    for fold in range(k_folds):
        print(f"\n📁 Fold {fold + 1}/{k_folds}")
        
        # Split data
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else len(unique_sections)
        test_sections = unique_sections[start_idx:end_idx]
        train_sections = np.concatenate([unique_sections[:start_idx], unique_sections[end_idx:]])
        
        train_mask = np.isin(sections, train_sections)
        test_mask = np.isin(sections, test_sections)
        
        X_seq_train, X_seq_test = X_seq_scaled[train_mask], X_seq_scaled[test_mask]
        X_static_train, X_static_test = X_static_scaled[train_mask], X_static_scaled[test_mask]
        y_train, y_test = y_scaled[train_mask], y_scaled[test_mask]
        node_indices_train, node_indices_test = node_indices[train_mask], node_indices[test_mask]
        
        print(f"   Train: {len(X_seq_train)} samples, Test: {len(X_seq_test)} samples")
        
        # Data loaders
        if y_train.mean() > 0:
            class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)
            sample_weights = class_weights[y_train.astype(int)]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        else:
            sampler = None
        
        train_dataset = TensorDataset(
            torch.from_numpy(X_seq_train),
            torch.from_numpy(X_static_train),
            torch.from_numpy(y_train),
            torch.from_numpy(node_indices_train.astype(np.int64))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_seq_test),
            torch.from_numpy(X_static_test),
            torch.from_numpy(y_test),
            torch.from_numpy(node_indices_test.astype(np.int64))
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            sampler=sampler, shuffle=(sampler is None), pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, pin_memory=True
        )
        
        # Initialize model
        model = SpatioTemporalGraphTransformer(
            seq_dim=X_seq_scaled.shape[2],
            static_dim=X_static_scaled.shape[1],
            num_nodes=len(section_to_node),
            **config['model']
        ).to(device)
        
        criterion = ImbalancedFocalLoss(
            **config['loss'],
            pos_weight=torch.tensor(pos_weight, device=device)
        )
        
        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler']['eta_min']
        )
        
        adj_tensor = torch.from_numpy(adj_matrix).to(device)
        use_amp = config['mixed_precision'] and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Training loop
        best_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(config['training']['epochs']):
            model.train()
            train_loss = 0
            
            for seq_batch, static_batch, target_batch, node_idx_batch in train_loader:
                seq_batch = seq_batch.to(device)
                static_batch = static_batch.to(device)
                target_batch = target_batch.to(device).float()
                node_idx_batch = node_idx_batch.to(device)
                
                batch_adj_mask = adj_tensor[node_idx_batch][:, node_idx_batch]
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(seq_batch, static_batch, node_idx_batch, batch_adj_mask)
                    loss = criterion(logits, target_batch)
                
                if use_amp and scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            if epoch % 5 == 0 or epoch == config['training']['epochs'] - 1:
                model.eval()
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for seq_batch, static_batch, target_batch, node_idx_batch in test_loader:
                        seq_batch = seq_batch.to(device)
                        static_batch = static_batch.to(device)
                        target_batch = target_batch.to(device).float()
                        node_idx_batch = node_idx_batch.to(device)
                        
                        batch_adj_mask = adj_tensor[node_idx_batch][:, node_idx_batch]
                        
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            logits = model(seq_batch, static_batch, node_idx_batch, batch_adj_mask)
                        
                        val_predictions.append(logits.cpu().numpy())
                        val_targets.append(target_batch.cpu().numpy())
                
                val_predictions = np.concatenate(val_predictions)
                val_targets = np.concatenate(val_targets)
                
                # Quick F1 for early stopping
                from src.evaluation.metrics import find_optimal_threshold
                from sklearn.metrics import precision_recall_fscore_support
                
                y_pred_proba = 1 / (1 + np.exp(-val_predictions))
                optimal_threshold = find_optimal_threshold(
                    val_targets, y_pred_proba, 
                    prioritize_recall=config['evaluation']['prioritize_recall']
                )
                y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_targets, y_pred_binary, average='binary', zero_division=0
                )
                
                print(f"   Epoch {epoch:3d}: Loss: {train_loss/len(train_loader):.4f}, "
                      f"F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    patience_counter = 0
                    if config['output']['save_models']:
                        torch.save(model.state_dict(),
                                 os.path.join(output_dir, f"best_model_fold_{fold}.pth"))
                else:
                    patience_counter += 1
                
                if patience_counter >= config['training']['early_stopping_patience']:
                    print(f"   Early stopping at epoch {epoch}")
                    break
        
        # Final evaluation
        model.load_state_dict(torch.load(
            os.path.join(output_dir, f"best_model_fold_{fold}.pth"),
            map_location=device, weights_only=True
        ))
        model.eval()
        
        final_predictions = []
        final_targets = []
        with torch.no_grad():
            for seq_batch, static_batch, target_batch, node_idx_batch in test_loader:
                seq_batch = seq_batch.to(device)
                static_batch = static_batch.to(device)
                target_batch = target_batch.to(device).float()
                node_idx_batch = node_idx_batch.to(device)
                
                batch_adj_mask = adj_tensor[node_idx_batch][:, node_idx_batch]
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(seq_batch, static_batch, node_idx_batch, batch_adj_mask)
                
                final_predictions.append(logits.cpu().numpy())
                final_targets.append(target_batch.cpu().numpy())
        
        final_predictions = np.concatenate(final_predictions)
        final_targets = np.concatenate(final_targets)
        
        final_metrics = evaluate_binary_model_with_bootstrap(
            final_targets, final_predictions,
            prioritize_recall=config['evaluation']['prioritize_recall'],
            n_bootstrap=config['evaluation']['bootstrap_samples']
        )
        fold_metrics.append(final_metrics)
        
        print(f"   Fold {fold + 1} Final Metrics:")
        for metric, stats in final_metrics.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"     {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Calculate overall statistics
    overall_stats = {}
    analyzer = StatisticalAnalyzer()
    
    for metric in ['precision', 'recall', 'f1', 'auc', 'accuracy']:
        values = [fm[metric]['mean'] for fm in fold_metrics]
        mean, ci_lower, ci_upper = analyzer.compute_confidence_interval(values)
        overall_stats[metric] = {
            'mean': mean,
            'std': np.std(values, ddof=1),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'values': values
        }
    
    # Save results
    if config['output']['save_metrics']:
        results = {
            'fold_metrics': fold_metrics,
            'overall_statistics': overall_stats,
            'config': config
        }
        
        with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 FINAL RESULTS")
    print("="*80)
    for metric, stats in overall_stats.items():
        print(f"{metric.upper():>12}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Spatio-Temporal Graph Transformer')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    results = train_model(config)
    
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()