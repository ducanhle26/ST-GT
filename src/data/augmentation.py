"""
Data Augmentation Utilities
"""
import numpy as np
from typing import Tuple
from imblearn.over_sampling import SMOTE


def safe_smote_augmentation(X_static: np.ndarray, y: np.ndarray, 
                           random_state: int = 42, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely apply SMOTE with error handling
    
    Args:
        X_static: Static features
        y: Labels
        random_state: Random seed
        k_neighbors: Number of neighbors for SMOTE
        
    Returns:
        Resampled X and y
    """
    try:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        if len(unique_classes) < 2:
            print("   Skipping SMOTE: Only one class present")
            return X_static, y
        
        if min_class_count < 2:
            print("   Skipping SMOTE: Minority class has < 2 samples")
            return X_static, y
        
        k_neighbors = min(k_neighbors, min_class_count - 1)
        
        if k_neighbors < 1:
            print("   Skipping SMOTE: Not enough neighbors")
            return X_static, y
        
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_static, y)
        
        print(f"   SMOTE applied: {len(X_static)} → {len(X_resampled)} samples")
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"   SMOTE failed: {str(e)}. Using original data.")
        return X_static, y


def augment_sequence_data(X_seq: np.ndarray, X_static: np.ndarray, y: np.ndarray, 
                         sections: np.ndarray, node_indices: np.ndarray, 
                         pos_indices: np.ndarray, augmentation_factor: int = 2,
                         noise_std: float = 0.05) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment minority class sequences by adding Gaussian noise
    
    Args:
        X_seq: Sequential features
        X_static: Static features
        y: Labels
        sections: Section IDs
        node_indices: Node indices
        pos_indices: Indices of positive samples
        augmentation_factor: Number of augmented samples per original
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        Augmented arrays
    """
    if len(pos_indices) == 0:
        return X_seq, X_static, y, sections, node_indices
    
    valid_indices = pos_indices[pos_indices < len(X_seq)]
    if len(valid_indices) == 0:
        print("Warning: No valid indices for augmentation")
        return X_seq, X_static, y, sections, node_indices
    
    augmented_X_seq = X_seq.copy()
    augmented_X_static = X_static.copy()
    augmented_y = y.copy()
    augmented_sections = sections.copy()
    augmented_node_indices = node_indices.copy()
    
    for idx in valid_indices:
        for _ in range(augmentation_factor):
            noise = np.random.normal(0, noise_std, X_seq[idx].shape).astype(np.float32)
            augmented_seq = X_seq[idx] + noise
            augmented_X_seq = np.concatenate([augmented_X_seq, [augmented_seq]], axis=0)
            augmented_X_static = np.concatenate([augmented_X_static, [X_static[idx]]], axis=0)
            augmented_y = np.concatenate([augmented_y, [y[idx]]], axis=0)
            augmented_sections = np.concatenate([augmented_sections, [sections[idx]]], axis=0)
            augmented_node_indices = np.concatenate([augmented_node_indices, [node_indices[idx]]], axis=0)
    
    return augmented_X_seq, augmented_X_static, augmented_y, augmented_sections, augmented_node_indices


def validate_array_consistency(X_seq: np.ndarray, X_static: np.ndarray, y: np.ndarray, 
                             sections: np.ndarray, node_indices: np.ndarray, 
                             step: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray]:
    """
    Validate and fix array size consistency
    
    Args:
        X_seq, X_static, y, sections, node_indices: Data arrays
        step: Description of current processing step
        
    Returns:
        Validated and aligned arrays
    """
    sizes = [len(X_seq), len(X_static), len(y), len(sections), len(node_indices)]
    if not all(size == sizes[0] for size in sizes):
        print(f"WARNING: Array size mismatch at {step}")
        print(f"  X_seq: {len(X_seq)}, X_static: {len(X_static)}, y: {len(y)}, "
              f"sections: {len(sections)}, node_indices: {len(node_indices)}")
        
        min_size = min(sizes)
        X_seq = X_seq[:min_size]
        X_static = X_static[:min_size]
        y = y[:min_size]
        sections = sections[:min_size]
        node_indices = node_indices[:min_size]
        print(f"  Fixed by truncating to {min_size} samples")
    
    return X_seq, X_static, y, sections, node_indices