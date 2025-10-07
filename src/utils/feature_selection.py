"""
Enhanced Feature Selection with Statistical Analysis
"""
import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class EnhancedFeatureSelector:
    """Enhanced feature selector with stability analysis"""
    
    def __init__(self, random_state: int = 42, n_bootstrap: int = 100):
        """
        Args:
            random_state: Random seed for reproducibility
            n_bootstrap: Number of bootstrap iterations
        """
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.feature_stability_scores_ = {}
        
    def compute_feature_stability(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], 
                                method: str = 'rf') -> Dict[str, Dict[str, float]]:
        """
        Compute feature importance stability across bootstrap samples
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            method: 'rf' for Random Forest or 'lasso' for Lasso
            
        Returns:
            Dictionary of stability statistics for each feature
        """
        importance_scores = []
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            if method == 'rf':
                model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=i, 
                    class_weight='balanced'
                )
                model.fit(X_boot, y_boot)
                importances = model.feature_importances_
                
            elif method == 'lasso':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_boot)
                model = LogisticRegression(
                    penalty='l1', 
                    solver='liblinear', 
                    C=0.1, 
                    random_state=i, 
                    class_weight='balanced'
                )
                model.fit(X_scaled, y_boot)
                importances = np.abs(model.coef_[0])
            else:
                raise ValueError(f"Unknown method: {method}")
            
            importance_scores.append(importances)
        
        importance_scores = np.array(importance_scores)
        
        # Calculate statistics for each feature
        stability_stats = {}
        for i, feature_name in enumerate(feature_names):
            feature_scores = importance_scores[:, i]
            stability_stats[feature_name] = {
                'mean': np.mean(feature_scores),
                'std': np.std(feature_scores),
                'cv': np.std(feature_scores) / (np.mean(feature_scores) + 1e-8),  # Coefficient of variation
                'ci_lower': np.percentile(feature_scores, 2.5),
                'ci_upper': np.percentile(feature_scores, 97.5)
            }
        
        self.feature_stability_scores_ = stability_stats
        return stability_stats
    
    def select_stable_features(self, stability_stats: Dict[str, Dict[str, float]], 
                             top_k: int = 15, 
                             stability_threshold: float = 0.5) -> List[str]:
        """
        Select features based on importance and stability
        
        Args:
            stability_stats: Dictionary of stability statistics
            top_k: Number of features to select
            stability_threshold: Maximum coefficient of variation (lower is more stable)
            
        Returns:
            List of selected feature names
        """
        # Filter features by coefficient of variation
        stable_features = {
            name: stats for name, stats in stability_stats.items() 
            if stats['cv'] < stability_threshold
        }
        
        if len(stable_features) < top_k:
            print(f"Warning: Only {len(stable_features)} stable features found")
            stable_features = stability_stats
        
        # Sort by mean importance
        sorted_features = sorted(
            stable_features.items(), 
            key=lambda x: x[1]['mean'], 
            reverse=True
        )
        
        return [name for name, _ in sorted_features[:top_k]]
    
    def get_feature_report(self, feature_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get detailed report of feature stability
        
        Args:
            feature_names: Optional list to filter features
            
        Returns:
            Dictionary of feature statistics
        """
        if feature_names is None:
            return self.feature_stability_scores_
        
        return {
            name: stats for name, stats in self.feature_stability_scores_.items() 
            if name in feature_names
        }