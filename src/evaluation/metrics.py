"""
Evaluation Metrics with Statistical Analysis
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    precision_recall_curve
)
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis utilities"""
    
    @staticmethod
    def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute mean and confidence interval
        
        Args:
            data: Array of values
            confidence: Confidence level
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        h = std * stats.t.ppf((1 + confidence) / 2., n-1) / np.sqrt(n)
        return mean, mean - h, mean + h
    
    @staticmethod
    def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, 
                        metric_func, n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        Bootstrap confidence intervals for metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels/probabilities
            metric_func: Function to compute metric
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        return {
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'ci_lower': np.percentile(bootstrap_scores, 2.5),
            'ci_upper': np.percentile(bootstrap_scores, 97.5)
        }


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          prioritize_recall: bool = False) -> float:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        prioritize_recall: If True, optimize for recall
        
    Returns:
        Optimal threshold value
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    if prioritize_recall:
        f1_scores = recalls + (1 - precisions) - 1
    else:
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5


def evaluate_binary_model_with_bootstrap(y_true: np.ndarray, y_pred_logits: np.ndarray, 
                                       use_optimal_threshold: bool = True, 
                                       prioritize_recall: bool = False,
                                       n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation with bootstrap confidence intervals
    
    Args:
        y_true: True labels
        y_pred_logits: Model logits (before sigmoid)
        use_optimal_threshold: Whether to find optimal threshold
        prioritize_recall: Optimize for recall vs F1
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary of metrics with statistics
    """
    y_pred_proba = 1 / (1 + np.exp(-y_pred_logits))
    
    if len(np.unique(y_true)) == 1:
        return {
            'precision': {'mean': 1.0 if y_true[0] == 1 else 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 1.0},
            'recall': {'mean': 1.0 if y_true[0] == 1 else 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 1.0},
            'f1': {'mean': 1.0 if y_true[0] == 1 else 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 1.0},
            'auc': {'mean': 0.5, 'std': 0.0, 'ci_lower': 0.5, 'ci_upper': 0.5},
            'accuracy': {'mean': 1.0, 'std': 0.0, 'ci_lower': 1.0, 'ci_upper': 1.0},
            'threshold': 0.5
        }
    
    try:
        optimal_threshold = find_optimal_threshold(y_true, y_pred_proba, prioritize_recall)
        y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='binary', zero_division=0
        )
        auc = average_precision_score(y_true, y_pred_proba)
        accuracy = (y_true == y_pred_binary).mean()
        
        # Bootstrap confidence intervals
        analyzer = StatisticalAnalyzer()
        
        def precision_score(y_t, y_p):
            p, _, _, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
            return p
        
        def recall_score(y_t, y_p):
            _, r, _, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
            return r
        
        def f1_score(y_t, y_p):
            _, _, f, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
            return f
        
        def accuracy_score(y_t, y_p):
            return (y_t == y_p).mean()
        
        precision_stats = analyzer.bootstrap_metric(y_true, y_pred_binary, precision_score, n_bootstrap)
        recall_stats = analyzer.bootstrap_metric(y_true, y_pred_binary, recall_score, n_bootstrap)
        f1_stats = analyzer.bootstrap_metric(y_true, y_pred_binary, f1_score, n_bootstrap)
        accuracy_stats = analyzer.bootstrap_metric(y_true, y_pred_binary, accuracy_score, n_bootstrap)
        
        def auc_score(y_t, y_p_prob):
            return average_precision_score(y_t, y_p_prob)
        
        auc_stats = analyzer.bootstrap_metric(y_true, y_pred_proba, auc_score, n_bootstrap)
        
        return {
            'precision': precision_stats,
            'recall': recall_stats,
            'f1': f1_stats,
            'auc': auc_stats,
            'accuracy': accuracy_stats,
            'threshold': optimal_threshold,
            'pos_ratio': y_true.mean()
        }
    except Exception as e:
        print(f"Warning: Error in evaluation: {e}")
        return {
            'precision': {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0},
            'recall': {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0},
            'f1': {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0},
            'auc': {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0},
            'accuracy': {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0},
            'threshold': 0.5,
            'pos_ratio': y_true.mean() if len(y_true) > 0 else 0.0
        }