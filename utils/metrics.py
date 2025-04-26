"""
Metrics utilities for MR-VFL scheduler comparison.
"""

import numpy as np

def calculate_metrics(results):
    """
    Calculate additional metrics from results.
    
    Args:
        results: Dictionary of results for each scheduler
        
    Returns:
        Dictionary of additional metrics
    """
    metrics = {}
    
    # Calculate relative performance
    if "Random" in results:
        baseline = "Random"
        baseline_performance = results[baseline]['avg_performance']
        baseline_reward = results[baseline]['avg_reward']
        baseline_time = results[baseline]['avg_decision_time']
        
        for name in results:
            if name != baseline:
                relative_performance = results[name]['avg_performance'] / baseline_performance
                relative_reward = results[name]['avg_reward'] / baseline_reward
                
                if baseline_time > 0 and results[name]['avg_decision_time'] > 0:
                    relative_speed = baseline_time / results[name]['avg_decision_time']
                else:
                    relative_speed = 1.0
                
                metrics[name] = {
                    'relative_performance': relative_performance,
                    'relative_reward': relative_reward,
                    'relative_speed': relative_speed,
                    'performance_improvement': (relative_performance - 1) * 100,
                    'reward_improvement': (relative_reward - 1) * 100,
                    'speed_improvement': (relative_speed - 1) * 100
                }
    
    # Calculate performance stability (coefficient of variation)
    for name in results:
        performances = results[name]['performances']
        if performances:
            mean_performance = np.mean(performances)
            std_performance = np.std(performances)
            
            if mean_performance > 0:
                cv_performance = std_performance / mean_performance
            else:
                cv_performance = 0
            
            if name not in metrics:
                metrics[name] = {}
            
            metrics[name]['performance_stability'] = 1 - cv_performance
    
    return metrics

def calculate_scheduling_metrics(results):
    """
    Calculate additional metrics from scheduling count results.
    
    Args:
        results: Dictionary of scheduling count results for each scheduler
        
    Returns:
        Dictionary of additional metrics
    """
    metrics = {}
    
    # Calculate quality ratio
    for name in results:
        high_quality = results[name]['high_quality_scheduled']
        low_quality = results[name]['low_quality_scheduled']
        total = results[name]['unique_vehicles_scheduled']
        
        if total > 0:
            quality_ratio = high_quality / total
            low_quality_ratio = low_quality / total
        else:
            quality_ratio = 0
            low_quality_ratio = 0
        
        metrics[name] = {
            'quality_ratio': quality_ratio,
            'low_quality_ratio': low_quality_ratio,
            'quality_score': quality_ratio - low_quality_ratio
        }
    
    # Calculate relative metrics
    if "Random" in results:
        baseline = "Random"
        baseline_unique = results[baseline]['unique_vehicles_scheduled']
        baseline_high_quality = results[baseline]['high_quality_scheduled']
        baseline_success_rate = results[baseline]['scheduling_success_rate']
        
        for name in results:
            if name != baseline:
                if baseline_unique > 0:
                    relative_unique = results[name]['unique_vehicles_scheduled'] / baseline_unique
                else:
                    relative_unique = 1.0
                
                if baseline_high_quality > 0:
                    relative_high_quality = results[name]['high_quality_scheduled'] / baseline_high_quality
                else:
                    relative_high_quality = 1.0
                
                if baseline_success_rate > 0:
                    relative_success_rate = results[name]['scheduling_success_rate'] / baseline_success_rate
                else:
                    relative_success_rate = 1.0
                
                metrics[name]['relative_unique'] = relative_unique
                metrics[name]['relative_high_quality'] = relative_high_quality
                metrics[name]['relative_success_rate'] = relative_success_rate
    
    return metrics
