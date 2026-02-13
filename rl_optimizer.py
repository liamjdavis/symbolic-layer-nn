import numpy as np
import copy
import torch
from collections import defaultdict

class GradientFreeOptimizer:
    """
    Base class for gradient-free optimization algorithms.
    Designed to work with the SudokuRLEnv.
    """
    def __init__(self, env, param_groups, population_size=10, sigma=0.1, learning_rate=0.01):
        """
        Args:
            env: SudokuRLEnv instance
            param_groups: List of dicts (like PyTorch optimizer) or list of tensors to optimize.
                          If we are optimizing NN weights, this is model.parameters().
            population_size: Number of candidates per step
            sigma: Noise standard deviation for perturbation
            learning_rate: Step size for update
        """
        self.env = env
        # Flatten parameters into a single vector for easier manipulation if needed,
        # or keep them as reference to the model's tensors.
        # For simplicity in this workbench, let's assume we work directly with the model attached to env.
        self.model = env.nn_model 
        self.population_size = population_size
        self.sigma = sigma
        self.lr = learning_rate
        
    def step(self, puzzles, solutions=None):
        """
        Perform one optimization step.
        """
        raise NotImplementedError

class RandomSearchOptimizer(GradientFreeOptimizer):
    """
    Simple Random Search:
    1. Perturb current weights
    2. Evaluate
    3. If better, keep.
    """
    def step(self, puzzles, solutions=None):
        # 1. Evaluate current baseline
        baseline_reward, baseline_metrics = self._evaluate_batch(self.model, puzzles, solutions)
        
        # 2. Perturb
        # We need a way to perturb weights without destroying the original model immediately
        # Clone model state
        original_state = copy.deepcopy(self.model.state_dict())
        
        best_reward = baseline_reward
        best_metrics = baseline_metrics
        best_state = original_state
        
        for i in range(self.population_size):
            # Apply perturbation
            perturbed_state = self._perturb(original_state, self.sigma)
            self.model.load_state_dict(perturbed_state)
            
            # Evaluate
            reward, metrics = self._evaluate_batch(self.model, puzzles, solutions)
            
            if reward > best_reward:
                best_reward = reward
                best_metrics = metrics
                best_state = perturbed_state
                # print(f"  Candidate {i}: New best reward {best_reward:.4f}")
        
        # 3. Update
        self.model.load_state_dict(best_state)
        return best_reward, best_metrics

    def _perturb(self, state_dict, sigma):
        new_state = {}
        for k, v in state_dict.items():
            if v.dtype in [torch.float32, torch.float64]:
                noise = torch.randn_like(v) * sigma
                new_state[k] = v + noise
            else:
                new_state[k] = v
        return new_state

    def _evaluate_batch(self, model, puzzles, solutions):
        """Helper to run env evaluation on a batch."""
        # env.evaluate runs one puzzle. We iterate.
        # Note: This is slow. Vectorized env would be better, but this is a workbench.
        total_reward = 0
        aggregated_metrics = defaultdict(float)
        
        # Temporarily ensure model is in eval mode (though env handles it)
        model.eval()
        
        n = len(puzzles)
        for i in range(n):
            p = puzzles[i]
            s = solutions[i] if solutions else None
            r, _, m = self.env.evaluate(p, s)
            total_reward += r
            for k, v in m.items():
                aggregated_metrics[k] += v
                
        # Average metrics
        avg_metrics = {k: v / n for k, v in aggregated_metrics.items()}
            
        return total_reward / n, avg_metrics

class EvolutionStrategyOptimizer(GradientFreeOptimizer):
    """
    OpenAI-ES style optimizer.
    w_new = w + alpha * (1/sigma) * mean(reward_i * noise_i)
    """
    def step(self, puzzles, solutions=None):
        # TODO: Implement ES
        # 1. Generate N noise vectors (epsilon)
        # 2. Evaluate w + epsilon and w - epsilon (Mirrored Sampling)
        # 3. Estimate gradient
        # 4. Update weights
        print("ES Step placeholder")
        return 0.0, {}