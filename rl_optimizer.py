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

class EggrollOptimizer(GradientFreeOptimizer):
    """
    EGGROLL Optimizer: Evolution Guided General Optimization via Low-rank Learning.
    Paper: https://arxiv.org/abs/2511.16652v1
    
    Approximates the gradient using low-rank perturbations E = (1/sqrt(r)) * A * B^T.
    """
    def __init__(self, env, param_groups, population_size=10, sigma=0.1, learning_rate=0.01, rank=4, seed=42):
        """
        Args:
            rank: Rank of the perturbation matrix approximation.
            seed: Master seed for reproducibility.
        """
        super().__init__(env, param_groups, population_size, sigma, learning_rate)
        self.rank = rank
        self.param_names = [n for n, _ in self.model.named_parameters()]
        
        # Initialize a dedicated generator for stability across runs
        self.rng = torch.Generator(device='cpu')
        self.rng.manual_seed(seed)
        
        # Determine device from model
        param = next(self.model.parameters(), None)
        self.device = param.device if param is not None else 'cpu'

    def step(self, puzzles, solutions=None):
        # Generate seeds
        step_seeds = torch.randint(
            high=2**32, 
            size=(self.population_size,), 
            generator=self.rng, 
            device='cpu'
        ).tolist()
        
        # Evaluate population
        rewards = []
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for i, seed in enumerate(step_seeds):
            # Perturb model
            self._apply_perturbation(seed, sign=1.0)
            # Evaluate
            reward, metrics = self._evaluate_batch(self.model, puzzles, solutions)
            rewards.append(reward)
            # Restore
            self.model.load_state_dict(original_state)
            
        # Compute update
        # Use first parameter's dtype to ensure consistency
        first_param = next(self.model.parameters(), None)
        target_dtype = first_param.dtype if first_param is not None else torch.float32
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=target_dtype)
        
        # Standardize rewards
        if rewards_tensor.std() > 1e-8:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / rewards_tensor.std()
        else:
            rewards_tensor = rewards_tensor - rewards_tensor.mean()
                
        final_update = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if len(param.shape) == 2:
                # Low-rank update
                rows, cols = param.shape
                eff_rank = min(self.rank, rows, cols)
                scale = 1.0 / np.sqrt(eff_rank)
                
                # Reconstruct factors
                A_stack, B_stack = self._generate_batch_low_rank_factors(
                    name, rows, cols, eff_rank, param.dtype, self.device, step_seeds
                )
                # Weighted sum
                weighted_A = A_stack * rewards_tensor.view(-1, 1, 1)
                grad_approx = scale * torch.einsum('nri,nci->rc', weighted_A, B_stack)
                
            else:
                # Full-rank update
                grad_approx = torch.zeros_like(param)
                for idx, seed in enumerate(step_seeds):
                    noise = self._generate_vector_noise(name, param.shape, param.dtype, self.device, seed)
                    grad_approx += rewards_tensor[idx] * noise
            
            # Perform update
            update_step = (self.lr / (self.population_size * self.sigma)) * grad_approx
            final_update[name] = param + update_step

        # Apply changes
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in final_update:
                    p.copy_(final_update[name])
                        
        return max(rewards), {}

    def _apply_perturbation(self, seed, sign=1.0):
        """Generates and applies noise to the model in-place."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            noise = self._generate_param_noise(name, param.shape, param.dtype, self.device, seed)
            with torch.no_grad():
                param.add_(noise, alpha=sign * self.sigma)

    def _generate_param_noise(self, name, shape, dtype, device, seed):
        """Generate single instance of noise (E) for a parameter."""
        # Mix seed with param name hash for structural diversity
        local_seed = (seed + abs(hash(name))) % 2**32
        g = torch.Generator(device='cpu')
        g.manual_seed(local_seed)
        
        if len(shape) == 2:
            # Low-rank
            rows, cols = shape
            eff_rank = min(self.rank, rows, cols)
            scale = 1.0 / np.sqrt(eff_rank)
            
            A = torch.randn(rows, eff_rank, generator=g, dtype=dtype, device='cpu').to(device)
            B = torch.randn(cols, eff_rank, generator=g, dtype=dtype, device='cpu').to(device)
            return scale * torch.mm(A, B.t())
        else:
            # Vector
            return torch.randn(shape, generator=g, dtype=dtype, device='cpu').to(device)

    def _generate_vector_noise(self, name, shape, dtype, device, seed):
        """Re-implementation of noise gen for just vectors/non-matrices."""
        local_seed = (seed + abs(hash(name))) % 2**32
        g = torch.Generator(device='cpu')
        g.manual_seed(local_seed)
        return torch.randn(shape, generator=g, dtype=dtype, device='cpu').to(device)

    def _generate_batch_low_rank_factors(self, name, rows, cols, rank, dtype, device, seeds):
        """
        Reconstructs A and B matrices for the entire batch of seeds.
        Returns:
            A_stack: (N, rows, rank)
            B_stack: (N, cols, rank)
        """
        N = len(seeds)
        
        A_list = []
        B_list = []
        
        # Reuse generator
        g = torch.Generator(device='cpu')
        
        for seed in seeds:
            local_seed = (seed + abs(hash(name))) % 2**32
            g.manual_seed(local_seed)
            
            # Generate on CPU
            a = torch.randn(rows, rank, generator=g, dtype=dtype)
            b = torch.randn(cols, rank, generator=g, dtype=dtype)
            
            A_list.append(a)
            B_list.append(b)
            
        # Stack and move to device once
        A_stack = torch.stack(A_list).to(device)
        B_stack = torch.stack(B_list).to(device)
        
        return A_stack, B_stack

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
