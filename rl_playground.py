import torch
import numpy as np
import argparse
import sys
from nn import SudokuNet, load_kaggle_data
from symbolic_layer import ArgmaxLayer, MaxSATLayer, LPLayer, QUBOLayer
from environment import SudokuRLEnv
from rl_optimizer import RandomSearchOptimizer, EvolutionStrategyOptimizer, EggrollOptimizer

class SudokuRLPlayground:
    """
    RL Playground for experimenting with different optimizers and symbolic layers.
    Encapsulates the setup and execution of the RL loop.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.layer = None
        self.env = None
        self.optimizer = None
        self.test_puzzles = []
        self.test_solutions = []

    def setup(self):
        """Initializes all components for the playground."""
        print(f"=== Setting up RL Playground: {self.args.layer.upper()} Layer + {self.args.optimizer.upper()} Optimizer ===")
        self._load_data()
        self._load_model()
        self._setup_layer()
        self._setup_environment()
        self._setup_optimizer()

    def run(self):
        """Executes the optimization loop."""
        if not self.optimizer:
            raise ValueError("Optimizer not initialized. Call setup() first.")

        print("\nStarting RL Optimization Loop...")
        for step in range(self.args.steps):
            try:
                best_reward, best_metrics = self.optimizer.step(self.test_puzzles, self.test_solutions)
                
                # Print status
                print(f"Step {step+1}/{self.args.steps} | Best Reward: {best_reward:.4f}")
                if (step + 1) % 10 == 0 or step == 0:
                    print("  Metrics:")
                    for k in sorted(best_metrics.keys()):
                        v = best_metrics[k]
                        if isinstance(v, float):
                            print(f"    {k:<20}: {v:.4f}")
                        else:
                            print(f"    {k:<20}: {v}")
                        
            except NotImplementedError:
                print("  Optimizer step not implemented yet.")
                break
                
        print("\nRL run complete.")

    def _load_data(self):
        """Loads Sudoku puzzles and solutions."""
        print("Loading Sudoku Data...")
        try:
            puzzles, solutions = load_kaggle_data('sudoku.csv', max_samples=1000)
            self.test_puzzles = puzzles[:self.args.samples]
            self.test_solutions = solutions[:self.args.samples]
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using dummy data.")
            self.test_puzzles = ["0"*81] * self.args.samples
            self.test_solutions = ["1"*81] * self.args.samples

    def _load_model(self):
        """Loads the pre-trained neural network model."""
        print(f"Loading Model from {self.args.model_path}...")
        try:
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Infer hidden sizes
            hidden_sizes = []
            layer_indices = []
            for key in state_dict.keys():
                if key.startswith("network.") and key.endswith(".weight"):
                    parts = key.split('.')
                    if len(parts) == 3 and parts[1].isdigit():
                        layer_indices.append(int(parts[1]))
            
            layer_indices.sort()
            
            # The last one is the output layer, so we exclude it
            for idx in layer_indices[:-1]:
                weight = state_dict[f'network.{idx}.weight']
                hidden_sizes.append(weight.shape[0])
            
            print(f"Detected hidden sizes from checkpoint: {hidden_sizes}")
            self.model = SudokuNet(hidden_sizes=hidden_sizes)
            self.model.load_state_dict(state_dict)
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using initialized model (random weights) with default [256, 256].")
            self.model = SudokuNet(hidden_sizes=[256, 256])

    def _setup_layer(self):
        """Initializes the symbolic layer."""
        if self.args.layer == 'argmax':
            self.layer = ArgmaxLayer()
        elif self.args.layer == 'maxsat':
            self.layer = MaxSATLayer()
        elif self.args.layer == 'lp':
            self.layer = LPLayer()
        elif self.args.layer == 'qubo':
            self.layer = QUBOLayer()
        else:
            raise ValueError(f"Unknown layer type: {self.args.layer}")

    def _setup_environment(self):
        """Sets up the Sudoku RL environment."""
        self.env = SudokuRLEnv(self.model, self.layer, device=self.device)

    def _setup_optimizer(self):
        """Initializes the gradient-free optimizer."""
        if self.args.optimizer == 'random':
            self.optimizer = RandomSearchOptimizer(
                self.env, self.model.parameters(), population_size=5, sigma=0.01
            )
        elif self.args.optimizer == 'es':
            self.optimizer = EvolutionStrategyOptimizer(self.env, self.model.parameters())
        elif self.args.optimizer == 'eggroll':
            self.optimizer = EggrollOptimizer(
                self.env, self.model.parameters(), population_size=10, sigma=0.1, learning_rate=0.01
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.args.optimizer}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sudoku RL Playground")
    parser.add_argument('--layer', type=str, default='argmax',
                        choices=['argmax', 'maxsat', 'lp', 'qubo'],
                        help="Symbolic layer type")
    parser.add_argument('--optimizer', type=str, default='eggroll',
                        choices=['random', 'es', 'eggroll'],
                        help="Gradient-free optimizer type")
    parser.add_argument('--samples', type=int, default=10,
                        help="Number of samples to evaluate on per step")
    parser.add_argument('--steps', type=int, default=100,
                        help="Number of optimization steps")
    parser.add_argument('--model_path', type=str, default='sudoku_model.pth',
                        help="Path to pre-trained model")
    return parser.parse_args()

def main():
    args = parse_args()
    playground = SudokuRLPlayground(args)
    playground.setup()
    playground.run()

if __name__ == "__main__":
    main()