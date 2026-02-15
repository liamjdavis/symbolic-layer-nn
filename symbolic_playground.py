import torch
import numpy as np
import argparse
from nn import SudokuNet, load_kaggle_data
from symbolic_layer import ArgmaxLayer, MaxSATLayer, LPLayer, QUBOLayer
from environment import SudokuRLEnv
from rl_optimizer import RandomSearchOptimizer, EvolutionStrategyOptimizer, EggrollOptimizer

def main():
    parser = argparse.ArgumentParser(description="Sudoku Symbolic Layer Workbench")
    parser.add_argument('--layer', type=str, default='argmax',
                        choices=['argmax', 'maxsat', 'lp', 'qubo'],
                        help="Symbolic layer type")
    parser.add_argument('--optimizer', type=str, default='random',
                        choices=['random', 'es', 'eggroll'],
                        help="Gradient-free optimizer type")
    parser.add_argument('--samples', type=int, default=10,
                        help="Number of samples to evaluate on per step")
    parser.add_argument('--steps', type=int, default=5,
                        help="Number of optimization steps")
    parser.add_argument('--model_path', type=str, default='sudoku_model.pth',
                        help="Path to pre-trained model")
    
    args = parser.parse_args()
    
    print(f"=== Setting up Workbench: {args.layer.upper()} Layer + {args.optimizer.upper()} Optimizer ===")

    # 1. Load Data
    print("Loading Sudoku Data...")
    try:
        puzzles, solutions = load_kaggle_data('sudoku.csv', max_samples=1000)
        # Use a small subset for rapid workbench testing
        test_puzzles = puzzles[:args.samples]
        test_solutions = solutions[:args.samples]
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy data.")
        test_puzzles = ["0"*81] * args.samples
        test_solutions = ["1"*81] * args.samples

    # 2. Load Model
    print(f"Loading Model from {args.model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
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
        
        # The last one is the output layer
        for idx in layer_indices[:-1]:
            weight = state_dict[f'network.{idx}.weight']
            hidden_sizes.append(weight.shape[0])
            
        print(f"Detected hidden sizes from checkpoint: {hidden_sizes}")
        model = SudokuNet(hidden_sizes=hidden_sizes)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using initialized model (random weights) with default [256, 256].")
        model = SudokuNet(hidden_sizes=[256, 256])

    # 3. Setup Symbolic Layer
    if args.layer == 'argmax':
        layer = ArgmaxLayer()
    elif args.layer == 'maxsat':
        layer = MaxSATLayer()
    elif args.layer == 'lp':
        layer = LPLayer()
    elif args.layer == 'qubo':
        layer = QUBOLayer()
    else:
        raise ValueError("Unknown layer type")
        
    # 4. Setup Environment
    env = SudokuRLEnv(model, layer, device=device)
    
    # 5. Setup Optimizer
    if args.optimizer == 'random':
        optimizer = RandomSearchOptimizer(env, model.parameters(), population_size=5, sigma=0.01)
    elif args.optimizer == 'es':
        optimizer = EvolutionStrategyOptimizer(env, model.parameters())
    elif args.optimizer == 'eggroll':
         optimizer = EggrollOptimizer(env, model.parameters(), population_size=10, sigma=0.1, learning_rate=0.01)

    # 6. Run Loop
    print("\nStarting Optimization Loop...")
    for step in range(args.steps):
        print(f"\nStep {step+1}/{args.steps}")
        
        # Take optimization step
        try:
            best_reward, best_metrics = optimizer.step(test_puzzles, test_solutions)
            print(f"  Step Best Reward:   {best_reward:.4f}")
            print("  Metrics:")
            # Sort metrics for cleaner display
            for k in sorted(best_metrics.keys()):
                v = best_metrics[k]
                if isinstance(v, float):
                    print(f"    {k:<20}: {v:.4f}")
                else:
                    print(f"    {k:<20}: {v}")
                    
        except NotImplementedError:
            print("  Optimizer step not implemented yet.")
            break
            
    print("\nWorkbench run complete.")
    
if __name__ == "__main__":
    main()
