import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class SudokuDataset(Dataset):
    """Dataset for Sudoku puzzles from Kaggle format."""
    
    def __init__(self, puzzles, solutions):
        """
        Args:
            puzzles: List of puzzle strings (81 chars, '0' for empty)
            solutions: List of solution strings (81 chars)
        """
        self.puzzles = puzzles
        self.solutions = solutions
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        # Convert puzzle to string if needed, then to numpy array
        puzzle_str = str(self.puzzles[idx])
        solution_str = str(self.solutions[idx])
        
        puzzle = np.array([int(c) for c in puzzle_str], dtype=np.float32)
        solution = np.array([int(c) for c in solution_str], dtype=np.long)
        
        # Normalize puzzle (0-9 -> 0-1)
        puzzle = puzzle / 9.0
        
        # Solution should be 0-indexed (1-9 -> 0-8) for cross-entropy
        solution = solution - 1
        
        return torch.tensor(puzzle), torch.tensor(solution)


class SudokuNet(nn.Module):
    """Simple fully connected ReLU network for Sudoku solving."""
    
    def __init__(self, hidden_sizes=[512, 512, 512]):
        """
        Args:
            hidden_sizes: List of hidden layer sizes
        """
        super(SudokuNet, self).__init__()
        
        layers = []
        input_size = 81
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer: 81 cells × 9 possible digits
        layers.append(nn.Linear(input_size, 81 * 9))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 81) - normalized puzzle
        Returns:
            (batch_size, 81, 9) - logits for each cell
        """
        out = self.network(x)
        return out.view(-1, 81, 9)


class SudokuTrainer:
    """Training utility for Sudoku network."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def setup_optimizer(self, lr=1e-3, weight_decay=0):
        """Setup Adam optimizer."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for puzzles, solutions in tqdm(train_loader, desc="Training", leave=False):
            puzzles = puzzles.to(self.device)
            solutions = solutions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(puzzles)  # (batch, 81, 9)
            
            # Compute loss (cross-entropy over all 81 cells)
            loss = self.criterion(outputs.view(-1, 9), solutions.view(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct_cells = 0
        total_cells = 0
        correct_puzzles = 0
        total_puzzles = 0
        
        with torch.no_grad():
            for puzzles, solutions in tqdm(val_loader, desc="Validating", leave=False):
                puzzles = puzzles.to(self.device)
                solutions = solutions.to(self.device)
                
                outputs = self.model(puzzles)
                loss = self.criterion(outputs.view(-1, 9), solutions.view(-1))
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=2)  # (batch, 81)
                correct_cells += (predictions == solutions).sum().item()
                total_cells += solutions.numel()
                
                # Puzzle-level accuracy (all 81 cells correct)
                correct_puzzles += (predictions == solutions).all(dim=1).sum().item()
                total_puzzles += solutions.size(0)
        
        avg_loss = total_loss / len(val_loader)
        cell_accuracy = correct_cells / total_cells
        puzzle_accuracy = correct_puzzles / total_puzzles
        
        return avg_loss, cell_accuracy, puzzle_accuracy
    
    def train(self, train_loader, val_loader, epochs=10):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, cell_acc, puzzle_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(cell_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Cell Accuracy: {cell_acc:.4f} ({cell_acc*100:.2f}%)")
            print(f"  Puzzle Accuracy: {puzzle_acc:.4f} ({puzzle_acc*100:.2f}%)")
            print()
    
    def predict(self, puzzle_string):
        """
        Predict solution for a single puzzle.
        
        Args:
            puzzle_string: 81-character string representing puzzle
        Returns:
            predicted solution as 81-character string
        """
        self.model.eval()
        
        # Preprocess
        puzzle = np.array([int(c) for c in puzzle_string], dtype=np.float32) / 9.0
        puzzle_tensor = torch.tensor(puzzle).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(puzzle_tensor)
            prediction = output.argmax(dim=2).squeeze(0).cpu().numpy()
        
        # Convert back to 1-9
        solution = ''.join(str(d + 1) for d in prediction)
        return solution
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {path}")


def load_kaggle_data(csv_path, max_samples=None):
    """
    Load Kaggle Sudoku dataset.
    
    Args:
        csv_path: Path to sudoku.csv
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        (puzzles, solutions) as lists of strings
    """
    import pandas as pd
    
    # Read CSV with dtype=str to preserve leading zeros
    df = pd.read_csv(csv_path, dtype=str)
    
    if max_samples:
        df = df.head(max_samples)
    
    puzzles = df['quizzes'].tolist()
    solutions = df['solutions'].tolist()
    
    return puzzles, solutions


# Example usage function
def example_usage():
    """
    Example of how to use this module.
    Run this after downloading the Kaggle dataset.
    """
    # Load data
    print("Loading data...")
    puzzles, solutions = load_kaggle_data('sudoku.csv', max_samples=50000)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_puzzles, val_puzzles, train_solutions, val_solutions = train_test_split(
        puzzles, solutions, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = SudokuDataset(train_puzzles, train_solutions)
    val_dataset = SudokuDataset(val_puzzles, val_solutions)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Create model
    model = SudokuNet(hidden_sizes=[256, 256])
    
    # Create trainer
    trainer = SudokuTrainer(model)
    trainer.setup_optimizer(lr=1e-3)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=10)
    
    # Save model
    trainer.save('sudoku_model.pth')
    
    # Test prediction
    test_puzzle = val_puzzles[0]
    prediction = trainer.predict(test_puzzle)
    print(f"Puzzle:     {test_puzzle}")
    print(f"Prediction: {prediction}")
    print(f"Solution:   {val_solutions[0]}")


if __name__ == "__main__":
    example_usage()

