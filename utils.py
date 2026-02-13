import numpy as np

def is_valid_sudoku(board):
    """
    Check if a 9x9 Sudoku board is valid.
    Board can be numpy array or list of lists.
    0 represents empty, but for a 'solution' check, we expect no 0s.
    """
    board = np.array(board).reshape(9, 9)
    
    # Check rows
    for i in range(9):
        if not is_valid_unit(board[i, :]):
            return False
            
    # Check cols
    for j in range(9):
        if not is_valid_unit(board[:, j]):
            return False
            
    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = board[i:i+3, j:j+3].flatten()
            if not is_valid_unit(box):
                return False
                
    return True

def is_valid_unit(unit):
    """Check if a row/col/box has unique numbers 1-9."""
    unit = unit[unit != 0] # Ignore 0s if checking partial
    if len(unit) == 0:
        return True
    return len(np.unique(unit)) == len(unit)

def parse_sudoku_string(s):
    """Convert 81-char string to 9x9 numpy array."""
    return np.array([int(c) for c in s], dtype=int).reshape(9, 9)

def board_to_string(board):
    """Convert 9x9 board to 81-char string."""
    return "".join(str(int(c)) for c in board.flatten())

def count_violations(board):
    """
    Count number of rule violations in a filled board.
    Useful for reward shaping.
    Returns: (row_violations, col_violations, box_violations)
    """
    board = np.array(board).reshape(9, 9)
    r_v = 0
    c_v = 0
    b_v = 0
    
    for i in range(9):
        if len(np.unique(board[i, :])) != 9: r_v += 1
        if len(np.unique(board[:, i])) != 9: c_v += 1
        
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = board[i:i+3, j:j+3].flatten()
            if len(np.unique(box)) != 9: b_v += 1
            
    return r_v, c_v, b_v
