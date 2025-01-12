import numpy as np
import random
import math
from typing import List, Optional
import time
from connectfour import ConnectFour

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: List[Node] = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_valid_moves()

    def select_child(self) -> 'Node':
        # UCB1 formula for node selection
        C = 1.41  # Exploration parameter
        return max(self.children, key=lambda c: c.wins/c.visits + 
                  C * math.sqrt(2 * math.log(self.visits) / c.visits))

    def add_child(self, move, state) -> 'Node':
        child = Node(state, self, move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float):
        self.visits += 1
        self.wins += result

def get_smart_move(state: ConnectFour) -> Optional[int]:
    """Helper function to get the best strategic move"""
    current_player = state.current_player
    opponent = 3 - current_player
    
    # First priority: Check for winning move
    winning_move = state.check_winning_move(current_player)
    if winning_move is not None:
        return winning_move
        
    # Second priority: Block opponent's winning move
    blocking_move = state.check_winning_move(opponent)
    if blocking_move is not None:
        return blocking_move
    
    # Third priority: Avoid moves that give opponent winning opportunity
    valid_moves = state.get_valid_moves()
    safe_moves = []
    
    for move in valid_moves:
        # Try the move
        test_state = state.clone()
        test_state.make_move(move)
        
        # Check if this move would give opponent a winning move
        opponent_winning_move = test_state.check_winning_move(opponent)
        if opponent_winning_move is None:
            safe_moves.append(move)
    
    return random.choice(safe_moves) if safe_moves else random.choice(valid_moves)


def detect_block_specific_horizontal_threat(state: ConnectFour, current_player: int) -> Optional[int]:
    """
    Detect and block specific horizontal threats in the bottom row:
    - (0, opponent, 0, opponent, 0)
    - (0, opponent, opponent, 0, 0)
    - (0, 0, opponent, opponent, 0)
    """
    opponent = 3 - current_player
    cols = state.board.shape[1]
    bottom_row = state.board[-1]  # Get the bottom row of the board
    valid_moves = state.get_valid_moves()

    # Check for the specified patterns in the bottom row
    for col in range(cols - 4):
        window = list(bottom_row[col:col + 5])

        if window == [0, opponent, 0, opponent, 0]:  # Pattern 1
            if col in valid_moves:
                return col
            if col + 2 in valid_moves:
                return col + 2
            if col + 4 in valid_moves:
                return col + 4

        if window == [0, opponent, opponent, 0, 0]:  # Pattern 2
            if col in valid_moves:
                return col
            if col + 3 in valid_moves:
                return col + 3

        if window == [0, 0, opponent, opponent, 0]:  # Pattern 3
            if col + 1 in valid_moves:
                return col + 1
            if col + 4 in valid_moves:
                return col + 4

    return None


class MCTS:
    def __init__(self, state: ConnectFour):
        self.root = Node(state)
        self.simulation_time = 2.5  # seconds

    def get_best_move(self) -> int:
        state = self.root.state
        
        # First, check for a winning move
        winning_move = state.check_winning_move(state.current_player)
        if winning_move is not None:
            time.sleep(2.0)
            return winning_move
            
        # Then, check if we need to block opponent's winning move
        opponent = 3 - state.current_player
        blocking_move = state.check_winning_move(opponent)
        if blocking_move is not None:
            time.sleep(2.0)
            return blocking_move

        # Check for moves that don't give opponent a winning opportunity
        valid_moves = state.get_valid_moves()
        safe_moves = []

        # Inside MCTS.get_best_move()
        block_two_in_a_row_threat = detect_block_specific_horizontal_threat(state, state.current_player)
        if block_two_in_a_row_threat is not None:
            time.sleep(2.0)
            return block_two_in_a_row_threat

        
        for move in valid_moves:
            test_state = state.clone()
            test_state.make_move(move)
            
            # Check if this move would give opponent a winning move
            opponent_winning_move = test_state.check_winning_move(opponent)
            if opponent_winning_move is None:
                safe_moves.append(move)
        
        # Update root's untried moves to only include safe moves if available
        if safe_moves:
            self.root.untried_moves = safe_moves

        # If no immediate winning or blocking moves, use MCTS
        end_time = time.time() + self.simulation_time
        
        while time.time() < end_time:
            node = self.root
            state = node.state.clone()

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                state.make_move(node.move)

            # Expansion
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                state.make_move(move)
                node = node.add_child(move, state)

            # Simulation - now using smart moves instead of pure random
            while state.check_winner() is None:
                smart_move = get_smart_move(state)
                if smart_move is None:
                    break
                state.make_move(smart_move)

            # Backpropagation
            winner = state.check_winner()
            while node is not None:
                if winner == 0:  # Draw
                    result = 0.5
                else:
                    result = 1.0 if winner == self.root.state.current_player else 0.0
                node.update(result)
                node = node.parent

        # Select best move from safe moves with highest visit count
        best_child = max(self.root.children, 
                        key=lambda c: c.visits)
        return best_child.move