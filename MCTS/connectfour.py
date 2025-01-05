from typing import List, Optional
import numpy as np

class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.last_move = None

    def make_move(self, col: int) -> bool:
        if col < 0 or col >= self.cols:
            return False
        
        # Find the first empty row in the selected column
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                return True
        return False

    def get_valid_moves(self) -> List[int]:
        return [col for col in range(self.cols) if self.board[0][col] == 0]

    def check_winner(self) -> Optional[int]:
        if self.last_move is None:
            return None

        row, col = self.last_move
        player = self.board[row][col]

        # Check horizontal
        def check_line(r: int, c: int, dr: int, dc: int) -> int:
            count = 0
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r][c] == player):
                count += 1
                r += dr
                c += dc
            return count

        # Check all directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = (check_line(row, col, dr, dc) + 
                    check_line(row, col, -dr, -dc) - 1)
            if count >= 4:
                return player

        # Check draw
        if len(self.get_valid_moves()) == 0:
            return 0

        return None

    def check_winning_move(self, player: int) -> Optional[int]:
        """Check if there's a winning move available for the given player."""
        for col in self.get_valid_moves():
            # Try the move
            temp_game = self.clone()
            temp_game.current_player = player
            temp_game.make_move(col)
            
            # Check if it's a winning move
            if temp_game.check_winner() == player:
                return col
        
        return None

    def clone(self) -> 'ConnectFour':
        new_game = ConnectFour()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        return new_game