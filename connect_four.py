import numpy as np
import copy

class ConnectFour:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.last_move = None

    def reset(self):
        """Resets the game board for a new game."""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1

    def make_move(self, col, piece):
        """Places a piece in the specified column."""
        row = self.get_next_open_row(col)
        if row is not None:
            self.board[row][col] = piece
            self.last_move = (row, col)  # Update last move here as well
            return row
        return None

    def get_last_move(self):
        """Returns the last move made (row, column)."""
        return self.last_move

    def play_move(self, col):
        """Plays a move for the current player in the specified column."""
        if self.is_valid_move(col):
            row = self.make_move(col, self.current_player)
            if self.winning_move(row, col):
                return self.current_player
            self.switch_player()
            return None
        return None

    def get_next_state(self, col):
        """Simulate a move and return the resulting board state."""
        temp_game = copy.deepcopy(self)
        if temp_game.is_valid_move(col):
            temp_game.make_move(col, temp_game.current_player)
            return temp_game.get_board_state()
        return None

    def check_three_in_row_with_open_spot(self, player):
        """Check if the given player has three in a row with an open spot."""
        rows, cols = self.board.shape

        # Check horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window = self.board[r, c:c+4]
                if (np.count_nonzero(window == player) == 3) and (np.count_nonzero(window == 0) == 1):
                    return True

        # Check vertical
        for r in range(rows - 3):
            for c in range(cols):
                window = self.board[r:r+4, c]
                if (np.count_nonzero(window == player) == 3) and (np.count_nonzero(window == 0) == 1):
                    return True

        # Check positive diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [self.board[r+i, c+i] for i in range(4)]
                if (window.count(player) == 3) and (window.count(0) == 1):
                    return True

        # Check negative diagonal
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [self.board[r-i, c+i] for i in range(4)]
                if (window.count(player) == 3) and (window.count(0) == 1):
                    return True

        return False

    def evaluate_board(self):
        """
        Evaluate the board for intermediate rewards:
        - Reward +1 for creating a winning opportunity (3 in a row with an open space).
        - Reward +1 for blocking opponent's winning chance.
        """
        reward = 0
        opponent = 3 - self.current_player  # Opponent's player ID

        # Reward for creating a winning opportunity
        if self.check_three_in_row_with_open_spot(self.current_player):
            reward += 1

        # Reward for blocking opponent's winning opportunity
        if self.check_three_in_row_with_open_spot(opponent):
            reward += 1

        return reward

    def step(self, action):
        """
        Applies the action (a move in a column) and returns the next state, reward, and if the game is over.
        Adds intermediate rewards for creating or blocking winning chances.
        """
        row = self.get_next_open_row(action)
        if row is not None:
            self.board[row][action] = self.current_player

            # Check if this move won the game
            if self.winning_move(row, action):
                return self.get_board_state(), 1, True  # Current player won

            # Check for intermediate rewards
            intermediate_reward = self.evaluate_board()

            # Check if the board is full (draw)
            if self.is_board_full():
                return self.get_board_state(), intermediate_reward, True

            # Switch player if the game continues
            self.switch_player()
            return self.get_board_state(), intermediate_reward, False

        # Invalid move (column full)
        return self.get_board_state(), -1, False

    def is_valid_move(self, col):
        """Checks if a column is not full."""
        return self.board[self.ROWS - 1][col] == 0
    
    def get_valid_moves(self):
        """Returns a list of columns where a move can be made (not full)."""
        valid_moves = []
        for col in range(self.COLS):  # Iterate over all columns
            if self.is_valid_move(col):
                valid_moves.append(col)  # Add column to valid_moves if it's not full
        return valid_moves


    def get_next_open_row(self, col):
        """Finds the next available row in a column."""
        for r in range(self.ROWS):
            if self.board[r][col] == 0:
                return r
        return None

    def switch_player(self):
        """Switches the current player."""
        self.current_player = 3 - self.current_player


    def winning_move(self, row, col):
        """Checks for a win condition in all directions."""
        piece = self.board[row][col]
        return (
            self.check_direction(row, col, 1, 0, piece) or  # Vertical
            self.check_direction(row, col, 0, 1, piece) or  # Horizontal
            self.check_direction(row, col, 1, 1, piece) or  # Diagonal /
            self.check_direction(row, col, 1, -1, piece)    # Diagonal \
        )

    def check_direction(self, row, col, dr, dc, piece):
        """Checks a specific direction (dr, dc) for four in a row."""
        count = 0
        for i in range(-3, 4):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == piece:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0
        return False

    def is_board_full(self):
        """Checks if the board is full."""
        return all(self.board[self.ROWS - 1][col] != 0 for col in range(self.COLS))

    def copy(self):
        """Returns a deep copy of the game state."""
        return copy.deepcopy(self)

    def get_board_state(self):
        """Returns a copy of the board state."""
        return self.board.copy()

    def __str__(self):
        """For printing the board in a readable format."""
        return str(np.flip(self.board, 0))
    
    def legal_moves(self):
        return [col for col in range(self.COLS) if self.board[0][col] == 0]

    
    @property
    def done(self):
        """Checks if the game is over by win or draw."""
        for col in range(self.COLS):
            if self.is_valid_move(col):  # If valid moves remain
                return False  # Game is not done if valid moves exist
        return True  # Game is over (draw or win)
    
    @property
    def terminal(self):
        terminal = np.all(self.board != 0)
        return 

