import numpy as np
from scipy.signal import convolve2d

class ConnectFour:
    """
    A Connect Four game implementation with AlphaZero integration capabilities.
    
    This class implements the classic Connect Four game where players alternate
    dropping pieces into a 6x7 grid, trying to get four in a row horizontally,
    vertically, or diagonally. 
    """
    
    def __init__(self):
        """
        Initialize a new Connect Four game.
        
        Creates an empty 6x7 board where:
        - 0 represents empty cells
        - 1 represents player 1's pieces (typically human)
        - 2 represents player 2's pieces (typically AI)
        """
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        
    def get_valid_moves(self, state=None):
        """
        Return a list of valid columns where a piece can be played.
        
        Returns:
            list: Column indices where moves are valid (not full)
        """
        if state is None:
            state = self.board
        return [col for col in range(self.cols) if state[0][col] == 0]
    
    def make_move(self, col):
        """
        Attempt to make a move in the specified column.
        
        Args:
            col (int): Column index where the piece should be dropped
            
        Returns:
            bool: True if move was successful, False if column is full
        """
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.current_player = 3 - self.current_player  # Toggle between 1 and 2
                return True
        return False

    def get_next_state(self, state, action, to_play=1):
        """
        Calculate the next state after a move without modifying current state.
        
        Args:
            state (numpy.ndarray): Current board state
            action (int): Column where piece will be dropped
            to_play (int): Player making the move (1 or 2)
            
        Returns:
            numpy.ndarray: New board state after the move
        """
        for row in range(self.rows-1, -1, -1):
            if state[row][action] == 0:
                next_state = state.copy()
                next_state[row][action] = to_play
                return next_state
        return state

    def step(self, state, action, to_play=1):
        """
        Execute a move and return game state information.
            
        Returns:
            tuple: (next_state, reward, done)
                - next_state: New board state
                - reward: Game result (1 for win, -1 for loss, 0 for ongoing)
                - done: Whether game has ended
        """
        next_state = self.get_next_state(state, action, to_play)
        reward = self.evaluate(next_state)
        done = reward != 0 or len(self.get_valid_moves(next_state)) == 0
        return next_state, reward, done
    
    def evaluate(self, state):
        """
        Evaluate the current board state for a winner.
            
        Returns:
            int: 1 for player 1 win, -1 for player 2 win, 0 for no winner
        """
        # Convert player 2's pieces to -1 for evaluation
        board_to_check = state.copy()
        if 2 in board_to_check:
            board_to_check[state == 2] = -1

        # Check both players
        for player in [1, -1]:
            # Horizontal check
            kernel = np.ones((1, 4))
            conv = convolve2d(board_to_check == player, kernel, mode='valid')
            if (conv == 4).any():
                return player
            
            # Vertical check
            kernel = np.ones((4, 1))
            conv = convolve2d(board_to_check == player, kernel, mode='valid')
            if (conv == 4).any():
                return player
            
            # Diagonal (top-left to bottom-right)
            kernel = np.eye(4)
            conv = convolve2d(board_to_check == player, kernel, mode='valid')
            if (conv == 4).any():
                return player
            
            # Diagonal (top-right to bottom-left)
            kernel = np.fliplr(np.eye(4))
            conv = convolve2d(board_to_check == player, kernel, mode='valid')
            if (conv == 4).any():
                return player
        
        return 0
    
    def check_winner(self):
        """
        Check for a winner or draw in the current game state.
        
        Returns:
            int or None:
                - 1 for player 1 win
                - 2 for player 2 win
                - 0 for draw
                - None if game is ongoing
        """
        eval_result = self.evaluate(self.board)
        if eval_result == 1:
            return 1
        elif eval_result == -1:
            return 2
        elif len(self.get_valid_moves()) == 0:
            return 0
        return None
    
    def clone(self):
        """
        Create a deep copy of the current game state.
        
        Returns:
            ConnectFour: New instance with copied state
        """
        new_game = ConnectFour()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game
    
    def get_state_for_model(self):
        """
        Convert the current board state to the format expected by the AI model.
        
        Returns:
            numpy.ndarray: Converted board state
        """
        model_state = np.zeros_like(self.board, dtype=np.int8)
        model_state[self.board == 1] = 1     # Human pieces
        model_state[self.board == 2] = -1    # AI pieces
        return model_state

    def encode_state(self, state):
        """
        Encode the board state for neural network processing.
        
        Creates a 3-channel representation of the board state:
        - Channel 1: Player 1's pieces
        - Channel 2: Empty spaces
        - Channel 3: Player 2's pieces
        
        Args:
            state (numpy.ndarray, optional): Board state to encode.
                                           If None, uses current board.
        
        Returns:
            numpy.ndarray: Encoded state with shape (3, rows, cols)
        """
        if state is None:
            state = self.board
        encoded_state = np.stack((
            state == 1,    # Player 1's pieces
            state == 0,    # Empty spaces
            state == -1    # Player 2's pieces
        )).astype(np.float32)
        return encoded_state

    def reset(self):
        """
        Reset the game to its initial state.
        
        Returns:
            numpy.ndarray: Empty board state
        """
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return self.board