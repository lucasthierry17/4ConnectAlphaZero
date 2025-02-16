import pygame
from connectfour import ConnectFour
from typing import List, Tuple, Optional
import torch
import numpy as np
from mcts import MCTS
from network import ResNet
from config import Config

pygame.init()

class ConnectFourGUI:
    """
    Graphical user interface for Connect Four game with AlphaZero AI opponent.
    
    Provides a visual interface for playing Connect Four against an AI model
    trained using the AlphaZero algorithm. Handles game visualization,
    user input, move animation, and AI move generation.
    """
    
    def __init__(self, model_path='models/model_weights.pth'):
        """
        Initialize the Connect Four GUI.
        
        Args:
            model_path (str): Path to the trained AlphaZero model weights
        """
        # Display settings
        self.SQUARESIZE = 100  # Size of each board cell
        self.RADIUS = self.SQUARESIZE // 2 - 5  # Radius of game pieces
        
        # Initialize game logic
        self.game = ConnectFour()
        self.width = self.game.cols * self.SQUARESIZE
        self.height = (self.game.rows + 1) * self.SQUARESIZE  # Extra row for piece animation
        
        # Initialize display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Connect Four vs AlphaZero')
        
        # Define colors
        self.BLUE = (0, 0, 255)     # Board color
        self.BLACK = (0, 0, 0)      # Not used currently
        self.RED = (255, 0, 0)      # Player 1 color
        self.YELLOW = (255, 200, 0)  # Player 2 (AI) color
        self.WHITE = (255, 255, 255) # Background color
        
        # Initialize font for messages
        self.font = pygame.font.SysFont("monospace", 75)
        
        # Initialize AI components
        self.config = Config()
        self.network = ResNet(self.game, self.config).to(self.config.device)
        self.network.load_state_dict(
            torch.load(model_path, map_location=self.config.device, weights_only=False)
        )
        self.network.eval()
        self.mcts = MCTS(self.network, self.game, self.config)
        
    def draw_board(self):
        """
        Draw the current game state.
        
        Renders the game board, including:
        - Blue background grid
        - White circular holes
        - Red and yellow game pieces
        """
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw blue board background
        for row in range(self.game.rows):
            for col in range(self.game.cols):
                # Draw board cell
                pygame.draw.rect(
                    self.screen, 
                    self.BLUE, 
                    (col * self.SQUARESIZE, 
                     (row + 1) * self.SQUARESIZE, 
                     self.SQUARESIZE, 
                     self.SQUARESIZE)
                )
                
                # Draw empty cell circle
                pygame.draw.circle(
                    self.screen,
                    self.WHITE,
                    (col * self.SQUARESIZE + self.SQUARESIZE // 2,
                     (row + 1) * self.SQUARESIZE + self.SQUARESIZE // 2),
                    self.RADIUS
                )
        
        # Draw placed pieces
        for row in range(self.game.rows):
            for col in range(self.game.cols):
                if self.game.board[row][col] == 1:
                    color = self.RED
                elif self.game.board[row][col] == 2:
                    color = self.YELLOW
                else:
                    continue
                    
                pygame.draw.circle(
                    self.screen,
                    color,
                    (col * self.SQUARESIZE + self.SQUARESIZE // 2,
                     (row + 1) * self.SQUARESIZE + self.SQUARESIZE // 2),
                    self.RADIUS
                )
                
        pygame.display.update()
        
    def display_winner(self, winner: int):
        """
        Display the game result message.
        
        Args:
            winner (int): Game result indicator
                         1 for player win
                         2 for AI win
                         0 for draw
        """
        if winner == 1:
            text = "YOU WIN!"
            color = self.RED
        elif winner == 2:
            text = "AI WINS!"
            color = self.YELLOW
        else:
            text = "DRAW!"
            color = self.BLUE
            
        label = self.font.render(text, True, color)
        rect = label.get_rect(center=(self.width // 2, self.SQUARESIZE // 2))
        self.screen.blit(label, rect)
        pygame.display.update()
        
    def get_column(self, pos_x: int) -> Optional[int]:
        """
        Convert mouse x-position to board column.
        
        Args:
            pos_x (int): Mouse x-coordinate
            
        Returns:
            Optional[int]: Column index or None if position is invalid
        """
        if pos_x < 0 or pos_x >= self.width:
            return None
        return pos_x // self.SQUARESIZE
    
    def animate_drop(self, col: int, color: Tuple[int, int, int]):
        """
        Animate a piece dropping into the selected column.
        
        Args:
            col (int): Column where piece is being dropped
            color (Tuple[int, int, int]): RGB color of the piece
        """
        for row in range(self.game.rows + 1):
            # Clear animation row
            pygame.draw.rect(
                self.screen,
                self.WHITE, 
                (0, 0, self.width, self.SQUARESIZE)
            )
            
            # Draw piece at current position
            pygame.draw.circle(
                self.screen,
                color,
                (col * self.SQUARESIZE + self.SQUARESIZE // 2,
                 row * self.SQUARESIZE + self.SQUARESIZE // 2),
                self.RADIUS
            )
                              
            pygame.display.update()
            pygame.time.wait(50)  # Animation delay
    
    def get_ai_move(self) -> int:
        """
        Get the AI's move using the AlphaZero model.
        
        Returns:
            int: Column chosen by the AI
        """
        model_state = self.game.get_state_for_model()
        action, _ = self.mcts.search(
            model_state,
            total_iterations=100,
            temperature=0.1
        )
        return action
        
    def play_game(self):
        """
        Main game loop handling player input and AI moves.
        
        Manages the game flow including:
        - Drawing the game board
        - Handling player input
        - Animating moves
        - Getting AI moves
        - Checking for game end
        """
        self.draw_board()
        game_over = False
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
                if event.type == pygame.MOUSEMOTION and not game_over:
                    # Show piece preview in animation row
                    pygame.draw.rect(
                        self.screen,
                        self.WHITE, 
                        (0, 0, self.width, self.SQUARESIZE)
                    )
                    
                    pos_x = event.pos[0]
                    color = self.RED if self.game.current_player == 1 else self.YELLOW
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (pos_x, self.SQUARESIZE // 2),
                        self.RADIUS
                    )
                    pygame.display.update()
                    
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    # Handle player's turn
                    if self.game.current_player == 1:
                        col = self.get_column(event.pos[0])
                        if col is not None and col in self.game.get_valid_moves():
                            # Player move
                            self.animate_drop(col, self.RED)
                            self.game.make_move(col)
                            self.draw_board()
                            
                            if self.game.check_winner() is not None:
                                self.display_winner(self.game.check_winner())
                                game_over = True
                                break
                            
                            # AI move
                            ai_col = self.get_ai_move()
                            self.animate_drop(ai_col, self.YELLOW)
                            self.game.make_move(ai_col)
                            self.draw_board()
                            
                            if self.game.check_winner() is not None:
                                self.display_winner(self.game.check_winner())
                                game_over = True
            
            if game_over:
                pygame.time.wait(5000)  # Display final state

if __name__ == "__main__":
    gui = ConnectFourGUI()
    gui.play_game()
