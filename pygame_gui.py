import pygame
from connectfour import ConnectFour
from typing import List, Tuple, Optional
from mcts import MCTS
import os
#from mcts_test import MCTS


class ConnectFourGUI:
    def __init__(self):
        pygame.init()
        self.SQUARESIZE = 100
        self.RADIUS = self.SQUARESIZE // 2 - 5
        
        self.game = ConnectFour()
        self.width = self.game.cols * self.SQUARESIZE
        self.height = (self.game.rows + 1) * self.SQUARESIZE  # Extra row for piece animation
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Connect Four')
        
        # Colors
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 200, 0)
        self.WHITE = (255, 255, 255)
        
        self.font = pygame.font.SysFont("monospace", 75)
        
    def draw_board(self):
        """Draw the game board"""
        # Clear the screen
        self.screen.fill(self.WHITE)
        
        # Draw the blue board
        for row in range(self.game.rows):
            for col in range(self.game.cols):
                pygame.draw.rect(self.screen, self.BLUE, 
                               (col * self.SQUARESIZE, 
                                (row + 1) * self.SQUARESIZE, 
                                self.SQUARESIZE, 
                                self.SQUARESIZE))
                
                # Draw the holes
                pygame.draw.circle(self.screen, self.WHITE,
                                 (col * self.SQUARESIZE + self.SQUARESIZE // 2,
                                  (row + 1) * self.SQUARESIZE + self.SQUARESIZE // 2),
                                 self.RADIUS)
        
        # Draw the pieces
        for row in range(self.game.rows):
            for col in range(self.game.cols):
                if self.game.board[row][col] == 1:
                    color = self.RED
                elif self.game.board[row][col] == 2:
                    color = self.YELLOW
                else:
                    continue
                    
                pygame.draw.circle(self.screen, color,
                                 (col * self.SQUARESIZE + self.SQUARESIZE // 2,
                                  (row + 1) * self.SQUARESIZE + self.SQUARESIZE // 2),
                                 self.RADIUS)
                
        pygame.display.update()
        
            
    def display_winner(self, winner: int):
        """Display the winner message"""
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
        """Convert mouse x position to board column"""
        if pos_x < 0 or pos_x >= self.width:
            return None
        return pos_x // self.SQUARESIZE
    
    def animate_drop(self, col: int, color: Tuple[int, int, int]):
        """Animate the piece dropping into position"""
        for row in range(self.game.rows + 1):
            # Clear the animation row
            pygame.draw.rect(self.screen, self.WHITE, 
                           (0, 0, self.width, self.SQUARESIZE))
            
            # Draw the piece
            pygame.draw.circle(self.screen, color,
                             (col * self.SQUARESIZE + self.SQUARESIZE // 2,
                              row * self.SQUARESIZE + self.SQUARESIZE // 2),
                             self.RADIUS)
                              
            pygame.display.update()
            pygame.time.wait(50)  # Animation speed
        
    def play_game(self):
        """Main game loop"""
        self.draw_board()
        game_over = False
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
                    
                if event.type == pygame.MOUSEMOTION and not game_over:
                    # Clear the animation row
                    pygame.draw.rect(self.screen, self.WHITE, 
                                   (0, 0, self.width, self.SQUARESIZE))
                    
                    # Draw the piece in the animation row
                    pos_x = event.pos[0]
                    color = self.RED if self.game.current_player == 1 else self.YELLOW
                    pygame.draw.circle(self.screen, color,
                                     (pos_x, self.SQUARESIZE // 2),
                                     self.RADIUS)
                    pygame.display.update()
                    
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    # Human's turn
                    if self.game.current_player == 1:
                        col = self.get_column(event.pos[0])
                        if col is not None and col in self.game.get_valid_moves():
                            self.animate_drop(col, self.RED)
                            self.game.make_move(col)
                            self.draw_board()
                            
                            if self.game.check_winner() is not None:
                                self.display_winner(self.game.check_winner())
                                game_over = True
                                break
                            
                            # AI's turn
                            mcts = MCTS(self.game.clone())
                            ai_col = mcts.get_best_move()
                            self.animate_drop(ai_col, self.YELLOW)
                            self.game.make_move(ai_col)
                            self.draw_board()
                            
                            if self.game.check_winner() is not None:
                                self.display_winner(self.game.check_winner())
                                game_over = True
            
            if game_over:
                pygame.time.wait(5000)  # Show final state for 5 seconds

if __name__ == "__main__":
    gui = ConnectFourGUI()
    gui.play_game()