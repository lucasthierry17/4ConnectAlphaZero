import numpy as np
import random
import math
from typing import List, Tuple, Optional
import time
import pygame

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

class MCTS:
    def __init__(self, state: ConnectFour):
        self.root = Node(state)
        self.simulation_time = 3.0  # seconds

    def get_best_move(self) -> int:
        state = self.root.state
        
        #First, check for a winning move
        winning_move = state.check_winning_move(state.current_player)
        if winning_move is not None:
            return winning_move
            
        # Then, check if we need to block opponent's winning move
        opponent = 3 - state.current_player
        blocking_move = state.check_winning_move(opponent)
        if blocking_move is not None:
            return blocking_move

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

            # Simulation
            while state.check_winner() is None:
                valid_moves = state.get_valid_moves()
                if not valid_moves:
                    break
                move = random.choice(valid_moves)
                state.make_move(move)

            # Backpropagation
            winner = state.check_winner()
            while node is not None:
                if winner == 0:  # Draw
                    result = 0.5
                else:
                    result = 1.0 if winner == self.root.state.current_player else 0.0
                node.update(result)
                node = node.parent

        # Select best move
        best_child = max(self.root.children, 
                        key=lambda c: c.visits)
        return best_child.move


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
        self.YELLOW = (255, 255, 0)
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
        
    def play_game(self):
        """Main game loop"""
        self.draw_board()
        game_over = False
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
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
                            print("AI is thinking...")
                            mcts = MCTS(self.game.clone())
                            ai_col = mcts.get_best_move()
                            

                            self.animate_drop(ai_col, self.YELLOW)
                            self.game.make_move(ai_col)
                            self.draw_board()
                            
                            if self.game.check_winner() is not None:
                                self.display_winner(self.game.check_winner())
                                game_over = True
            
            if game_over:
                pygame.time.wait(3000)  # Show final state for 3 seconds

if __name__ == "__main__":
    gui = ConnectFourGUI()
    gui.play_game()