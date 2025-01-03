import pygame
import numpy as np
import start_screen  # Import start screen functionality
from connect_four import ConnectFour
import torch
from alphazeromcts import AlphaZeroModel, mcts_search
#import mcts_test
#from improved import AlphaZeroModel, mcts_search
#from new_model import AlphaZeroModel, mcts_search

# Configuration
ROWS = 6
COLS = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLS * SQUARESIZE
height = (ROWS+1) * SQUARESIZE
size = (width, height)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

game = ConnectFour()


# In 4connect.py
model = AlphaZeroModel()  # Create model instance
model.load_state_dict(torch.load("2025_alphazero_connect4.pth"))  # Load weights into model


# Erstelle das Spielfeld 
def create_board():
    board = game.board 
    return board

# Anzeige des Spielfeldes
# Spielfeld zeichnen
def draw_board(board):
    for r in range(ROWS):
        for c in range(COLS):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, (r+1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int((r+1) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()




# Finde die nächste freie Zeile
def is_valid_location(board, col):
    return board[ROWS-1][col] == 0

# Lege den Stein in die nächste freie Zeile
def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == 0:
            return r

# Setze den Zug
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Siegprüfung
def winning_move(board, piece):
    # Horizontal prüfen
    for c in range(COLS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Vertikal prüfen
    for c in range(COLS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Diagonal prüfen (positive Steigung)
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Diagonal prüfen (negative Steigung)
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True


# Pygame setup
pygame.init()
screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("monospace", 50)

# Get opponent type from start screen
opponent_type = start_screen.start_game()

# Initialize game variables
board = create_board()
game_over = False
turn = 0

draw_board(board)
pygame.display.update()


# Helper function to check if the board is full
def is_board_full(board):
    return all(board[game.ROWS-1][col] != 0 for col in range(game.COLS))

def handle_move(col, game, turn):
    global game_over
    if game.is_valid_move(col):
        row = game.get_next_open_row(col)
        game.make_move(col, turn + 1)
        if game.winning_move(row, col):
            label = myfont.render(f"Player {turn + 1} wins!!", 1, RED if turn == 0 else YELLOW)
            screen.blit(label, (40, 10))
            pygame.display.update()
            game_over = True
        draw_board(game.board)
        switch_turn()

    # Check for tie after every move
    if is_board_full(board) and not game_over:
        label = myfont.render("It's a tie!", 1, (255, 255, 255))
        screen.blit(label, (40, 10))
        pygame.display.update()
        game_over = True


# Function to handle random player's move
def random_move(board, turn):
    if not is_board_full(board):
        col = np.random.randint(0, COLS)
        while not is_valid_location(board, col):  # Keep selecting a new column if invalid
            col = np.random.randint(0, COLS)

        row = get_next_open_row(board, col)
        drop_piece(board, row, col, 2)
        if winning_move(board, 2):
            label = myfont.render("Random opponent wins!!", 1, YELLOW)
            screen.blit(label, (40, 10))
            pygame.display.update()
            global game_over
            game_over = True
        draw_board(board)
        switch_turn()


def alphazero_move(game_instance, turn):
    """
    Function for AlphaZero agent to make a move.
    Args:
        game_instance: Instance of the ConnectFour game.
        turn: The turn indicator (used for piece dropping and turn switching).
    """
    # Get valid actions (columns with available spaces)
    valid_actions = game_instance.get_valid_moves()
    
    # Perform MCTS search considering only valid actions
    visit_counts = mcts_search(model, game_instance, simulations=150, c_puct=1.0)
    
    # Select the action (column) with the most visits (best action)
    best_action = max(visit_counts, key=visit_counts.get)

    # Ensure we are making a valid move
    if best_action not in valid_actions:
        print(f"Invalid move selected by MCTS: {best_action}. Valid actions: {valid_actions}")
        return
    
    # Get the next open row and drop the piece for the AlphaZero agent
    row = game_instance.get_next_open_row(best_action)
    drop_piece(game_instance.board, row, best_action, 2)  # AlphaZero uses '2' as its piece ID

    # Check for a win
    if winning_move(game_instance.board, 2):
        label = myfont.render("AlphaZero wins!!", 1, YELLOW)
        screen.blit(label, (40, 10))
        pygame.display.update()
        global game_over
        game_over = True

    # Update the board and display
    draw_board(game_instance.board)
    switch_turn()




# Switch turn function
def switch_turn():
    global turn
    turn = (turn + 1) % 2



# Main game loop
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        # Mouse motion for drag effect
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == 0:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
            else:
                pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()

        # Player's turn handling
        if event.type == pygame.MOUSEBUTTONDOWN:
            posx = event.pos[0]
            col = int(np.floor(posx / SQUARESIZE))

            # Execute turn based on opponent type
            if opponent_type == "human_vs_human":
                handle_move(col, game, turn)
            elif opponent_type == "human_vs_random" and turn == 0:
                handle_move(col, game, turn)  # Human move
                random_move(game.board, turn)  # Random player's move
            elif opponent_type == "human_vs_qlearning" and turn == 0:
                print("No such model implemented")
                #handle_move(col, game, turn)
                #qlearning_move(q_agent, game, turn)
            elif opponent_type == "human_vs_alphazero" and turn == 0:
                handle_move(col, game, turn)
                alphazero_move(game, 2)  # AlphaZero move (not yet implemented)

        if game_over:
            pygame.time.wait(3000)  
