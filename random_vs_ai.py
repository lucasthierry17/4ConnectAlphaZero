from connectfour import ConnectFour
from mcts import MCTS  # Assuming your uploaded MCTS code is in mcts.py
import numpy as np

def random_bot(state):
    return np.random.choice(state.get_valid_moves())

def play_game():
    game = ConnectFour()
    mcts = MCTS(game)
    current_player = game.current_player

    while not game.check_winner():
        if game.current_player == current_player:
            move = mcts.get_best_move()
        else:
            move = random_bot(game)
        game.make_move(move)

        if game.check_winner() or not game.get_valid_moves():
            break

    return game.check_winner()

def main():
    games = 100
    mcts_wins = 0
    random_wins = 0
    draws = 0

    for _ in range(games):
        result = play_game()
        if result == 1:
            mcts_wins += 1
        elif result == 2:
            random_wins += 1
        else:
            draws += 1

    print(f"MCTS Wins: {mcts_wins}")
    print(f"Random Bot Wins: {random_wins}")
    print(f"Draws: {draws}")
    print(f"MCTS Win Rate: {mcts_wins / games * 100:.2f}%")
    print(f"Random Bot Win Rate: {random_wins / games * 100:.2f}%")
    print(f"Draw Rate: {draws / games * 100:.2f}%")

if __name__ == "__main__":
    main()
