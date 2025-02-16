from connectfour import ConnectFour
from network import ResNet
from config import Config
from train import AlphaZero
import os
from datetime import datetime


def train_model(
    n_epochs=50, games_per_epoch=100, n_simulations=50, base_save_path="models"
):
    """
    Train an AlphaZero model for Connect Four.

    Args:
        n_epochs (int): Number of training epochs to run
        games_per_epoch (int): Number of self-play games per epoch
        n_simulations (int): Number of MCTS simulations per move
        base_save_path (str): Directory path to save model checkpoints

    Returns:
        str: Path to the final trained model
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    # Create a timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_save_path, f"run_{timestamp}")
    os.makedirs(run_dir)

    # Initialize game, model and trainer
    game = ConnectFour()
    config = Config(n_simulations=n_simulations)
    network = ResNet(game, config).to(config.device)
    alphazero = AlphaZero(game, config)

    # Setup training
    alphazero.initialize_training(network)

    print(
        f"Starting training for {n_epochs} epochs with {games_per_epoch} games per epoch"
    )
    print(f"Using {n_simulations} MCTS simulations per move")

    # Training loop
    for epoch in range(n_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{n_epochs}")

        # Play games and train for current epoch
        for game_num in range(games_per_epoch):
            alphazero.self_play()
            if alphazero.memory_full:
                alphazero.learn()

            # Print progress every 10 games
            if (game_num + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Game {game_num + 1}/{games_per_epoch}")

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(run_dir, f"model_epoch_{epoch + 1}.pth")
            alphazero.save_model(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    # Save final model
    final_model_path = os.path.join(run_dir, "model_final.pth")
    alphazero.save_model(final_model_path)

    print(f"\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")

    return final_model_path


if __name__ == "__main__":
    final_model = train_model(
        n_epochs=50,  # Number of training epochs
        games_per_epoch=100,  # Games to play per epoch
        n_simulations=50,  # MCTS simulations per move
        base_save_path="models",  # Directory to save models
    )
