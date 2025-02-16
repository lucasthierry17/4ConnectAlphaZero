import torch
import torch.nn as nn
import numpy as np
from mcts import MCTS

class AlphaZero:
    """
    Implementation of the AlphaZero training algorithm for Connect Four.
    
    This class manages the training process including:
    - Self-play game generation
    - Experience replay memory
    - Neural network training
    - Model saving/loading
    """
    
    def __init__(self, game, config):
        """
        Initialize AlphaZero trainer.
        
        Args:
            game (ConnectFour): Game environment
            config (Config): Training configuration parameters
        """
        self.game = game
        self.config = config
        self.network = None
        self.mcts = None
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

    def initialize_training(self, network):
        """
        Setup training components including network, optimizer and memory buffer.
        
        Args:
            network (ResNet): Neural network model to train
        """
        self.network = network
        self.mcts = MCTS(self.network, self.game, self.config)
        
        # Initialize optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=0.0001
        )

        # Initialize experience replay memory
        state_shape = self.game.encode_state(self.game.reset()).shape
        self.max_memory = self.config.minibatch_size * self.config.n_minibatches
        
        # Allocate memory buffers on device
        self.state_memory = torch.zeros(self.max_memory, *state_shape).to(self.config.device)
        self.value_memory = torch.zeros(self.max_memory, 1).to(self.config.device)
        self.policy_memory = torch.zeros(self.max_memory, self.game.cols).to(self.config.device)
        
        self.current_memory_index = 0
        self.memory_full = False

    def self_play(self):
        """
        Play a single game against itself using MCTS to generate training data.
        
        For each game state:
        1. Uses MCTS to get action probabilities and state value
        2. Stores state, value and policy in replay memory
        3. Trains network when memory buffer is full
        """
        state = self.game.reset()
        done = False

        while not done:
            # Get action and game tree from MCTS
            action, root = self.mcts.search(state, self.config.mcts_iterations)
            value = root.get_value()
            
            # Calculate policy from visit counts
            visits = np.zeros(self.game.cols)
            for child_action, child in root.children.items():
                visits[child_action] = child.n_visits
            visits /= np.sum(visits)

            # Store experience in memory
            self.append_to_memory(state, value, visits)

            # Train if memory is full
            if self.memory_full:
                self.learn()

            # Make move and prepare next state
            state, _, done = self.game.step(state, action)
            state = -state  # Flip state for next player

    def append_to_memory(self, state, value, visits):
        """
        Add state and its augmented version to replay memory.
        
        Stores both the original state and its horizontal flip to increase
        training data diversity.
        
        Args:
            state (numpy.ndarray): Game state
            value (float): State value from MCTS (-1 to 1)
            visits (numpy.ndarray): Action probabilities from MCTS
        """
        # Create original and flipped versions
        encoded_state = np.array(self.game.encode_state(state))
        encoded_state_augmented = np.array(self.game.encode_state(state[:, ::-1]))

        # Stack states and corresponding data
        states_stack = np.stack((encoded_state, encoded_state_augmented))
        visits_stack = np.stack((visits, visits[::-1]))

        # Convert to tensors and move to device
        state_tensor = torch.tensor(states_stack, dtype=torch.float).to(self.config.device)
        visits_tensor = torch.tensor(visits_stack, dtype=torch.float).to(self.config.device)
        value_tensor = torch.tensor([value, value], dtype=torch.float).to(self.config.device).unsqueeze(1)

        # Store in memory buffers
        idx = self.current_memory_index
        self.state_memory[idx:idx + 2] = state_tensor
        self.value_memory[idx:idx + 2] = value_tensor
        self.policy_memory[idx:idx + 2] = visits_tensor

        # Update memory index and full flag
        self.current_memory_index = (self.current_memory_index + 2) % self.max_memory
        if self.current_memory_index == 0 or self.current_memory_index == 1:
            self.memory_full = True

    def learn(self):
        """
        Train the neural network on a batch of experiences from memory.
        
        Performs multiple rounds of training on randomly sampled minibatches.
        Updates both policy and value predictions using appropriate loss functions.
        
        Returns:
            tuple: (policy_loss, value_loss, total_loss) averaged over all batches
        """
        self.network.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        n_batches = 0

        # Randomly sample from memory for training
        indices = np.arange(self.max_memory)
        np.random.shuffle(indices)

        # Train on multiple minibatches
        for batch_idx in range(self.config.n_minibatches):
            start_idx = batch_idx * self.config.minibatch_size
            end_idx = start_idx + self.config.minibatch_size
            batch_indices = indices[start_idx:end_idx]

            # Get batch of training data
            states = self.state_memory[batch_indices]
            value_targets = self.value_memory[batch_indices]
            policy_targets = self.policy_memory[batch_indices]

            # Forward pass
            value_preds, policy_logits = self.network(states)

            # Calculate losses
            policy_loss = self.loss_cross_entropy(policy_logits, policy_targets)
            value_loss = self.loss_mse(value_preds.view(-1), value_targets.view(-1))
            batch_loss = policy_loss + value_loss

            # Backward pass and optimization
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += batch_loss.item()
            n_batches += 1

        self.memory_full = False
        self.network.eval()

        # Return average losses
        return (total_policy_loss / n_batches, 
                total_value_loss / n_batches, 
                total_loss / n_batches)

    def save_model(self, path):
        """
        Save the neural network weights to a file.
        
        Args:
            path (str): Path to save the model weights
        """
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        """
        Load neural network weights from a file.
        
        Args:
            path (str): Path to the saved model weights
        """
        self.network.load_state_dict(
            torch.load(path, map_location=self.config.device)
        )
        self.network.eval()