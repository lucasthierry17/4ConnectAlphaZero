import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
import os
from connect_four import ConnectFour


class AlphaZeroModel(nn.Module):
    def __init__(self, rows=6, cols=7, num_actions=7):
        super(AlphaZeroModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_fc1 = nn.Linear(128 * rows * cols, 256)
        self.policy_fc2 = nn.Linear(256, num_actions)

        # Value head
        self.value_fc1 = nn.Linear(128 * rows * cols, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1, 6, 7)  # Input shape (batch, channels, rows, cols)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x_flat = x.view(x.size(0), -1)

        # Policy head
        policy = F.relu(self.policy_fc1(x_flat))
        policy = F.log_softmax(self.policy_fc2(policy), dim=1)

        # Value head
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(x_flat))))
        
        return policy, value

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0, action=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.action = action
        self.children = []
        self.visits = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct):
        if not self.children:
            return None
        ucb_scores = [child.ucb_score(c_puct) for child in self.children]
        return self.children[np.argmax(ucb_scores)]

    def add_child(self, action, next_state, prior):
        child = MCTSNode(next_state, parent=self, prior=prior, action=action)
        self.children.append(child)

    def ucb_score(self, c_puct):
        if self.visits == 0:
            return float('inf')
        return self.value() + c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def backup(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)



# Keep most of the code the same, but modify mcts_search and run_simulation
def mcts_search(model, game, simulations, c_puct=0.9):
    root = MCTSNode(state=game.get_board_state())
    for _ in range(simulations):
        node = root
        temp_game = game.copy()
        
        # Selection and expansion
        while node.is_fully_expanded() and not temp_game.done:
            node = node.select_child(c_puct)
            temp_game.play_move(node.action)
        
        if not temp_game.done:
            state_tensor = torch.FloatTensor(temp_game.get_board_state()).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = model(state_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
            
            valid_moves = [col for col in range(temp_game.COLS) if temp_game.is_valid_move(col)]
            for move in valid_moves:
                next_state = temp_game.get_next_state(move)
                node.add_child(move, next_state, policy[move])
            
            # Backup
            value = temp_game.get_reward() if temp_game.done else value.item()
            node.backup(value)

    # Get visit counts
    visit_counts = {child.action: child.visits for child in root.children}
    return visit_counts


def self_play(model, game, simulations=300):
    state = game.get_board_state()
    states, mcts_policies, rewards = [], [], []

    done = False
    while not done:
        visit_counts = mcts_search(model, game, simulations)
        action_probs = np.zeros(game.COLS)
        for action, count in visit_counts.items():
            action_probs[action] = count
        total = sum(action_probs)
        if total > 0:
            action_probs /= total
        else:
            action_probs = np.ones_like(action_probs) / len(action_probs)  # Uniform probabilities

        
        states.append(state)
        mcts_policies.append(action_probs)

        action = np.random.choice(len(action_probs), p=action_probs)
        state, reward, done = game.step(action)

        rewards.append(reward)

    return states, mcts_policies, rewards

def train(model, optimizer, data):
    states, policies, values = zip(*data)
    states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1)
    policies_tensor = torch.FloatTensor(np.array(policies))  # Ensure policies is a single ndarray
    values_tensor = torch.FloatTensor(values)

    total_loss = 0.0
    for _ in range(20):  # Multiple epochs over the same data
        optimizer.zero_grad()
        pred_policies, pred_values = model(states_tensor)
        
        policy_loss = -torch.sum(policies_tensor * pred_policies) / len(states)
        value_loss = torch.mean((values_tensor - pred_values.squeeze()) ** 2)
        loss = policy_loss + value_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return total_loss / 1  # Average loss

    


def log_training_metrics(epoch, losses, rewards):
    avg_loss = sum(losses) / len(losses)
    avg_reward = sum(rewards) / len(rewards)
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")


if __name__ == "__main__":
    # Load the model if it already exists or start from new
    model = AlphaZeroModel()
    if os.path.exists("alphazero_connect4.pth"):
        model.load_state_dict(torch.load("alphazero_connect4.pth"))
        print("Model loaded from alphazero_connect4.pth")
    else:
        print("No saved model found. Starting fresh.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):  # Total training iterations
        game = ConnectFour()
        states, policies, rewards = self_play(model, game)
        data = list(zip(states, policies, rewards))
        loss = train(model, optimizer, data)

        log_training_metrics(epoch, [loss], rewards)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"2025_alphazero_connect4_epoch{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

    torch.save(model.state_dict(), "2025_alphazero_connect4.pth")
    print("Final mode saved to 2025_alphazero_connect4.pth")
