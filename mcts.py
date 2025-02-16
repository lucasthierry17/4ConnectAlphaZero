import torch
import torch.nn.functional as F
import numpy as np
import math


class Node:
    """
    A node in the Monte Carlo Tree Search tree.

    Each node represents a game state and stores statistics about
    visits and scores for this state, along with references to
    parent and child nodes.
    """

    def __init__(self, parent, state, to_play, game, config):
        """
        Initialize a new node in the MCTS tree.

        Args:
            parent (Node): Parent node in the tree (None for root)
            state (numpy.ndarray): Game state this node represents
            to_play (int): Player to move in this state (1 or -1)
            game (ConnectFour): Game instance for rules and mechanics
            config (Config): Configuration parameters for MCTS
        """
        self.parent = parent
        self.state = state
        self.to_play = to_play
        self.config = config
        self.game = game
        self.prob = 0  # Prior probability from neural network
        self.children = {}  # Action -> Node mapping
        self.n_visits = 0  # Number of visits to this node
        self.total_score = 0  # Sum of values from all visits

    def check_winning_move(self, state, action, player):
        """
        Check if a move leads to an immediate win.

        Args:
            state (numpy.ndarray): Current game state
            action (int): Move to check
            player (int): Player making the move

        Returns:
            bool: True if the move wins the game
        """
        next_state = self.game.get_next_state(state, action, player)
        return self.game.evaluate(next_state) == player

    def get_safe_actions(self):
        """
        Get moves ranked by safety, prioritizing:
        1. Winning moves
        2. Moves that don't allow opponent to win next turn
        3. Forced moves (when all moves allow opponent to win)

        Returns:
            list: Prioritized list of valid actions
        """
        valid_moves = self.game.get_valid_moves(self.state)
        if not valid_moves:
            return []

        # First priority: immediate winning moves
        for action in valid_moves:
            if self.check_winning_move(self.state, action, self.to_play):
                return [action]

        # Categorize remaining moves as safe or forced
        safe_moves = []
        forced_moves = []

        for action in valid_moves:
            next_state = self.game.get_next_state(self.state, action, self.to_play)
            opponent_can_win = False

            # Check if opponent can win after this move
            for opp_action in self.game.get_valid_moves(next_state):
                if self.check_winning_move(next_state, opp_action, -self.to_play):
                    opponent_can_win = True
                    break

            if opponent_can_win:
                forced_moves.append(action)
            else:
                safe_moves.append(action)

        return safe_moves if safe_moves else forced_moves

    def expand(self):
        """
        Create child nodes for all valid actions.
        Prioritizes safe moves using get_safe_actions().
        """
        valid_moves = self.get_safe_actions()

        if not valid_moves:
            self.total_score = self.game.evaluate(self.state)
            return

        for action in valid_moves:
            child_state = -self.game.get_next_state(self.state, action)
            self.children[action] = Node(
                self, child_state, -self.to_play, self.game, self.config
            )

    def select_child(self):
        """
        Select the most promising child node using PUCT algorithm
        with additional safety bonuses.

        Returns:
            Node: Selected child node
        """
        best_puct = -np.inf
        best_child = None
        best_action = None

        for action, child in self.children.items():
            puct = self.calculate_puct(child)

            # Add safety bonus for moves that don't give opponent a win
            next_state = self.game.get_next_state(self.state, action, self.to_play)
            opponent_can_win = any(
                self.check_winning_move(next_state, opp_action, -self.to_play)
                for opp_action in self.game.get_valid_moves(next_state)
            )

            if not opponent_can_win:
                puct += 0.1  # Safety bonus

            if puct > best_puct:
                best_puct = puct
                best_child = child
                best_action = action

        return best_child

    def calculate_puct(self, child):
        """
        Calculate the PUCT score for a child node.
        PUCT balances exploitation (node value) with exploration (visit count).

        Args:
            child (Node): Child node to evaluate

        Returns:
            float: PUCT score
        """
        exploitation_term = 1 - (child.get_value() + 1) / 2
        exploration_term = child.prob * math.sqrt(self.n_visits) / (child.n_visits + 1)
        return exploitation_term + self.config.exploration_constant * exploration_term

    def backpropagate(self, value):
        """
        Update node statistics with the result of a simulation.

        Args:
            value (float): Value to backpropagate (-1 to 1)
        """
        self.total_score += value
        self.n_visits += 1
        if self.parent:
            self.parent.backpropagate(-value)

    def is_leaf(self):
        """Check if node is a leaf (has no children)."""
        return not self.children

    def is_terminal(self):
        """Check if node represents a terminal game state."""
        return self.n_visits > 0 and not self.children

    def get_value(self):
        """Get the average value of this node from all visits."""
        return 0 if self.n_visits == 0 else self.total_score / self.n_visits


class MCTS:
    """
    Monte Carlo Tree Search implementation with neural network guidance
    and safety-aware move selection.
    """

    def __init__(self, network, game, config):
        """
        Initialize MCTS with neural network for move evaluation.

        Args:
            network (torch.nn.Module): Neural network for state evaluation
            game (ConnectFour): Game instance for rules and mechanics
            config (Config): Configuration parameters
        """
        self.network = network
        self.game = game
        self.config = config

    def search(self, state, total_iterations, temperature=None):
        """
        Perform MCTS search from a given state.

        Args:
            state (numpy.ndarray): Starting game state
            total_iterations (int): Number of MCTS iterations to perform
            temperature (float, optional): Temperature for move selection

        Returns:
            tuple: (selected_action, root_node)
        """
        # Initialize root node
        root = Node(None, state, 1, self.game, self.config)
        valid_moves = self.game.get_valid_moves(state)

        # Get initial policy and value from neural network
        state_tensor = torch.as_tensor(
            self.game.encode_state(state),
            dtype=torch.float32,
            device=self.config.device,
        ).unsqueeze(0)

        with torch.no_grad():
            value, logits = self.network(state_tensor)

        # Calculate action probabilities with Dirichlet noise
        action_probs = F.softmax(logits.view(self.game.cols), dim=0).cpu().numpy()
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.game.cols)
        action_probs = (
            1 - self.config.dirichlet_eps
        ) * action_probs + self.config.dirichlet_eps * noise

        # Convert valid_moves to numpy array if needed
        if isinstance(valid_moves, list):
            valid_moves = np.array(valid_moves)

        # Normalize probabilities for valid moves
        action_probs = action_probs[valid_moves]
        action_probs /= np.sum(action_probs)

        # Create child nodes with initial probabilities
        for action, prob in zip(valid_moves, action_probs):
            child_state = -self.game.get_next_state(state, action)
            root.children[action] = Node(root, child_state, -1, self.game, self.config)
            root.children[action].prob = prob

        root.total_score = value.item()
        root.n_visits += 1

        # Main MCTS loop
        for _ in range(total_iterations):
            current_node = root

            # Selection phase
            while not current_node.is_leaf():
                current_node = current_node.select_child()

            # Expansion and evaluation phase
            if not current_node.is_terminal():
                current_node.expand()
                if current_node.children:
                    # Evaluate new nodes with neural network
                    state_tensor = (
                        torch.tensor(
                            self.game.encode_state(current_node.state),
                            dtype=torch.float,
                        )
                        .unsqueeze(0)
                        .to(self.config.device)
                    )

                    with torch.no_grad():
                        value, logits = self.network(state_tensor)
                        value = value.item()

                    # Update child probabilities
                    valid_moves = self.game.get_valid_moves(current_node.state)
                    if isinstance(valid_moves, list):
                        valid_moves = np.array(valid_moves)
                    action_probs = (
                        F.softmax(logits.view(self.game.cols)[valid_moves], dim=0)
                        .cpu()
                        .numpy()
                    )

                    for child, prob in zip(
                        current_node.children.values(), action_probs
                    ):
                        child.prob = prob
                else:
                    value = self.game.evaluate(current_node.state)
            else:
                value = self.game.evaluate(current_node.state)

            # Backpropagation phase
            current_node.backpropagate(value)

        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        return self.select_action(root, temperature), root

    def select_action(self, root, temperature=None):
        """
        Select an action based on visit counts and safety considerations.

        Args:
            root (Node): Root node of the MCTS tree
            temperature (float, optional): Temperature parameter for move selection

        Returns:
            int: Selected action
        """
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        action_counts = {key: val.n_visits for key, val in root.children.items()}

        if not action_counts:
            return None

        # First priority: winning moves
        for action in action_counts:
            if root.check_winning_move(root.state, action, root.to_play):
                return action

        # Second priority: safe moves
        safe_moves = {}
        for action in action_counts:
            next_state = root.game.get_next_state(root.state, action, root.to_play)
            opponent_moves = root.game.get_valid_moves(next_state)

            opponent_can_win = any(
                root.check_winning_move(next_state, opp_action, -root.to_play)
                for opp_action in opponent_moves
            )

            if not opponent_can_win:
                safe_moves[action] = action_counts[action]

        # Use safe moves if available
        if safe_moves:
            action_counts = safe_moves

        # Select action based on temperature
        if temperature == 0 or len(action_counts) == 1:
            return max(action_counts, key=action_counts.get)
        elif temperature == np.inf:
            return np.random.choice(list(action_counts.keys()))
        else:
            values = np.array(list(action_counts.values()), dtype=np.float32)
            if values.sum() == 0:
                # Use uniform distribution if all values are zero
                distribution = np.ones_like(values) / len(values)
            else:
                # Apply temperature to visit counts
                distribution = values ** (1 / temperature)
                distribution = distribution / distribution.sum()

            return np.random.choice(list(action_counts.keys()), p=distribution)
