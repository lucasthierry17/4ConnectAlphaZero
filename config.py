import torch

class Config:
    def __init__(self, n_simulations=50):
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.n_filters = 128
        self.n_res_blocks = 8
        self.exploration_constant = 2
        self.temperature = 1.25
        self.dirichlet_alpha = 1.
        self.dirichlet_eps = 0.25
        self.learning_rate = 0.001
        self.minibatch_size = 128
        self.n_minibatches = 4
        self.mcts_iterations = n_simulations  # Number of MCTS iterations per move