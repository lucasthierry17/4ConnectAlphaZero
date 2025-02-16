import torch
from torch import nn


class ResNet(nn.Module):
    """
    ResNet neural network architecture for Connect Four.

    This network uses a residual architecture with two heads:
    - Policy head: Predicts move probabilities
    - Value head: Evaluates board position

    The network processes the board state through multiple residual blocks
    and then splits into separate policy and value prediction pathways.
    """

    def __init__(self, game, config):
        """
        Initialize the ResNet model.

        Args:
            game (ConnectFour): Game instance for board dimensions
            config (Config): Network configuration parameters
                           containing n_filters and n_res_blocks
        """
        super().__init__()
        self.board_size = (game.rows, game.cols)
        n_actions = game.cols  # Number of possible moves (columns)
        n_filters = config.n_filters

        # Main convolutional body of the network
        self.base = ConvBase(config)

        # Policy head predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                n_filters // 4 * self.board_size[0] * self.board_size[1], n_actions
            ),
        )

        # Value head evaluates board position
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters // 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters // 32 * self.board_size[0] * self.board_size[1], 1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, rows, cols)
                            representing the board state

        Returns:
            tuple: (value, policy)
                - value: Board evaluation in range [-1, 1]
                - policy: Move probabilities (logits) for each column
        """
        x = self.base(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy


class ConvBase(nn.Module):
    """
    Convolutional base of the network with residual blocks.

    Processes the input board state through an initial convolution
    followed by multiple residual blocks.
    """

    def __init__(self, config):
        """
        Initialize the convolutional base.

        Args:
            config (Config): Network configuration parameters
                           containing n_filters and n_res_blocks
        """
        super().__init__()
        n_filters = config.n_filters
        n_res_blocks = config.n_res_blocks

        # Initial convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(n_filters) for _ in range(n_res_blocks)]
        )

    def forward(self, x):
        """
        Forward pass through the convolutional base.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, rows, cols)

        Returns:
            torch.Tensor: Processed feature maps
        """
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Each block consists of two convolution layers with batch normalization
    and ReLU activation. The input is added to the output through a skip
    connection.
    """

    def __init__(self, n_filters):
        """
        Initialize a residual block.

        Args:
            n_filters (int): Number of convolutional filters
        """
        super().__init__()
        self.conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with skip connection added
        """
        # First convolution block
        output = self.conv_1(x)
        output = self.batch_norm_1(output)
        output = self.relu(output)

        # Second convolution block
        output = self.conv_2(output)
        output = self.batch_norm_2(output)

        # Add skip connection and apply ReLU
        return self.relu(output + x)
