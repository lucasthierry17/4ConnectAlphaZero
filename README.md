# 🎮 Connect Four AlphaZero

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Connect Four implementation trained using the AlphaZero algorithm, featuring self-play, neural network training, and a graphical user interface.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e178aea7-eb38-4101-b6f2-3db102fa624f" alt="Connect Four Game" width="300"/>
</p>

## ✨ Features

- 🎯 Complete Connect Four game implementation
- 🧠 AlphaZero training algorithm with self-play
- 🔄 Monte Carlo Tree Search (MCTS) for move selection
- 🎨 PyGame-based graphical interface
- 📊 ResNet neural network architecture

## 🚀 Quick Start

### Prerequisites

```bash
python 3.8+
pip
virtual environment (recommended)
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/connect-four-alphazero.git
cd connect-four-alphazero
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the AI

Run the training script to train a new model:
```bash
python main.py
```

The script will:
- Create a `models` directory
- Save model checkpoints every 5 epochs
- Save the final model after training

Default parameters:
```python
n_epochs = 50
games_per_epoch = 100
n_simulations = 50
```

### Playing Against AI

Launch the game interface:
```bash
python play.py
```

Game controls:
- 🖱️ Move mouse to position piece
- 🎯 Click to drop piece
- ❌ Close window to exit

## 📁 Project Structure

```
4ConnectAlphaZero/
├── connectfour.py    # Game mechanics
├── network.py        # Neural network
├── mcts.py           # Monte Carlo Tree Search
├── train.py         # AlphaZero algorithm
├── play.py          # Game interface
├── main.py          # Training script
├── config.py        # Configuration
└── requirements.txt  # Dependencies
```

## 🛠️ Technical Details

### AI Components

- **Neural Network**: ResNet architecture with policy and value heads
- **MCTS**: Enhanced with safety-aware move selection
- **Training**: Self-play with experience replay
- **State Representation**: 3-channel binary encoding

### Training Process

1. Self-play games generate training data
2. States and moves stored in replay memory
3. Network trained on batched experiences
4. Model checkpoints saved periodically

## ⚙️ Configuration

Key parameters in `config.py`:
```python
class Config:
    n_filters = 128
    n_res_blocks = 10
    learning_rate = 0.001
    mcts_iterations = 50
    temperature = 1.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push to branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Contact

Your Name - [@yourusername](https://twitter.com/yourusername)

Project Link: [https://github.com/yourusername/connect-four-alphazero](https://github.com/yourusername/connect-four-alphazero)

## 🙏 Acknowledgments

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/)
