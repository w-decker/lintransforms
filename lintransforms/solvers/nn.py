import torch.nn as nn
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from ..utils import Solver

class _MLP(nn.Module):
    """
    Backend implementation of a Multi-Layer Perceptron (MLP) using PyTorch.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list = [64, 32], activation=nn.ReLU):
        super(_MLP, self).__init__()
        layers = nn.ModuleList()
        in_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@dataclass
class MLP(Solver):
    """
    Multi-Layer Perceptron (MLP) solver for approximating transformations.
    """
    input_dim: int
    output_dim: int
    hidden_layers: list = (64, 32)
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    verbose: bool = False
    activation: nn.Module = nn.ReLU
    loss_fn : nn.Module = nn.MSELoss

    def solve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = _MLP(self.input_dim, self.output_dim, list(self.hidden_layers), self.activation)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.loss_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for xb, yb in dataloader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"[{epoch+1}/{self.epochs}] Loss: {loss.item():.4f}")

        self.model = model
        return (torch.tensor(x).float()).detach().numpy()

    @property
    def name(self) -> str:
        return f"MLP(hidden={self.hidden_layers}, lr={self.lr})"
