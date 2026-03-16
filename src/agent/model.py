from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim

from constants import MODEL_FILE
from logger import get_logger

logger = get_logger(__name__)


class MLP(nn.Module):
    """Simple multi-layer perceptron with configurable depth and width."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        width: int,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            num_layers: Number of hidden layers of size ``width``.
            width: Number of units in each hidden layer.
        """
        super().__init__()

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)


class Model:
    """Wrapper around MLP with training, inference, save and load utilities."""

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        width: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the model, optionally loading weights from disk.

        Architecture is always defined by the parameters; a saved file only
        supplies weights, never the structure.

        Args:
            num_layers: Number of hidden layers for the MLP.
            width: Number of units in each hidden layer.
            learning_rate: Learning rate for Adam.
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            model_path: Optional directory containing a saved ``MODEL_FILE``.
        """
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            width=width,
        )

        if model_path and (model_path / MODEL_FILE).exists():
            model_file = model_path / MODEL_FILE
            logger.info(f"Loading trained model weights from {model_file}")
            self._load_weights(model_file)
        else:
            logger.info("Creating new model for the Agent")

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _load_weights(self, model_file: Path) -> None:
        """Load state_dict weights from disk into the already-constructed model.

        Args:
            model_file: Path to the ``.pt`` file produced by ``save_model``.
        """
        state_dict = torch.load(model_file, weights_only=True, map_location="cpu")
        self.model.load_state_dict(state_dict)

    def _predict(self, states: NDArray) -> NDArray:
        """Run inference on a batch of states.

        Args:
            states: Array of shape (batch_size, input_dim).

        Returns:
            Array of shape (batch_size, output_dim).
        """
        self.model.eval()
        with torch.no_grad():
            arr = np.asarray(states, dtype=np.float32)
            outputs = self.model(torch.from_numpy(arr))
            return outputs.cpu().numpy()

    def predict_one(self, state: NDArray) -> NDArray:
        """Predict Q-values for a single state.

        Args:
            state: 1-D array of shape (input_dim,).

        Returns:
            Array of shape (1, output_dim).
        """
        state_2d = np.asarray(state, dtype=np.float32).reshape(1, self.input_dim)
        return self._predict(state_2d)

    def predict_batch(self, states: NDArray) -> NDArray:
        """Predict Q-values for a batch of states.

        Args:
            states: Array of shape (batch_size, input_dim).

        Returns:
            Array of shape (batch_size, output_dim).
        """
        return self._predict(states)

    def train_batch(self, states: NDArray, q_sa: NDArray) -> None:
        """Train the model for one step on a batch.

        Args:
            states: Input states of shape (batch_size, input_dim).
            q_sa: Target Q-values of shape (batch_size, output_dim).
        """
        self.model.train()

        states_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
        q_sa_tensor = torch.from_numpy(np.asarray(q_sa, dtype=np.float32))

        self.optimizer.zero_grad()
        predictions = self.model(states_tensor)
        loss = self.loss_fn(predictions, q_sa_tensor)
        loss.backward()
        self.optimizer.step()

    def save_model(self, out_path: Path) -> None:
        """Save model weights to ``out_path / MODEL_FILE``.

        Saves ``state_dict`` only (portable across PyTorch versions).

        Args:
            out_path: Directory to write the model file into.
        """
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path / MODEL_FILE)
