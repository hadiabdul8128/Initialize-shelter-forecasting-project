"""PyTorch neural network training, inference, and portable artifact storage."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from shelter_forecasting.features import FEATURE_COLUMNS

DEFAULT_HIDDEN_SIZES = (64, 32)


class ShelterMLP(nn.Module):
    """A compact feed-forward network for next-day population change."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(width, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            width = hidden_size
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class NeuralPredictor:
    """Fitted PyTorch model and NumPy scaling values used for inference."""

    model: ShelterMLP
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    hidden_sizes: tuple[int, ...]
    dropout: float
    best_epoch: int
    training_history: pd.DataFrame

    def predict_change(self, features: pd.DataFrame) -> np.ndarray:
        values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        scaled = (values - self.feature_mean) / self.feature_scale
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(scaled.astype(np.float32))).numpy()
        return prediction * self.target_scale + self.target_mean


def train_neural_network(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    epochs: int = 500,
    patience: int = 60,
    learning_rate: float = 0.002,
    weight_decay: float = 0.0001,
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
    dropout: float = 0.05,
    random_state: int = 42,
) -> NeuralPredictor:
    """Train with chronological validation and restore the best epoch."""
    feature_mean, feature_scale = _scaling_values(x_train)
    target_mean, target_scale = _target_scaling_values(y_train)
    train_features = _scale_features(x_train, feature_mean, feature_scale)
    validation_features = _scale_features(x_validation, feature_mean, feature_scale)
    train_target = _scale_target(y_train, target_mean, target_scale)
    validation_target = _scale_target(y_validation, target_mean, target_scale)

    model = _new_model(
        len(FEATURE_COLUMNS),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        random_state=random_state,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.HuberLoss(delta=1.0)
    x_train_tensor = torch.from_numpy(train_features)
    y_train_tensor = torch.from_numpy(train_target)
    x_validation_tensor = torch.from_numpy(validation_features)
    y_validation_tensor = torch.from_numpy(validation_target)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_mae = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_prediction = model(x_train_tensor)
        train_loss = loss_function(train_prediction, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(x_validation_tensor)
            validation_mae = torch.mean(
                torch.abs(validation_prediction - y_validation_tensor)
            ).item()
        history.append(
            {
                "epoch": epoch,
                "train_huber_loss": float(train_loss.item()),
                "validation_scaled_mae": float(validation_mae),
            }
        )

        if validation_mae < best_validation_mae - 1e-5:
            best_validation_mae = validation_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return NeuralPredictor(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        best_epoch=best_epoch,
        training_history=pd.DataFrame(history),
    )


def refit_neural_network(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    epochs: int,
    learning_rate: float = 0.002,
    weight_decay: float = 0.0001,
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
    dropout: float = 0.05,
    random_state: int = 42,
) -> NeuralPredictor:
    """Refit for a fixed number of selected epochs on all available rows."""
    feature_mean, feature_scale = _scaling_values(features)
    target_mean, target_scale = _target_scaling_values(target)
    scaled_features = _scale_features(features, feature_mean, feature_scale)
    scaled_target = _scale_target(target, target_mean, target_scale)
    model = _new_model(
        len(FEATURE_COLUMNS),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        random_state=random_state,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.HuberLoss(delta=1.0)
    feature_tensor = torch.from_numpy(scaled_features)
    target_tensor = torch.from_numpy(scaled_target)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(feature_tensor)
        loss = loss_function(prediction, target_tensor)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "train_huber_loss": float(loss.item())})

    return NeuralPredictor(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        best_epoch=max(1, epochs),
        training_history=pd.DataFrame(history),
    )


def save_neural_bundle(
    predictor: NeuralPredictor,
    *,
    artifact_dir: Path,
    residual_quantiles: dict[str, float],
    trained_through: str,
) -> tuple[Path, Path]:
    """Save framework-native weights and JSON-only inference metadata."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "pytorch_model.pt"
    metadata_path = artifact_dir / "neural_metadata.json"
    torch.save(predictor.model.state_dict(), weights_path)
    metadata: dict[str, Any] = {
        "format_version": 1,
        "model_type": "PyTorch ShelterMLP",
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": predictor.feature_mean.tolist(),
        "feature_scale": predictor.feature_scale.tolist(),
        "target_mean": predictor.target_mean,
        "target_scale": predictor.target_scale,
        "hidden_sizes": list(predictor.hidden_sizes),
        "dropout": predictor.dropout,
        "best_epoch": predictor.best_epoch,
        "residual_quantiles": residual_quantiles,
        "trained_through": trained_through,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return weights_path, metadata_path


def load_neural_bundle(artifact_dir: str | Path) -> tuple[NeuralPredictor, dict[str, Any]]:
    """Load a saved neural network without Python object pickling."""
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "neural_metadata.json").read_text(encoding="utf-8"))
    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Saved model feature columns do not match this code version")

    hidden_sizes = tuple(metadata["hidden_sizes"])
    model = ShelterMLP(
        len(FEATURE_COLUMNS),
        hidden_sizes=hidden_sizes,
        dropout=float(metadata["dropout"]),
    )
    state = torch.load(
        artifact_dir / "pytorch_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    predictor = NeuralPredictor(
        model=model,
        feature_mean=np.asarray(metadata["feature_mean"], dtype=np.float32),
        feature_scale=np.asarray(metadata["feature_scale"], dtype=np.float32),
        target_mean=float(metadata["target_mean"]),
        target_scale=float(metadata["target_scale"]),
        hidden_sizes=hidden_sizes,
        dropout=float(metadata["dropout"]),
        best_epoch=int(metadata["best_epoch"]),
        training_history=pd.DataFrame(),
    )
    return predictor, metadata


def _new_model(
    input_size: int,
    *,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    random_state: int,
) -> ShelterMLP:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return ShelterMLP(input_size, hidden_sizes=hidden_sizes, dropout=dropout)


def _scaling_values(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _target_scaling_values(target: pd.Series) -> tuple[float, float]:
    values = target.to_numpy(dtype=np.float32)
    mean = float(values.mean())
    scale = float(values.std())
    return mean, scale if scale >= 1e-8 else 1.0


def _scale_features(
    features: pd.DataFrame,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return ((values - mean) / scale).astype(np.float32)


def _scale_target(target: pd.Series, mean: float, scale: float) -> np.ndarray:
    values = target.to_numpy(dtype=np.float32)
    return ((values - mean) / scale).astype(np.float32)
