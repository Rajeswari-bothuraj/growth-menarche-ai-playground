"""PyTorch utilities for the interactive neural-network playground."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


class InteractiveNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int,
        activation: str = "relu"
    ):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")

        if input_size < 1 or output_size < 2:
            raise ValueError(
                "The network needs at least one input and two output classes"
            )

        if any(
            (not isinstance(n, int)) or n < 1
            for n in hidden_layers
        ):
            raise ValueError(
                "Every hidden layer must have a positive integer neuron count"
            )

        self.activation_name = activation
        self.hidden_sizes = list(hidden_layers)

        layers = []
        previous = input_size

        for count in hidden_layers:
            layers.append(nn.Linear(previous, count))
            layers.append(ACTIVATIONS[activation]())
            previous = count

        layers.append(nn.Linear(previous, output_size))

        # Keep Sequential layout for compatibility with existing API.
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward_trace(self, x: torch.Tensor) -> dict:
        """Return actual intermediate values from the trained network."""

        trace = [
            {
                "layer": "Input",
                "values": x.detach().cpu().tolist()
            }
        ]

        current = x
        hidden_index = 0
        logits = None
        pre_activation = None

        for layer in self.network:

            if isinstance(layer, nn.Linear):
                current = layer(current)

                if hidden_index < len(self.hidden_sizes):
                    pre_activation = current
                else:
                    logits = current

            elif not isinstance(layer, nn.Identity):

                current = layer(current)

                trace.append({
                    "layer": f"Hidden {hidden_index + 1}",
                    "pre_activation": pre_activation.detach().cpu().tolist(),
                    "activation": current.detach().cpu().tolist()
                })

                hidden_index += 1

        probabilities = torch.softmax(logits, dim=1)

        trace.append({
            "layer": "Output",
            "logits": logits.detach().cpu().tolist(),
            "probabilities": probabilities.detach().cpu().tolist()
        })

        return {
            "trace": trace,
            "logits": logits.detach().cpu().tolist(),
            "probabilities": probabilities.detach().cpu().tolist()
        }


def _tensor(array, dtype=torch.float32):

    result = torch.as_tensor(
        np.asarray(array),
        dtype=dtype
    )

    if result.ndim == 1:
        result = result.reshape(-1, 1)

    return result


def train_network(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layers=None,
    activation="relu",
    learning_rate=0.001,
    epochs=100,
    weight_decay=0.001,
    random_state=42
):

    hidden_layers = (
        [16, 8]
        if hidden_layers is None
        else list(hidden_layers)
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Training and test sets must both contain records"
        )

    if not 1 <= int(epochs) <= 2000:
        raise ValueError(
            "Epochs must be between 1 and 2000"
        )

    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError(
            "Learning rate must be positive and weight decay non-negative"
        )

    torch.manual_seed(int(random_state))

    Xtr = _tensor(X_train)
    Xte = _tensor(X_test)

    ytr = _tensor(y_train, torch.long).reshape(-1)
    yte = _tensor(y_test, torch.long).reshape(-1)

    classes = torch.unique(ytr)

    if len(classes) < 2:
        raise ValueError(
            "Classification requires at least two training classes"
        )

    if len(torch.unique(yte)) < 1:
        raise ValueError("The test set is empty")

    model = InteractiveNeuralNetwork(
        Xtr.shape[1],
        hidden_layers,
        len(classes),
        activation
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = {
        "loss": [],
        "accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for _ in range(int(epochs)):

        model.train()

        optimizer.zero_grad()

        train_logits = model(Xtr)

        loss = criterion(train_logits, ytr)

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            train_pred = model(Xtr).argmax(dim=1)

            test_logits = model(Xte)

            test_loss = criterion(test_logits, yte)

            test_pred = test_logits.argmax(dim=1)

            history["accuracy"].append(
                (train_pred == ytr).float().mean().item()
            )

            history["test_accuracy"].append(
                (test_pred == yte).float().mean().item()
            )

        history["loss"].append(float(loss.item()))

        history["test_loss"].append(
            float(test_loss.item())
        )

    return model, history


def predict(model, input_data):

    model.eval()

    x = _tensor(input_data)

    with torch.no_grad():

        probabilities = torch.softmax(
            model(x),
            dim=1
        )

        predicted = probabilities.argmax(dim=1)

    return {
        "predicted_classes": predicted.cpu().tolist(),
        "probabilities": probabilities.cpu().tolist()
    }


def explain_forward_pass(model, input_data):

    model.eval()

    x = _tensor(input_data)

    with torch.no_grad():
        return model.forward_trace(x)


def get_network_parameters(model):

    return {
        name: parameter.detach().cpu().tolist()
        for name, parameter in model.named_parameters()
    }


def compare_configurations(
    X_train,
    y_train,
    X_test,
    y_test,
    configurations: Iterable[list[int]],
    **training_options
):

    from sklearn.metrics import accuracy_score, f1_score

    results = []

    for architecture in configurations:

        model, history = train_network(
            X_train,
            y_train,
            X_test,
            y_test,
            hidden_layers=architecture,
            **training_options
        )

        prediction = predict(
            model,
            X_test
        )["predicted_classes"]

        results.append({
            "hidden_layers": list(architecture),

            "accuracy": float(
                accuracy_score(y_test, prediction)
            ),

            "f1_macro": float(
                f1_score(
                    y_test,
                    prediction,
                    average="macro",
                    zero_division=0
                )
            ),

            "final_train_loss": history["loss"][-1],

            "final_test_loss": history["test_loss"][-1],

            "history": history
        })

    return results


def decision_surface(
    model,
    X_reference,
    feature_indices=(0, 1),
    resolution=100,
    fixed_values=None
):

    """Create a 2-D prediction grid."""

    model.eval()

    X = np.asarray(
        X_reference,
        dtype=np.float32
    )

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(
            "Decision boundary requires at least two processed numeric features"
        )

    i, j = map(int, feature_indices)

    if (
        i == j
        or min(i, j) < 0
        or max(i, j) >= X.shape[1]
    ):
        raise ValueError(
            "Select two different valid feature indices"
        )

    if resolution < 20 or resolution > 200:
        raise ValueError(
            "Resolution must be between 20 and 200"
        )

    x_min = float(X[:, i].min())
    x_max = float(X[:, i].max())

    y_min = float(X[:, j].min())
    y_max = float(X[:, j].max())

    x_pad = (x_max - x_min) * 0.08 or 0.5
    y_pad = (y_max - y_min) * 0.08 or 0.5

    gx = np.linspace(
        x_min - x_pad,
        x_max + x_pad,
        resolution
    )

    gy = np.linspace(
        y_min - y_pad,
        y_max + y_pad,
        resolution
    )

    xx, yy = np.meshgrid(gx, gy)

    base = np.asarray(
        fixed_values
        if fixed_values is not None
        else X.mean(axis=0),
        dtype=np.float32
    )

    if base.shape[0] != X.shape[1]:
        raise ValueError(
            "fixed_values must match the transformed feature count"
        )

    grid = np.repeat(
        base.reshape(1, -1),
        xx.size,
        axis=0
    )

    grid[:, i] = xx.ravel()
    grid[:, j] = yy.ravel()

    with torch.no_grad():

        probs = torch.softmax(
            model(_tensor(grid)),
            dim=1
        ).cpu().numpy()

    return {
        "x": gx.tolist(),
        "y": gy.tolist(),

        "predicted_class": probs.argmax(
            axis=1
        ).reshape(
            resolution,
            resolution
        ).tolist(),

        "class_probabilities": probs.reshape(
            resolution,
            resolution,
            -1
        ).tolist(),

        "feature_indices": [i, j],

        "fixed_values": base.tolist(),

        "note": (
            "Other features are held at their displayed fixed values."
        )
    }