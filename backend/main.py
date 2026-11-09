
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
import io
import uuid
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from pytorch_engine import (
    train_network,
    predict as torch_predict,
    get_network_parameters,
    explain_forward_pass,
    compare_configurations,
    decision_surface
)


# ==================================================
# APPLICATION SETUP
# ==================================================

app = FastAPI(
    title="Early Menarche AI Playground",
    description="Interactive and Explainable Neural Network Playground",
    version="0.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

datasets = {}
models = {}

def describe_target_class(target_name, class_value):
    """Return a readable label for the early-menarche target classes."""
    if target_name == "early_menarche":
        try:
            numeric_value = int(float(class_value))
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value == 1:
            return "1 — Early menarche (age at first menstruation below 12)"
        if numeric_value == 0:
            return "0 — Not early menarche (age at first menstruation 12 or older)"
    if target_name == "attended_school":
        try:
            numeric_value = int(float(class_value))
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value == 1:
            return "1 — Attended school"
        if numeric_value == 0:
            return "0 — Never attended school"
    if target_name == "child_is_girl":
        try:
            numeric_value = int(float(class_value))
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value == 1:
            return "1 — Girl"
        if numeric_value == 0:
            return "0 — Boy"
    return str(class_value)


# Load the three bundled, prepared datasets automatically at startup.
# This means the playground is ready to use without a manual CSV upload.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR_CANDIDATES = [
    PROJECT_ROOT / "datasets",
    Path(__file__).resolve().parent / "datasets",
]
DATASET_DIR = next((path for path in DATASET_DIR_CANDIDATES if path.is_dir()), DATASET_DIR_CANDIDATES[0])
BUNDLED_DATASETS = [
    # Dataset 1: small workflow test dataset (unchanged).
    (1, "test_menarche.csv", "Dataset 1 — Test menarche"),
    # Dataset 2: DS3 original features with synthetic assignment outcome.
    (2, "IHDS-II DS3 data.csv", "Dataset 2 — IHDS DS3 features (synthetic menarche outcome)"),
    # Dataset 3: girls-only DS4 features with synthetic assignment outcome.
    (3, "IHDS-II DS4 menarche data.csv", "Dataset 3 — IHDS DS4 girls (synthetic menarche outcome)"),
]
for slot_number, filename, display_name in BUNDLED_DATASETS:
    dataset_path = DATASET_DIR / filename
    if dataset_path.exists():
        try:
            frame = pd.read_csv(dataset_path)
            if frame.empty:
                print(f"[datasets] Skipping empty file: {dataset_path}")
                continue
            dataset_id = str(uuid.uuid4())
            datasets[dataset_id] = {
                "slot": slot_number,
                "name": filename,
                "display_name": display_name,
                "df": frame,
            }
            print(f"[datasets] Loaded Dataset {slot_number}: {filename} ({len(frame)} rows)")
        except Exception as exc:
            print(f"[datasets] Could not load {dataset_path}: {exc}")
    else:
        print(f"[datasets] Dataset {slot_number} file not found: {dataset_path}")


# ==================================================
# REQUEST MODELS
# ==================================================

class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    features: list[str]
    test_size: float = 0.2
    hidden_layers: list[int] = Field(default_factory=lambda: [16, 8])
    activation: str = "relu"
    alpha: float = 0.001
    learning_rate_init: float = 0.001
    max_iter: int = 500
    random_state: int = 42


class PyTorchTrainRequest(BaseModel):
    dataset_id: str
    target: str
    features: list[str]
    test_size: float = 0.2
    hidden_layers: list[int] = Field(default_factory=lambda: [16, 8])
    activation: str = "relu"
    learning_rate: float = 0.001
    epochs: int = 100
    weight_decay: float = 0.001
    random_state: int = 42


class PredictRequest(BaseModel):
    model_id: str
    values: dict


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "message": "Early Menarche AI Playground backend running"
    }


# ==================================================
# DATASET UPLOAD
# ==================================================

@app.post("/api/datasets/{slot}/upload")
async def upload_dataset(
    slot: int,
    file: UploadFile = File(...)
):

    if slot not in (1, 2, 3):
        raise HTTPException(
            400,
            "Dataset slot must be 1, 2 or 3"
        )

    if not file.filename:
        raise HTTPException(400, "Filename is missing")

    raw = await file.read()

    try:

        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))

        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))

        else:
            raise HTTPException(
                400,
                "Upload CSV or Excel files only"
            )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            400,
            f"Could not read file: {str(e)}"
        )

    if df.empty:
        raise HTTPException(400, "Dataset is empty")

    dataset_id = str(uuid.uuid4())

    datasets[dataset_id] = {
        "slot": slot,
        "name": file.filename,
        "df": df
    }

    return {
        "dataset_id": dataset_id,
        "slot": slot,
        "name": file.filename,
        "rows": len(df),
        "columns": list(df.columns)
    }


# ==================================================
# LIST DATASETS
# ==================================================

@app.get("/api/datasets")
def list_datasets():

    return [
        {
            "dataset_id": key,
            "slot": value["slot"],
            "name": value["name"],
            "display_name": value.get("display_name", value["name"]),
            "rows": len(value["df"]),
            "columns": list(value["df"].columns)
        }
        for key, value in datasets.items()
    ]


# ==================================================
# DATASET PREVIEW
# ==================================================

@app.get("/api/datasets/{dataset_id}/preview")
def preview(dataset_id: str):

    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")

    df = datasets[dataset_id]["df"].head(10)

    df = df.astype(object).where(pd.notna(df), None)

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records")
    }


# ==================================================
# COMMON DATA VALIDATION
# ==================================================

def prepare_data(df, target, features):

    if target not in df.columns:
        raise HTTPException(400, "Target column not found")

    if not features:
        raise HTTPException(400, "Select at least one feature")

    if len(features) != len(set(features)):
        raise HTTPException(400, "Duplicate features are not allowed")

    if any(column not in df.columns for column in features):
        raise HTTPException(400, "Invalid feature columns")

    if target in features:
        raise HTTPException(
            400,
            "Target cannot also be an input feature"
        )

    # Prevent direct leakage for the DS3 early-menarche classification task.
    # The age at first menstruation is used to derive the target and must not
    # be supplied to the model as an input feature.
    if target == "early_menarche":
        leakage_columns = {
            "MH1E", "menarche_age_years", "synthetic_menarche_age_years",
            "synthetic_early_menarche", "age_source", "age_assignment_method",
            "synthetic_test_record_id", "test_data_only"
        }
        leaked = sorted(set(features).intersection(leakage_columns))
        if leaked:
            raise HTTPException(
                400,
                "Remove menarche outcome/metadata columns from inputs: " + ", ".join(leaked)
            )

    data = df[features + [target]].dropna(subset=[target])

    if len(data) < 20:
        raise HTTPException(
            400,
            "At least 20 labelled rows are required"
        )

    X = data[features]
    y = data[target].astype(str)

    if y.nunique() < 2:
        class_counts = y.value_counts(dropna=False).to_dict()
        raise HTTPException(
            400,
            f"Target '{target}' has only one class after filtering: {class_counts}. "
            "For Dataset 2, select early_menarche (0 = Not early menarche, 1 = Early menarche)."
        )

    if y.nunique() > 20:
        raise HTTPException(
            400,
            "Too many target classes. Select a categorical target."
        )

    return X, y


# ==================================================
# PREPROCESSING
# ==================================================

def create_preprocessor(X_train, features):

    numeric_columns = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:

        transformers.append((
            "num",

            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),

            numeric_columns
        ))

    if categorical_columns:

        transformers.append((
            "cat",

            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),

            categorical_columns
        ))

    if not transformers:
        raise HTTPException(
            400,
            "No usable input features found"
        )

    return ColumnTransformer(transformers)


# ==================================================
# SCIKIT-LEARN TRAINING
# ==================================================

@app.post("/api/train")
def train(req: TrainRequest):

    if req.dataset_id not in datasets:
        raise HTTPException(404, "Upload/select a dataset first")

    if not 0.1 <= req.test_size <= 0.4:
        raise HTTPException(
            400,
            "Test size must be between 0.1 and 0.4"
        )

    if not req.hidden_layers or any(
        n < 1 or n > 512 for n in req.hidden_layers
    ):
        raise HTTPException(400, "Invalid hidden layer configuration")

    if req.activation not in ("identity", "logistic", "tanh", "relu"):
        raise HTTPException(400, "Unsupported Scikit-learn activation")

    df = datasets[req.dataset_id]["df"].copy()

    X, y = prepare_data(df, req.target, req.features)

    counts = y.value_counts()

    stratify = y if counts.min() >= 2 else None

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=req.test_size,
            random_state=req.random_state,
            stratify=stratify
        )

        prep = create_preprocessor(X_train, req.features)

        clf = MLPClassifier(
            hidden_layer_sizes=tuple(req.hidden_layers),
            activation=req.activation,
            alpha=req.alpha,
            learning_rate_init=req.learning_rate_init,
            max_iter=req.max_iter,
            random_state=req.random_state,
            early_stopping=True
        )

        pipe = Pipeline([
            ("prep", prep),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)

        predictions = pipe.predict(X_test)
        probabilities = pipe.predict_proba(X_test)

        labels = pipe.named_steps["model"].classes_

    except Exception as e:
        raise HTTPException(
            400,
            f"Scikit-learn training failed: {str(e)}"
        )

    result = {

        "engine": "Scikit-learn",

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),

        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),

        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels
        ).tolist(),

        "class_labels": labels.tolist(),

        "train_rows": len(X_train),
        "test_rows": len(X_test),

        "loss_curve": clf.loss_curve_,

        "validation_scores": getattr(
            clf,
            "validation_scores_",
            []
        ),

        "feature_names": req.features
    }

    if len(labels) == 2:

        try:
            result["roc_auc"] = float(
                roc_auc_score(y_test, probabilities[:, 1])
            )
        except ValueError:
            result["roc_auc"] = None

    model_id = str(uuid.uuid4())

    models[model_id] = {
        "engine": "sklearn",
        "pipe": pipe,
        "features": req.features,
        "classes": labels.tolist(),
        "slot": datasets[req.dataset_id]["slot"]
    }

    result.update({
        "model_id": model_id,
        "dataset_slot": datasets[req.dataset_id]["slot"],
        "warning": (
            "Exploratory model only. Validate target definition, "
            "temporal ordering, participant-level independence "
            "and missingness before research claims."
        )
    })

    return result


# ==================================================
# PYTORCH TRAINING
# ==================================================

@app.post("/api/train-pytorch")
def train_pytorch(req: PyTorchTrainRequest):

    if req.dataset_id not in datasets:
        raise HTTPException(404, "Upload a dataset first")

    if not 0.1 <= req.test_size <= 0.4:
        raise HTTPException(400, "Invalid test size")

    if not 1 <= req.epochs <= 2000:
        raise HTTPException(400, "Epochs must be between 1 and 2000")

    if not req.hidden_layers or any(
        n < 1 or n > 512 for n in req.hidden_layers
    ):
        raise HTTPException(400, "Invalid hidden layer configuration")

    if req.activation not in (
        "relu",
        "tanh",
        "sigmoid",
        "leaky_relu",
        "elu"
    ):
        raise HTTPException(400, "Unsupported activation function")

    if req.learning_rate <= 0 or req.weight_decay < 0:
        raise HTTPException(400, "Invalid learning rate or weight decay")

    df = datasets[req.dataset_id]["df"].copy()

    X, y = prepare_data(df, req.target, req.features)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = np.bincount(y_encoded)

    if class_counts.min() < 2:
        raise HTTPException(
            400,
            "Each class needs at least two records for stratified splitting"
        )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=req.test_size,
            random_state=req.random_state,
            stratify=y_encoded
        )

        prep = create_preprocessor(X_train, req.features)

        X_train_processed = prep.fit_transform(X_train)
        X_test_processed = prep.transform(X_test)

        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()

        if hasattr(X_test_processed, "toarray"):
            X_test_processed = X_test_processed.toarray()

        X_train_processed = np.asarray(
            X_train_processed,
            dtype=np.float32
        )

        X_test_processed = np.asarray(
            X_test_processed,
            dtype=np.float32
        )

        if not np.isfinite(X_train_processed).all():
            raise ValueError("Training data contains invalid values")

        if not np.isfinite(X_test_processed).all():
            raise ValueError("Test data contains invalid values")

        model, history = train_network(
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            hidden_layers=req.hidden_layers,
            activation=req.activation,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            weight_decay=req.weight_decay,
            random_state=req.random_state
        )

        prediction_result = torch_predict(
            model,
            X_test_processed
        )

        predictions = np.array(
            prediction_result["predicted_classes"]
        )

        probabilities = np.array(
            prediction_result["probabilities"]
        )

        labels = label_encoder.classes_.tolist()

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            400,
            f"PyTorch training failed: {str(e)}"
        )

    result = {

        "engine": "PyTorch",

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),

        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),

        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=list(range(len(labels)))
        ).tolist(),

        "class_labels": labels,

        "train_rows": len(X_train),
        "test_rows": len(X_test),

        "loss_curve": history["loss"],
        "accuracy_curve": history["accuracy"],
        "test_loss_curve": history["test_loss"],
        "test_accuracy_curve": history["test_accuracy"],

        "network_parameters": get_network_parameters(model),

        "hidden_layers": req.hidden_layers,
        "activation": req.activation,

        "input_size": int(X_train_processed.shape[1]),
        "output_size": len(labels),

        "feature_names": req.features
    }

    if len(labels) == 2:

        try:
            result["roc_auc"] = float(
                roc_auc_score(y_test, probabilities[:, 1])
            )
        except ValueError:
            result["roc_auc"] = None

    model_id = str(uuid.uuid4())

    models[model_id] = {

        "engine": "pytorch",
        "model": model,
        "prep": prep,
        "label_encoder": label_encoder,
        "features": req.features,
        "target": req.target,
        "classes": labels,
        "slot": datasets[req.dataset_id]["slot"],
        "dataset_id": req.dataset_id,
        "raw_frame": datasets[req.dataset_id]["df"].copy(),

        # Store configuration for accurate network inspection
        "hidden_layers": req.hidden_layers,
        "activation": req.activation
    }

    result.update({

        "model_id": model_id,

        "dataset_slot": datasets[req.dataset_id]["slot"],

        "warning": (
            "Exploratory model only. Validate target definition, "
            "temporal ordering, participant-level independence "
            "and missingness before research claims."
        )
    })

    return result


# ==================================================
# PREDICTION
# ==================================================

@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):

    if req.model_id not in models:
        raise HTTPException(404, "Train a model first")

    model_info = models[req.model_id]

    missing = [
        column
        for column in model_info["features"]
        if column not in req.values
    ]

    if missing:
        raise HTTPException(
            400,
            f"Missing inputs: {missing}"
        )

    row = pd.DataFrame([
        {
            column: req.values[column]
            for column in model_info["features"]
        }
    ])

    # PYTORCH PREDICTION

    if model_info["engine"] == "pytorch":

        try:

            processed = model_info["prep"].transform(row)

            if hasattr(processed, "toarray"):
                processed = processed.toarray()

            processed = np.asarray(
                processed,
                dtype=np.float32
            )

            prediction_result = torch_predict(
                model_info["model"],
                processed
            )

            probabilities = prediction_result["probabilities"][0]

            predicted_index = int(
                prediction_result["predicted_classes"][0]
            )

            classes = model_info["classes"]

            predicted_value = classes[predicted_index]
            target_name = model_info.get("target")
            return {
                "engine": "PyTorch",
                "predicted_class": predicted_value,
                "predicted_label": describe_target_class(target_name, predicted_value),
                "class_descriptions": {
                    str(value): describe_target_class(target_name, value)
                    for value in classes
                },
                "probabilities": {
                    str(classes[i]): float(probabilities[i])
                    for i in range(len(classes))
                },
                "model_id": req.model_id
            }

        except Exception as e:
            raise HTTPException(
                400,
                f"PyTorch prediction failed: {str(e)}"
            )

    # SCIKIT-LEARN PREDICTION

    pipe = model_info["pipe"]

    probabilities = pipe.predict_proba(row)[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_value = pipe.classes_[predicted_index]
    target_name = model_info.get("target")
    return {
        "engine": "Scikit-learn",
        "predicted_class": str(predicted_value),
        "predicted_label": describe_target_class(target_name, predicted_value),
        "class_descriptions": {
            str(value): describe_target_class(target_name, value)
            for value in pipe.classes_
        },
        "probabilities": {
            str(class_name): float(probability)
            for class_name, probability in zip(pipe.classes_, probabilities)
        },
        "model_id": req.model_id
    }


# ==================================================
# TRAINED NETWORK INFORMATION
# ==================================================

@app.get("/api/models/{model_id}/network")
def get_network(model_id: str):

    if model_id not in models:
        raise HTTPException(404, "Model not found")

    model_info = models[model_id]

    if model_info["engine"] != "pytorch":
        raise HTTPException(
            400,
            "Network visualization is available for PyTorch models"
        )

    model = model_info["model"]

    linear_layers = [
        layer
        for layer in model.network
        if hasattr(layer, "out_features")
    ]

    return {

        "model_id": model_id,

        "engine": "PyTorch",

        "input_size": int(
            linear_layers[0].in_features
        ),

        "hidden_layers": model_info["hidden_layers"],

        "output_size": int(
            linear_layers[-1].out_features
        ),

        "activation": model_info["activation"],

        "parameters": get_network_parameters(model)
    }


# ==================================================
# INTERACTIVE FORWARD PROPAGATION / NEURON ACTIVATIONS
# ==================================================

class ForwardPassRequest(BaseModel):
    model_id: str
    values: dict


@app.post("/api/models/forward-pass")
def forward_pass(req: ForwardPassRequest):
    if req.model_id not in models:
        raise HTTPException(404, "Model not found")
    info = models[req.model_id]
    if info["engine"] != "pytorch":
        raise HTTPException(400, "Forward propagation inspection requires a PyTorch model")
    missing = [f for f in info["features"] if f not in req.values or req.values[f] is None or req.values[f] == ""]
    if missing:
        raise HTTPException(400, f"Missing inputs: {missing}")
    row = pd.DataFrame([{f: req.values[f] for f in info["features"]}])
    # Prepared DS2/DS3 feature columns already contain their prepared numeric
    # scale. Do not apply any additional age conversion or scaling here; use
    # the same fitted preprocessing pipeline used during training.
    for column in row.select_dtypes(include=[np.number]).columns:
        row[column] = pd.to_numeric(row[column], errors="coerce")
    if row.isna().any().any():
        bad = row.columns[row.isna().any()].tolist()
        raise HTTPException(400, f"Enter valid numeric/category values for: {bad}")
    try:
        processed = info["prep"].transform(row)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
        processed = np.asarray(processed, dtype=np.float32)
        result = explain_forward_pass(info["model"], processed)
        return {"model_id": req.model_id, "features": info["features"],
                "processed_input": processed[0].tolist(),
                "class_labels": info["classes"], **result}
    except Exception as exc:
        raise HTTPException(400, f"Forward pass failed: {exc}")


# ==================================================
# DECISION BOUNDARY FOR TWO SELECTED RAW FEATURES
# ==================================================

class DecisionBoundaryRequest(BaseModel):
    feature_x: str
    feature_y: str
    resolution: int = 80


@app.post("/api/models/{model_id}/decision-boundary")
def get_decision_boundary(model_id: str, req: DecisionBoundaryRequest):
    if model_id not in models:
        raise HTTPException(404, "Model not found")
    info = models[model_id]
    if info["engine"] != "pytorch":
        raise HTTPException(400, "Decision boundary is currently available for PyTorch models")
    if req.feature_x == req.feature_y or req.feature_x not in info["features"] or req.feature_y not in info["features"]:
        raise HTTPException(400, "Choose two different features used by this model")
    df = datasets.get(info.get("dataset_id"), {}).get("df")
    if df is None:
        # The model stores the raw training frame below; this fallback handles older in-memory models.
        df = info.get("raw_frame")
    if df is None:
        raise HTTPException(400, "Training reference data is unavailable. Retrain the model to enable this plot.")
    raw = df[info["features"]].copy()
    numeric = raw.select_dtypes(include=np.number).columns.tolist()
    if req.feature_x not in numeric or req.feature_y not in numeric:
        raise HTTPException(400, "Decision boundary requires two numeric raw features")
    try:
        transformed = info["prep"].transform(raw)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float32)
        # Only use the selected raw numeric dimensions when preprocessing preserves one column per feature.
        if transformed.shape[1] != len(info["features"]) or len(numeric) != len(info["features"]):
            raise ValueError("For this initial plot, select a model with only numeric features and no one-hot expansion")
        indices = [info["features"].index(req.feature_x), info["features"].index(req.feature_y)]
        result = decision_surface(info["model"], transformed, indices, req.resolution)
        # Return the observed raw data points as well as the prediction grid.
        # The chart can overlay these points to show the real sample distribution.
        observed = raw[[req.feature_x, req.feature_y]].copy()
        observed["target"] = df.loc[observed.index, info["target"]]
        observed = observed.replace([np.inf, -np.inf], np.nan).dropna()
        observed = observed[observed["target"].notna()]
        encoded_targets = info["label_encoder"].transform(observed["target"].astype(str))
        result.update({
            "feature_names": [req.feature_x, req.feature_y],
            "class_labels": info["classes"],
            "observations": [
                {
                    "x": float(row[req.feature_x]),
                    "y": float(row[req.feature_y]),
                    "class": int(encoded_targets[i]),
                    "label": str(row["target"])
                }
                for i, (_, row) in enumerate(observed.iterrows())
            ]
        })
        return result
    except Exception as exc:
        raise HTTPException(400, f"Could not calculate decision boundary: {exc}")


# ==================================================
# TRAINING COMPARISON: SAME DATA AND SAME SPLIT
# ==================================================

class CompareRequest(BaseModel):
    dataset_id: str
    target: str
    features: list[str]
    configurations: list[list[int]] = Field(default_factory=lambda: [[8], [16, 8], [32, 16]])
    test_size: float = 0.2
    activation: str = "relu"
    learning_rate: float = 0.001
    epochs: int = 100
    weight_decay: float = 0.001
    random_state: int = 42


@app.post("/api/train-comparison")
def train_comparison(req: CompareRequest):
    if req.dataset_id not in datasets:
        raise HTTPException(404, "Upload/select a dataset first")
    if not 0.1 <= req.test_size <= 0.4:
        raise HTTPException(400, "Test size must be between 0.1 and 0.4")
    if not 1 <= req.epochs <= 2000:
        raise HTTPException(400, "Epochs must be between 1 and 2000")
    if not 2 <= len(req.configurations) <= 6:
        raise HTTPException(400, "Provide between 2 and 6 hidden-layer configurations")
    if any(not config or any(n < 1 or n > 512 for n in config) for config in req.configurations):
        raise HTTPException(400, "Each configuration must contain valid positive neuron counts")
    if req.activation not in ("relu", "tanh", "sigmoid", "leaky_relu", "elu"):
        raise HTTPException(400, "Unsupported activation")
    X, y = prepare_data(datasets[req.dataset_id]["df"].copy(), req.target, req.features)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    if np.bincount(y_encoded).min() < 2:
        raise HTTPException(400, "Each class needs at least two records")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=req.test_size, random_state=req.random_state, stratify=y_encoded)
        prep = create_preprocessor(X_train, req.features)
        A = prep.fit_transform(X_train)
        B = prep.transform(X_test)
        if hasattr(A, "toarray"): A = A.toarray()
        if hasattr(B, "toarray"): B = B.toarray()
        A = np.asarray(A, dtype=np.float32); B = np.asarray(B, dtype=np.float32)
        results = compare_configurations(A, y_train, B, y_test, req.configurations,
            activation=req.activation, learning_rate=req.learning_rate, epochs=req.epochs,
            weight_decay=req.weight_decay, random_state=req.random_state)
        # Keep the response compact; detailed per-epoch history is returned for charting.
        return {"dataset_slot": datasets[req.dataset_id]["slot"],
                "dataset_id": req.dataset_id, "class_labels": encoder.classes_.tolist(),
                "train_rows": len(X_train), "test_rows": len(X_test),
                "results": results,
                "warning": "Use this comparison for exploration only. Do not select a final model using the test set; use a validation set for model selection."}
    except Exception as exc:
        raise HTTPException(400, f"Training comparison failed: {exc}")
