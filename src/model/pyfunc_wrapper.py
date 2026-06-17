import mlflow.pyfunc
import joblib
import torch
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from model.classifier import SleepClassifier


class SleepRiskPredictor(mlflow.pyfunc.PythonModel):
    """
    Custom MlFlow wrapper to bundle scikit-learn scaler,
    label encoder, and torch model
    """

    def load_context(self, context):
        """Loads artifacts scaler and checkpoint into memory on startup"""

        # load scalers and model
        self.scaler: ColumnTransformer = joblib.load(context.artifacts["scaler"])
        self.label_encoder: LabelEncoder = joblib.load(
            context.artifacts["label_encoder"]
        )
        self.model = SleepClassifier.load_from_checkpoint(
            context.artifacts["checkpoint"]
        )

        # set model to eval mode
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame) -> list[str]:
        """Runs the inference pipeline for the model"""

        X_scaled = self.scaler.transform(model_input)

        if hasattr(X_scaled, "toarray"):
            X_scaled = X_scaled.toarray()
        elif hasattr(X_scaled, "to_numpy"):
            X_scaled = X_scaled.to_numpy()

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)

        predictions_np = predictions.cpu().numpy()
        string_result = self.label_encoder.inverse_transform(predictions_np)

        return string_result.tolist()
