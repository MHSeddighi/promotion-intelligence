import json
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

class ModelRegistry:
    def __init__(self, models_dir: str | Path = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

    def load_model(self, model_name: str):
        local_path = self.models_dir / f"{model_name}.pkl"
        meta_path = self.models_dir / f"{model_name}_meta.json"

        if not local_path.exists():
            return None, None

        model = joblib.load(local_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        return model, metadata

    def list_models(self) -> list[str]:
        return [f.stem for f in self.models_dir.glob("*.pkl")]

