import bentoml
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@bentoml.service(
    name="model_autopsy_service",
    traffic={"timeout": 10}
)
class ModelAutopsyService:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.randint(0, 2, 100)
        self.model.fit(X_dummy, y_dummy)

    @bentoml.api
    def predict(self, input_data: list) -> list:
        preds = self.model.predict(input_data)
        return preds.tolist()

    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy", "model_type": "RandomForestClassifier"}
