import bentoml
from bentoml.io import NumpyNdarray
import numpy as np
import joblib

model_runner = bentoml.sklearn.load_runner("iris_model", model_store=".")
svc = bentoml.Service("iris_classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_data: np.ndarray) -> np.ndarray:
    return model_runner.predict.run(input_data)