import os

from catboost import CatBoostClassifier
model_path = os.path.join(os.path.dirname(__file__), "CatBoostModel.cbm")
print(model_path)

model=  CatBoostClassifier()
model.load_model(model_path)