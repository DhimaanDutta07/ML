from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Preprocessor():
    def __init__(self, x=None):
        self.cols = ["age", "sex", "resting_bp", "cholesterol", "max_heart_rate"]

    def build(self):
        num_pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
        ])

        return ColumnTransformer([
            ("num", num_pipe, self.cols)
        ])
