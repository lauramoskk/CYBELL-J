from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(model):
    """
    Pré-processamento oficial do CYBELL-J.

    1. Imputação pela mediana.
    2. Padronização z-score.
    3. Modelo.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )