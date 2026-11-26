"""Pipeline de atributos e treino para a competição Titanic do Kaggle."""

from .features import (
    TitanicFeatureEngineer,
    build_preprocessor,
    extract_title,
)

__all__ = [
    "TitanicFeatureEngineer",
    "build_preprocessor",
    "extract_title",
]
