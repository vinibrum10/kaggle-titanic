"""Engenharia de atributos reutilizável e segura para validação cruzada.

Todas as estatísticas (medianas, moda, cortes de tarifa) são aprendidas
em ``fit`` e apenas aplicadas em ``transform``, evitando vazamento de
informação do conjunto de teste para o treino.
"""

from __future__ import annotations

import re
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Normaliza títulos raros/equivalentes para um conjunto compacto.
# "Master" identifica meninos, o que é essencial para imputar idade e
# capturar a alta taxa de sobrevivência desse grupo.
TITLE_NORMALIZATION = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Ms": "Miss",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Lady": "Mrs",
    "Countess": "Mrs",
    "the Countess": "Mrs",
    "Dona": "Mrs",
    "Don": "Mr",
    "Sir": "Mr",
    "Jonkheer": "Mr",
    "Capt": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Dr": "Rare",
    "Rev": "Rare",
}

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FamilySize",
    "SibSp",
    "Parch",
    "FarePerPerson",
    "HasCabin",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Title",
    "Embarked",
    "Deck",
    "IsAlone",
    "FareBin",
]


def extract_title(name: object) -> str:
    """Extrai e normaliza o título social presente no nome do passageiro."""
    if not isinstance(name, str):
        return "Rare"
    match = re.search(r",\s*([^\.]+)\.", name)
    raw = match.group(1).strip() if match else "Unknown"
    return TITLE_NORMALIZATION.get(raw, "Rare")


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria atributos derivados e imputa valores ausentes a partir do treino.

    Atributos gerados
    -----------------
    Title : título normalizado extraído de ``Name``
    FamilySize : ``SibSp + Parch + 1``
    IsAlone : 1 se viajava sozinho
    Deck : primeira letra da cabine, ou ``U`` se ausente
    HasCabin : 1 se a cabine foi informada
    FarePerPerson : tarifa dividida pelo tamanho da família
    FareBin : faixa de tarifa (quantis aprendidos no treino)
    Age / Fare / Embarked : imputados (idade por título, tarifa por classe)
    """

    def __init__(self, n_fare_bins: int = 4):
        self.n_fare_bins = n_fare_bins

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        df = self._copy(X)
        titles = df["Name"].map(extract_title)
        self.age_medians_ = df.groupby(titles)["Age"].median().to_dict()
        self.age_median_global_ = float(df["Age"].median())
        self.fare_medians_by_pclass_ = df.groupby("Pclass")["Fare"].median().to_dict()
        self.fare_median_global_ = float(df["Fare"].median())
        embarked = df["Embarked"].dropna()
        self.embarked_mode_ = embarked.mode().iloc[0] if not embarked.empty else "S"

        fare = df["Fare"].copy()
        fare_fill = df["Pclass"].map(self.fare_medians_by_pclass_).fillna(
            self.fare_median_global_
        )
        fare = fare.fillna(fare_fill)
        # pd.qcut devolve os cortes; expandimos as pontas para cobrir o teste.
        _, edges = pd.qcut(
            fare, q=self.n_fare_bins, retbins=True, duplicates="drop"
        )
        edges = np.asarray(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.fare_bin_edges_ = edges
        self.n_fare_bins_actual_ = int(len(edges) - 1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._copy(X)
        df["Title"] = df["Name"].map(extract_title)
        df["FamilySize"] = df["SibSp"].astype(int) + df["Parch"].astype(int) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        deck = df["Cabin"].fillna("U").astype(str).str[0]
        # A cabine T aparece uma única vez no treino; agrupa com desconhecido.
        df["Deck"] = deck.replace({"T": "U"})

        age_fill = df["Title"].map(self.age_medians_).fillna(self.age_median_global_)
        df["Age"] = df["Age"].fillna(age_fill)

        fare_fill = df["Pclass"].map(self.fare_medians_by_pclass_).fillna(
            self.fare_median_global_
        )
        df["Fare"] = df["Fare"].fillna(fare_fill)

        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"].clip(lower=1)

        labels = [f"Q{i}" for i in range(1, self.n_fare_bins_actual_ + 1)]
        df["FareBin"] = pd.cut(
            df["Fare"],
            bins=self.fare_bin_edges_,
            labels=labels,
            include_lowest=True,
        ).astype(str)

        # Garante tipos estáveis para o ColumnTransformer.
        df["Pclass"] = df["Pclass"].astype(int)
        return df

    @staticmethod
    def _copy(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def build_preprocessor() -> Pipeline:
    """Retorna o pipeline de engenharia + pré-processamento (escala e one-hot).

    O resultado é uma matriz numérica pronta para qualquer estimador
    do scikit-learn, inclusive a rede MLP.
    """
    engineer = TitanicFeatureEngineer()
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", engineer),
            ("preprocess", column_transformer),
        ]
    )
