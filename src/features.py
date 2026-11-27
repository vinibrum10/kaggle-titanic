"""Engenharia de atributos reutilizável e segura para validação cruzada.

Todas as estatísticas (medianas, moda, cortes de tarifa e a sobrevivência
de grupos familiares/de ticket) são aprendidas em ``fit`` e apenas
aplicadas em ``transform``, evitando vazamento de informação do conjunto
de validação ou de teste para o treino.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Optional, Union

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

DEFAULT_GROUP_SURVIVAL = 0.5

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FamilySize",
    "SibSp",
    "Parch",
    "FarePerPerson",
    "HasCabin",
    "FamilySurvival",
    "GroupSize",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Title",
    "Embarked",
    "Deck",
    "IsAlone",
    "FareBin",
    "PclassSex",
]


def extract_title(name: object) -> str:
    """Extrai e normaliza o título social presente no nome do passageiro."""
    if not isinstance(name, str):
        return "Rare"
    match = re.search(r",\s*([^\.]+)\.", name)
    raw = match.group(1).strip() if match else "Unknown"
    return TITLE_NORMALIZATION.get(raw, "Rare")


def extract_surname(name: object) -> str:
    """Sobrenome = texto antes da primeira vírgula do campo ``Name``."""
    if not isinstance(name, str) or "," not in name:
        return "UNKNOWN"
    surname = name.split(",", 1)[0].strip()
    return surname or "UNKNOWN"


def fare_token(fare: object) -> str:
    """Chave estável para a tarifa, usada no agrupamento familiar."""
    if fare is None or (isinstance(fare, (float, np.floating)) and np.isnan(fare)):
        return "NA"
    try:
        value = float(fare)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(value):
        return "NA"
    return f"{value:.5f}"


def family_group_key(name: object, fare: object) -> str:
    """Grupo familiar clássico: sobrenome + tarifa."""
    return f"{extract_surname(name)}|{fare_token(fare)}"


def ticket_group_key(ticket: object) -> str:
    if ticket is None or (isinstance(ticket, (float, np.floating)) and np.isnan(ticket)):
        return "UNKNOWN"
    text = str(ticket).strip()
    return text or "UNKNOWN"


def survival_from_others(labels: Iterable[int]) -> float:
    """Truque clássico de Family_Survival, em versão leave-one-out.

    Se algum outro membro rotulado do grupo sobreviveu, devolve 1.
    Se todos os outros morreram, devolve 0. Sem companheiros rotulados,
    devolve 0,5 (sem informação).
    """
    others = list(labels)
    if not others:
        return DEFAULT_GROUP_SURVIVAL
    if max(others) == 1:
        return 1.0
    return 0.0


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
    PclassSex : interação ``Pclass`` × ``Sex`` (ex.: ``1_female``)
    FamilySurvival : sobrevivência do grupo familiar (sobrenome + tarifa)
        ou, se ainda indefinida, do grupo do ticket. Calculada só com
        rótulos vistos no ``fit`` e em regime leave-one-out para as
        linhas do próprio treino — a sobrevivência do passageiro nunca
        entra no atributo dele.
    GroupSize : tamanho do grupo (união família ∪ ticket) no conjunto
        de referência (treino + teste, sem rótulos)
    Age / Fare / Embarked : imputados (idade por título, tarifa por classe)

    Parameters
    ----------
    n_fare_bins : número de faixas de tarifa tentadas no ``qcut``.
    passenger_pool : passageiros usados só para definir a composição dos
        grupos (tamanho e quem viaja junto). Pode incluir o teste, mas
        **não** deve carregar o alvo. Os rótulos vêm exclusivamente de
        ``fit(X, y)``.
    """

    def __init__(
        self,
        n_fare_bins: int = 4,
        passenger_pool: Optional[pd.DataFrame] = None,
    ):
        self.n_fare_bins = n_fare_bins
        self.passenger_pool = passenger_pool

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
        _, edges = pd.qcut(
            fare, q=self.n_fare_bins, retbins=True, duplicates="drop"
        )
        edges = np.asarray(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.fare_bin_edges_ = edges
        self.n_fare_bins_actual_ = int(len(edges) - 1)

        self._fit_group_indices(df, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._basic_features(X)
        family_survival = []
        group_size = []
        for pid, name, fare, ticket in zip(
            df["PassengerId"], df["Name"], df["Fare"], df["Ticket"]
        ):
            pid = int(pid)
            fam_key = family_group_key(name, fare)
            tix_key = ticket_group_key(ticket)
            fam_ids = self.family_members_.get(fam_key, [])
            tix_ids = self.ticket_members_.get(tix_key, [])
            members = set(fam_ids) | set(tix_ids)
            members.add(pid)
            group_size.append(int(len(members)))

            other_fam = [
                self.labeled_survival_[i]
                for i in fam_ids
                if i in self.labeled_survival_ and i != pid
            ]
            value = survival_from_others(other_fam)
            if value == DEFAULT_GROUP_SURVIVAL:
                other_tix = [
                    self.labeled_survival_[i]
                    for i in tix_ids
                    if i in self.labeled_survival_ and i != pid
                ]
                value = survival_from_others(other_tix)
            family_survival.append(value)

        df["FamilySurvival"] = np.asarray(family_survival, dtype=float)
        df["GroupSize"] = np.asarray(group_size, dtype=int)
        return df

    def _basic_features(self, X: pd.DataFrame) -> pd.DataFrame:
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

        df["Pclass"] = df["Pclass"].astype(int)
        df["PclassSex"] = (
            df["Pclass"].astype(str) + "_" + df["Sex"].astype(str)
        )
        return df

    def _fit_group_indices(
        self, df: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]
    ) -> None:
        sources = [df]
        if self.passenger_pool is not None:
            sources.append(self._copy(self.passenger_pool))

        family_members: dict[str, list[int]] = defaultdict(list)
        ticket_members: dict[str, list[int]] = defaultdict(list)
        seen_family: dict[str, set[int]] = defaultdict(set)
        seen_ticket: dict[str, set[int]] = defaultdict(set)

        for source in sources:
            filled = source.copy()
            fare_fill = filled["Pclass"].map(self.fare_medians_by_pclass_).fillna(
                self.fare_median_global_
            )
            filled["Fare"] = filled["Fare"].fillna(fare_fill)
            for pid, name, fare, ticket in zip(
                filled["PassengerId"], filled["Name"], filled["Fare"], filled["Ticket"]
            ):
                pid = int(pid)
                fam_key = family_group_key(name, fare)
                tix_key = ticket_group_key(ticket)
                if pid not in seen_family[fam_key]:
                    family_members[fam_key].append(pid)
                    seen_family[fam_key].add(pid)
                if pid not in seen_ticket[tix_key]:
                    ticket_members[tix_key].append(pid)
                    seen_ticket[tix_key].add(pid)

        self.family_members_ = dict(family_members)
        self.ticket_members_ = dict(ticket_members)

        labeled: dict[int, int] = {}
        if y is not None:
            y_arr = np.asarray(y).ravel()
            if len(y_arr) != len(df):
                raise ValueError(
                    "y precisa ter o mesmo número de linhas que X para "
                    "calcular FamilySurvival."
                )
            for pid, label in zip(df["PassengerId"].astype(int), y_arr):
                if pd.isna(label):
                    continue
                labeled[int(pid)] = int(label)
        self.labeled_survival_ = labeled

    @staticmethod
    def _copy(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def build_preprocessor(
    passenger_pool: Optional[pd.DataFrame] = None,
) -> Pipeline:
    """Retorna o pipeline de engenharia + pré-processamento (escala e one-hot).

    O resultado é uma matriz numérica pronta para qualquer estimador
    do scikit-learn. ``passenger_pool`` é o conjunto (em geral treino +
    teste, sem o alvo) usado só para medir o tamanho dos grupos.
    """
    engineer = TitanicFeatureEngineer(passenger_pool=passenger_pool)
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
