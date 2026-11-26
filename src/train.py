"""Treina e compara modelos com validação cruzada estratificada e gera a submissão."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Permite rodar tanto como ``python -m src.train`` quanto como script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_preprocessor  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5


def project_root() -> Path:
    return ROOT


def default_models() -> dict:
    """Modelos com hiperparâmetros conservadores para um conjunto pequeno."""
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        ),
        "Rede neural (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
    }


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])
    return X, y, test


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []
    for name, estimator in default_models().items():
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("model", estimator),
            ]
        )
        scores = cross_val_score(
            pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1
        )
        rows.append(
            {
                "modelo": name,
                "cv_media": float(scores.mean()),
                "cv_desvio": float(scores.std()),
                "folds": [float(s) for s in scores],
            }
        )
        print(
            f"{name}: {scores.mean():.4f} ± {scores.std():.4f} "
            f"(folds={[round(float(s), 4) for s in scores]})"
        )
    results = pd.DataFrame(rows).sort_values("cv_media", ascending=False)
    results["cv_media"] = results["cv_media"].astype(float)
    return results.reset_index(drop=True)


def fit_best_and_predict(
    X: pd.DataFrame, y: pd.Series, test: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    estimator = default_models()[model_name]
    pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )
    pipe.fit(X, y)
    preds = pipe.predict(test)
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": np.asarray(preds, dtype=int),
        }
    )
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino e submissão do Titanic")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root() / "data",
        help="Pasta com train.csv e test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "submissions" / "submission.csv",
        help="Caminho do arquivo de submissão",
    )
    args = parser.parse_args()

    X, y, test = load_data(args.data_dir)
    print(f"Treino: {len(X)} linhas | Teste: {len(test)} linhas")
    results = evaluate_models(X, y)
    best = results.iloc[0]["modelo"]
    print(f"\nModelo escolhido: {best}")
    submission = fit_best_and_predict(X, y, test, best)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Submissão salva em {args.output} ({len(submission)} linhas)")
    print(results[["modelo", "cv_media", "cv_desvio"]].to_string(index=False))


if __name__ == "__main__":
    main()
