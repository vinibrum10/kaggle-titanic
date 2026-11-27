"""Treina modelos regularizados com CV estratificada repetida e gera a submissão."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Permite rodar tanto como ``python -m src.train`` quanto como script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_preprocessor  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 10
N_REPEATS = 3
ENSEMBLE_NAME = "Ensemble (votação suave)"
ENSEMBLE_MARGIN = 0.01
ENSEMBLE_MIN_MEMBERS = 3
# Empate estatístico: diferenças de média menores que 0,1 p.p. cedem ao menor desvio.
MEAN_TIE_MARGIN = 0.001


def project_root() -> Path:
    return ROOT


def make_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )


def build_passenger_pool(X: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Une treino e teste (sem o alvo) para medir o tamanho dos grupos."""
    cols = list(X.columns)
    return pd.concat([X[cols], test[cols]], axis=0, ignore_index=True)


def default_models() -> dict:
    """Modelos regularizados, conservadores para um conjunto pequeno (~891 linhas)."""
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=2000,
            C=0.5,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            reg_alpha=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        ),
        "SVC": CalibratedClassifierCV(
            SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            ensemble=False,
            cv=5,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            p=2,
        ),
    }


def _estimator_key(name: str) -> str:
    mapping = {
        "Regressão Logística": "logreg",
        "Random Forest": "rf",
        "Gradient Boosting": "gb",
        "XGBoost": "xgb",
        "SVC": "svc",
        "KNN": "knn",
    }
    return mapping.get(name, name.lower().replace(" ", "_"))


def build_ensemble(member_names: list[str]) -> VotingClassifier:
    models = default_models()
    estimators = [
        (_estimator_key(name), models[name]) for name in member_names
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


def make_pipeline(estimator, passenger_pool: Optional[pd.DataFrame] = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(passenger_pool=passenger_pool)),
            ("model", estimator),
        ]
    )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])
    return X, y, test


def _score_estimator(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    passenger_pool: Optional[pd.DataFrame],
) -> np.ndarray:
    pipe = make_pipeline(estimator, passenger_pool)
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)


def _result_row(name: str, scores: np.ndarray) -> dict:
    return {
        "modelo": name,
        "cv_media": float(scores.mean()),
        "cv_desvio": float(scores.std()),
        "folds": [float(s) for s in scores],
        "n_scores": int(len(scores)),
    }


def _print_scores(name: str, scores: np.ndarray) -> None:
    print(
        f"{name}: {scores.mean():.4f} ± {scores.std():.4f} "
        f"(n={len(scores)}, min={scores.min():.4f}, max={scores.max():.4f})"
    )


def select_ensemble_members(
    results: pd.DataFrame,
    margin: float = ENSEMBLE_MARGIN,
    min_members: int = ENSEMBLE_MIN_MEMBERS,
) -> list[str]:
    """Escolhe os melhores modelos a priori, sem olhar o teste.

    Entram os que ficam a até ``margin`` da melhor média; se forem
    menos de ``min_members``, completa com os seguintes da tabela.
    """
    ranked = results[~results["modelo"].str.startswith("Ensemble")].copy()
    ranked = ranked.sort_values(
        ["cv_media", "cv_desvio"], ascending=[False, True]
    )
    best = float(ranked.iloc[0]["cv_media"])
    names = list(ranked.loc[ranked["cv_media"] >= best - margin, "modelo"])
    if len(names) < min_members:
        names = list(ranked["modelo"].head(min_members))
    return names


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    passenger_pool: Optional[pd.DataFrame] = None,
    include_ensemble: bool = True,
) -> pd.DataFrame:
    """Compara os modelos (e o ensemble) com 10 folds estratificados × 3 repetições."""
    cv = make_cv()
    pool = passenger_pool if passenger_pool is not None else X
    rows = []
    for name, estimator in default_models().items():
        scores = _score_estimator(estimator, X, y, cv, pool)
        rows.append(_result_row(name, scores))
        _print_scores(name, scores)

    results = pd.DataFrame(rows)
    if include_ensemble:
        members = select_ensemble_members(results)
        print(f"\nMembros do ensemble: {', '.join(members)}")
        scores = _score_estimator(build_ensemble(members), X, y, cv, pool)
        ensemble_row = _result_row(ENSEMBLE_NAME, scores)
        ensemble_row["membros"] = members
        rows.append(ensemble_row)
        _print_scores(ENSEMBLE_NAME, scores)
        results = pd.DataFrame(rows)

    results = results.sort_values(
        ["cv_media", "cv_desvio"], ascending=[False, True]
    ).reset_index(drop=True)
    return results


def resolve_estimator(
    model_name: str, cv_results: Optional[pd.DataFrame] = None
):
    if model_name == ENSEMBLE_NAME:
        if cv_results is None:
            raise ValueError(
                "Para retreinar o ensemble é preciso passar os resultados da CV."
            )
        members = select_ensemble_members(cv_results)
        return build_ensemble(members), members
    return default_models()[model_name], None


def fit_best_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    model_name: str,
    passenger_pool: Optional[pd.DataFrame] = None,
    cv_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    pool = passenger_pool if passenger_pool is not None else pd.concat(
        [X, test], ignore_index=True
    )
    estimator, _ = resolve_estimator(model_name, cv_results)
    pipe = make_pipeline(estimator, pool)
    pipe.fit(X, y)
    preds = pipe.predict(test)
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": np.asarray(preds, dtype=int),
        }
    )
    return submission


def collect_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    model_name: str,
    passenger_pool: Optional[pd.DataFrame] = None,
    cv_results: Optional[pd.DataFrame] = None,
    top: int = 15,
) -> tuple[Pipeline, pd.DataFrame]:
    """Ajusta o modelo final e devolve importâncias (ou |coeficientes|) médias."""
    pool = passenger_pool if passenger_pool is not None else pd.concat(
        [X, test], ignore_index=True
    )
    estimator, _ = resolve_estimator(model_name, cv_results)
    pipe = make_pipeline(estimator, pool)
    pipe.fit(X, y)

    names = pipe.named_steps["prep"].named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]
    vectors: list[np.ndarray] = []

    def _add_from(est) -> None:
        if hasattr(est, "feature_importances_"):
            vec = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            vec = np.abs(np.asarray(est.coef_, dtype=float).ravel())
        else:
            return
        total = vec.sum()
        if total > 0:
            vec = vec / total
        vectors.append(vec)

    if hasattr(model, "named_estimators_"):
        for est in model.named_estimators_.values():
            _add_from(est)
    else:
        _add_from(model)

    if not vectors:
        return pipe, pd.DataFrame(columns=["atributo", "importancia"])

    importance = np.mean(np.vstack(vectors), axis=0)
    table = (
        pd.DataFrame({"atributo": names, "importancia": importance})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    return pipe, table.head(top)


def pick_best(results: pd.DataFrame) -> pd.Series:
    """Maior média; se a diferença for menor que ``MEAN_TIE_MARGIN``, menor desvio."""
    best_mean = float(results["cv_media"].max())
    tied = results.loc[results["cv_media"] >= best_mean - MEAN_TIE_MARGIN]
    tied = tied.sort_values(
        ["cv_desvio", "cv_media"], ascending=[True, False]
    )
    return tied.iloc[0]


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
    pool = build_passenger_pool(X, test)
    print(f"Treino: {len(X)} linhas | Teste: {len(test)} linhas")
    print(
        f"Validação: RepeatedStratifiedKFold "
        f"({N_SPLITS} folds × {N_REPEATS} repetições, semente {RANDOM_STATE})"
    )

    results = evaluate_models(X, y, passenger_pool=pool)
    best = pick_best(results)
    best_name = best["modelo"]
    print(f"\nModelo escolhido: {best_name} "
          f"({best['cv_media']:.4f} ± {best['cv_desvio']:.4f})")

    submission = fit_best_and_predict(
        X, y, test, best_name, passenger_pool=pool, cv_results=results
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    n_surv = int((submission["Survived"] == 1).sum())
    print(
        f"Submissão salva em {args.output} ({len(submission)} linhas, "
        f"{n_surv} sobreviventes previstos, {n_surv / len(submission):.1%})"
    )
    printable = results[["modelo", "cv_media", "cv_desvio"]].copy()
    print(printable.to_string(index=False))

    _, top = collect_top_features(
        X, y, test, best_name, passenger_pool=pool, cv_results=results
    )
    if not top.empty:
        print("\nAtributos mais importantes (média normalizada no modelo final):")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
