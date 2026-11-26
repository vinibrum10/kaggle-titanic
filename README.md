# Titanic — do dado bruto à submissão

Solução completa da competição introdutória [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic), no Kaggle. O objetivo é prever, para cada passageiro, se ele sobreviveu ao naufrágio (`Survived = 1` ou `0`). A métrica oficial é a **acurácia**.

Autor: **Vinicius Brum**.

Este repositório foi pensado para portfólio: pipeline de atributos reutilizável (sem vazamento entre treino e teste), validação cruzada honesta, notebook de exploração e um arquivo de submissão pronto para o Kaggle.

> **Recado sincero sobre redes neurais.** O Titanic é um problema tabular pequeno (~900 linhas). Ele demonstra que você sabe conduzir o ciclo completo de um projeto de classificação (exploração, engenharia de atributos, validação e submissão). A MLP incluída aqui serve de comparação — e, neste conjunto, fica atrás dos ensembles de árvores, o que é esperado. Para evidenciar treino de redes neurais de verdade, o próximo passo é uma competição de imagem ou de texto.

## O problema

O Kaggle fornece 891 passageiros com o resultado conhecido (`train.csv`) e 418 cujo resultado está oculto (`test.csv`). Cada linha traz classe da passagem, nome, sexo, idade, parentes a bordo, ticket, tarifa, cabine e porto de embarque. Você treina um modelo no conjunto rotulado, envia as 418 previsões e entra no ranking.

A competição não tem prazo nem prêmio em dinheiro: o valor está em aprender o fluxo e, se quiser, documentar o trabalho no GitHub.

## Abordagem

1. **Exploração** das relações entre sobrevivência, sexo, classe, idade, família e cabine (veja `notebooks/titanic.ipynb`).
2. **Engenharia de atributos** em um `Transformer` do scikit-learn, com estatísticas aprendidas só no fold de treino:
   - título social extraído do nome (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`);
   - tamanho da família e indicador de quem viajava sozinho;
   - imputação de idade pela mediana do título;
   - imputação de tarifa pela mediana da classe, mais faixas em quantis e tarifa por pessoa;
   - convés da cabine (`Deck`) e indicador de cabine conhecida;
   - porto de embarque imputado pela moda.
3. **Pré-processamento** com `Pipeline` + `ColumnTransformer`: padronização das numéricas e *one-hot* das categóricas.
4. **Comparação** de cinco modelos com validação cruzada **estratificada de 5 folds** (`random_state=42`).
5. Retreino do melhor modelo em todo o conjunto de treino e geração de `submissions/submission.csv`.

## Resultados

Acurácia na validação cruzada estratificada (5 folds, semente 42). Os números abaixo foram calculados neste repositório, não copiados de kernels públicos.

| Modelo | Acurácia CV | Desvio-padrão | Folds |
| --- | ---: | ---: | --- |
| **Gradient Boosting** | **84,40%** | 1,51% | 87,15%; 83,15%; 83,15%; 83,71%; 84,83% |
| XGBoost | 84,28% | 1,93% | 87,15%; 85,39%; 83,15%; 81,46%; 84,27% |
| Random Forest | 83,84% | 1,37% | 85,47%; 83,15%; 82,02%; 83,15%; 85,39% |
| Regressão logística | 82,94% | 1,78% | 84,92%; 82,58%; 79,78%; 83,15%; 84,27% |
| Rede neural (MLP 64×32) | 81,03% | 2,03% | 79,33%; 83,71%; 78,09%; 82,02%; 82,02% |

**Modelo escolhido: Gradient Boosting** (scikit-learn, 150 árvores, profundidade 3). A diferença para o XGBoost é pequena e está dentro do desvio dos folds; o Gradient Boosting levou a melhor média e um pouco mais de estabilidade.

A submissão prevista classifica **159 de 418** passageiros do teste como sobreviventes (38,0%), taxa próxima à do treino (38,4%).

> A acurácia da validação cruzada costuma ficar um pouco acima da leaderboard pública (soluções sólidas no Titanic giram em torno de 77–80% lá). O teste oculto é pequeno e ruidoso; perseguir a leaderboard é a forma mais rápida de superajustar.

## Como executar

### 1. Ambiente

```bash
git clone https://github.com/SEU_USUARIO/kaggle-titanic.git
cd kaggle-titanic
python3 -m venv .venv
source .venv/bin/activate   # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dados

Os CSVs **não entram no repositório**. Baixe `train.csv` e `test.csv` em [kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data) e coloque-os em `data/`:

```text
data/train.csv    # 891 passageiros + cabeçalho
data/test.csv     # 418 passageiros + cabeçalho
```

É preciso ter uma conta no Kaggle (gratuita). Pela linha de comando, com a [API oficial](https://github.com/Kaggle/kaggle-api) configurada:

```bash
kaggle competitions download -c titanic -p data
unzip data/titanic.zip -d data
```

### 3. Treino e submissão

```bash
python -m src.train
```

O comando compara os modelos, escolhe o de maior acurácia CV e escreve `submissions/submission.csv`.

### 4. Notebook

```bash
jupyter notebook notebooks/titanic.ipynb
```

O notebook já foi executado ponta a ponta neste repositório (figuras e saídas salvas). Para reexecutar:

```bash
jupyter nbconvert --to notebook --execute notebooks/titanic.ipynb --inplace
```

Depois, envie `submissions/submission.csv` em [kaggle.com/competitions/titanic/submit](https://www.kaggle.com/competitions/titanic/submit).

## Estrutura

```text
kaggle-titanic/
├── README.md
├── requirements.txt
├── notebooks/titanic.ipynb      # EDA, modelagem e submissão (pt-BR)
├── src/
│   ├── features.py              # TitanicFeatureEngineer + preprocessor
│   └── train.py                 # CV, escolha do modelo e submissão
├── data/                        # train.csv e test.csv (não versionados)
└── submissions/submission.csv   # 418 previsões no formato do Kaggle
```

## Publicando no GitHub

```bash
cd kaggle-titanic
git init
git add .
git commit -m "Solução completa da competição Titanic do Kaggle"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/kaggle-titanic.git
git push -u origin main
```

Troque `SEU_USUARIO` pelo seu usuário. Confira se `data/train.csv` e `data/test.csv` não entraram no commit (eles estão no `.gitignore`).

As regras da competição Titanic permitem publicar a solução. Use este repositório como evidência do processo; ele não substitui, sozinho, um projeto de deep learning.

## Próximos passos

- Enviar a submissão e anotar a pontuação pública da leaderboard.
- Ajustar com cuidado um ou dois hiperparâmetros do Gradient Boosting, sempre olhando a CV — não a leaderboard.
- Entrar em uma [Playground Series](https://www.kaggle.com/competitions?hostSegmentIdFilter=8), que abre todo mês, tem ranking de verdade e dados ainda tabulares.
- Em seguida, uma competição de **imagem ou texto**, se a meta for mostrar treinamento de redes neurais.
