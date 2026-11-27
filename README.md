# Titanic — do dado bruto à submissão

Solução completa da competição introdutória [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic), no Kaggle. O objetivo é prever, para cada passageiro, se ele sobreviveu ao naufrágio (`Survived = 1` ou `0`). A métrica oficial é a **acurácia**.

Autor: **Vinicius Brum**.

Este repositório foi pensado para portfólio: pipeline de atributos reutilizável (sem vazamento entre treino, validação e teste), validação cruzada honesta, notebook de exploração e um arquivo de submissão pronto para o Kaggle.

> **Recado sincero sobre o ranking.** O topo da leaderboard pública do Titanic tem centenas de notas de **100%**. Esse resultado não vem de um modelo melhor: a lista completa de quem sobreviveu no conjunto de teste é pública na internet, e muitas submissões só copiam essa lista. Isso não ensina nada e não deve ir para um portfólio. No mundo real, as melhores soluções **legítimas** ficam em torno de **80–82%** na leaderboard pública. Qualquer coisa consistentemente acima de 0,80 já está entre as soluções sérias. Esta solução não usa (e não deve usar) rótulos vazados do teste.

> **Sobre redes neurais.** O Titanic é um problema tabular pequeno (~900 linhas). Ele demonstra que você sabe conduzir o ciclo completo de um projeto de classificação (exploração, engenharia de atributos, validação e submissão). Para evidenciar treino de redes neurais de verdade, o próximo passo é uma competição de imagem ou de texto.

## O problema

O Kaggle fornece 891 passageiros com o resultado conhecido (`train.csv`) e 418 cujo resultado está oculto (`test.csv`). Cada linha traz classe da passagem, nome, sexo, idade, parentes a bordo, ticket, tarifa, cabine e porto de embarque. Você treina um modelo no conjunto rotulado, envia as 418 previsões e entra no ranking.

A competição não tem prazo nem prêmio em dinheiro: o valor está em aprender o fluxo e, se quiser, documentar o trabalho no GitHub.

## Abordagem

1. **Exploração** das relações entre sobrevivência, sexo, classe, idade, família e cabine (veja `notebooks/titanic.ipynb`).
2. **Engenharia de atributos** em um `Transformer` do scikit-learn, com estatísticas aprendidas só no fold de treino:
   - título social extraído do nome (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`);
   - tamanho da família, indicador de quem viajava sozinho e tarifa por pessoa;
   - imputação de idade pela mediana do título;
   - imputação de tarifa pela mediana da classe, mais faixas em quantis;
   - convés da cabine (`Deck`) e indicador de cabine conhecida;
   - interação `Pclass` × `Sex` (por exemplo, `1_female`);
   - **`FamilySurvival`**, o atributo que mais costuma fazer diferença nas soluções legítimas: agrupa passageiros pelo par *sobrenome + tarifa* e, se ainda não houver sinal, pelo *ticket*. A taxa do grupo usa **apenas rótulos do fold de treino**, em regime **leave-one-out** — a sobrevivência de um passageiro nunca entra no atributo dele. Sem companheiros rotulados, o valor padrão é 0,5. O tamanho do grupo (`GroupSize`) pode olhar treino e teste juntos, porque isso não usa o alvo.
3. **Pré-processamento** com `Pipeline` + `ColumnTransformer`: padronização das numéricas e *one-hot* das categóricas.
4. **Comparação** de modelos regularizados (regressão logística, Random Forest rasa, Gradient Boosting e XGBoost rasos, SVC calibrado e KNN) com validação cruzada **estratificada de 10 folds, repetida 3 vezes** (30 pontuações, `random_state=42`).
5. **Ensemble** de votação suave com os melhores modelos (os que ficam a até 1 ponto percentual da melhor média). A escolha final olha a média da CV e, em caso de empate prático (menos de 0,1 p.p.), o menor desvio.
6. Retreino do modelo escolhido em todo o conjunto de treino e geração de `submissions/submission.csv`. A versão anterior (sem `FamilySurvival`, CV de 5 folds) ficou em `submissions/submission_v1.csv`.

## Resultados

Acurácia na validação cruzada estratificada repetida (10 folds × 3 repetições, semente 42, 30 pontuações). Os números abaixo foram calculados neste repositório, só com `train.csv` — nenhum rótulo externo do teste foi usado.

| Modelo | Acurácia CV | Desvio-padrão |
| --- | ---: | ---: |
| Gradient Boosting | 84,77% | 3,43% |
| **Ensemble (votação suave)** | **84,74%** | **3,11%** |
| Regressão logística | 84,66% | 2,85% |
| SVC | 84,66% | 3,40% |
| XGBoost | 84,44% | 3,27% |
| Random Forest | 83,39% | 3,38% |
| KNN | 81,37% | 3,12% |

**Modelo escolhido: ensemble de votação suave** (Gradient Boosting + regressão logística + SVC + XGBoost). A média é praticamente a do Gradient Boosting isolado (diferença de 0,04 ponto percentual, dentro do ruído da CV); o ensemble foi preferido por ser um pouco mais estável. O KNN ficou claramente atrás e não entrou na votação.

Atributos que mais pesaram no modelo final (média normalizada das importâncias/coeficientes dos membros que as expõem): `Title_Mr`, `Sex_female`, **`FamilySurvival`**, `Sex_male`, `Pclass_3`, `PclassSex_1_female`, `FarePerPerson`, `Title_Master`, `Age` e `PclassSex_2_female`. O truque do grupo familiar aparece entre os três sinais mais fortes, logo depois de título e sexo.

A submissão prevista classifica **153 de 418** passageiros do teste como sobreviventes (36,6%), taxa próxima à do treino (38,4%).

> A acurácia da validação cruzada costuma ficar um pouco acima da leaderboard pública. O teste oculto é pequeno (418 linhas) e ruidoso; perseguir a leaderboard é a forma mais rápida de superajustar. **Não afirmamos uma nota na leaderboard** — ela só existe depois do envio no Kaggle. O alvo realista de uma solução legítima continua na faixa de 0,80–0,82 no ranking público.

## Como executar

### 1. Ambiente

```bash
git clone https://github.com/vinibrum10/kaggle-titanic.git
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

O comando compara os modelos, monta o ensemble, escolhe pela CV e escreve `submissions/submission.csv`.

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
├── notebooks/titanic.ipynb         # EDA, modelagem e submissão (pt-BR)
├── src/
│   ├── features.py                 # TitanicFeatureEngineer + preprocessor
│   └── train.py                    # CV repetida, ensemble e submissão
├── data/                           # train.csv e test.csv (não versionados)
└── submissions/
    ├── submission.csv              # 418 previsões (modelo atual)
    └── submission_v1.csv           # versão anterior, conservada para comparação
```

## Publicando no GitHub

```bash
cd kaggle-titanic
git init
git add .
git commit -m "Melhora a solução do Titanic com FamilySurvival e ensemble"
git branch -M main
git remote add origin https://github.com/vinibrum10/kaggle-titanic.git
git push -u origin main
```

Confira se `data/train.csv` e `data/test.csv` não entraram no commit (eles estão no `.gitignore`).

As regras da competição Titanic permitem publicar a solução. Use este repositório como evidência do processo; ele não substitui, sozinho, um projeto de deep learning.

## Próximos passos

- Enviar a submissão e anotar a pontuação pública da leaderboard — sem ficar retreinando em função dela.
- Entrar em uma [Playground Series](https://www.kaggle.com/competitions?hostSegmentIdFilter=8), que abre todo mês, tem ranking de verdade e dados ainda tabulares.
- Em seguida, uma competição de **imagem ou texto**, se a meta for mostrar treinamento de redes neurais.
