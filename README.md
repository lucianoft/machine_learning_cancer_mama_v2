# Tech Challenge — Câncer de Mama (Fase 2 · Projeto 1)

Predição **benigno vs maligno** com Machine Learning (Breast Cancer Wisconsin), agora com
hiperparâmetros otimizados por **algoritmo genético**, explicações em linguagem natural
via **LLM** e API REST em Docker com observabilidade.

Continuação da Fase 1 (`ENTREGA_FASE_1.pdf`, mantida intacta em
[`desafio/`](../desafio)) — este repositório (`desafioFase2/`) é a entrega da Fase 2.

## Estrutura do projeto

```
desafioFase2/
├── Tech_Challenge_Breast_Cancer.ipynb   # Notebook (EDA, baseline — herdado da Fase 1)
├── breast_cancer_wisconsin.csv          # Dataset
├── ga/                                   # Algoritmo genético (hiperparâmetros)
│   ├── data.py, search_space.py, individual.py
│   ├── operators.py, fitness.py, genetic_algorithm.py
│   └── run_ga_experiments.py             # roda os experimentos, gera reports/
├── llm/                                  # Integração com LLM (Groq ou OpenAI, configurável)
│   ├── prompts.py, explainer.py, evaluation.py
├── api/                                  # FastAPI (/predict, /predict/explain, /metrics)
├── tests/                                # pytest — API, GA, LLM (mockada)
├── reports/                              # saída dos experimentos do GA (csv + png)
├── models/                               # pipelines treinadas (.pkl) — original e otimizada
├── postman/                              # Collection para testes manuais
├── RELATORIO_TECNICO.md                  # Relatório técnico (Fase 1 + Fase 2)
├── ARQUITETURA.md                        # Diagrama e descrição da arquitetura
├── docs/arquitetura_sistema.png          # Imagem do diagrama
├── ENTREGA_FASE_2.pdf                    # PDF de entrega (Fase 2)
├── requirements.txt / requirements-api.txt / requirements-dev.txt
├── .env.example                          # LLM_PROVIDER + GROQ_API_KEY / OPENAI_API_KEY
├── Dockerfile
└── docker-compose.yml
```

## Ambiente virtual

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt   # já inclui requirements-api.txt + pytest/httpx
```

## 1. Notebook (baseline — igual à Fase 1)

```bash
pip install -r requirements.txt
jupyter notebook Tech_Challenge_Breast_Cancer.ipynb
```

Gera `models/pipeline_breast_cancer.pkl` (baseline, hiperparâmetros manuais).

## 2. Algoritmo genético — otimização de hiperparâmetros

```bash
python -m ga.run_ga_experiments
```

Roda ≥3 experimentos (população/taxa de mutação diferentes) para `logistic_regression`
e `random_forest`, salva:

- `reports/ga_convergence_<modelo>_<experimento>.png` — curva de convergência
- `reports/ga_experiments_results.csv` — resultado de cada experimento
- `reports/ga_vs_baseline.csv` — comparativo otimizado vs. original (mesmo treino/teste)
- `models/pipeline_breast_cancer_ga_optimized.pkl` (+ metadata) — pipeline vencedora,
  servida pela API por padrão

Detalhes da codificação, operadores e fitness: `ARQUITETURA.md` e `RELATORIO_TECNICO.md`.

## 3. LLM — avaliação da qualidade das explicações

```bash
cp .env.example .env   # LLM_PROVIDER=groq por padrão — preencha GROQ_API_KEY
python -m llm.evaluation
```

Roda a explicação em casos representativos do teste e salva `reports/llm_evaluation.csv`
(rubrica de avaliação — ver `RELATORIO_TECNICO.md`).

## 4. Testes automatizados

```bash
pytest
```

Cobre API (`TestClient`, LLM mockada), `ModelService`, operadores e fitness do GA.

## 5. API REST (Docker)

**Pré-requisito:** arquivos em `models/` (`pipeline_breast_cancer.pkl` no mínimo — gerado
pelo notebook; `pipeline_breast_cancer_ga_optimized.pkl` é o preferido, gerado pelo GA).

```bash
cp .env.example .env   # preencha GROQ_API_KEY ou OPENAI_API_KEY (opcional — só pro /predict/explain)
docker compose up --build
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Métricas Prometheus: http://localhost:8000/metrics
- Postman: `postman/Tech_Challenge_Breast_Cancer.postman_collection.json`

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status da API |
| GET | `/metadata` | Metadados do modelo |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/predict` | Classificação benigno/maligno |
| POST | `/predict/explain` | Classificação + explicação em linguagem natural (LLM) |

### Exemplos

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @api/example_request.json

curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d @api/example_request.json
```

### Sem Docker

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

## 6. Dataset

O arquivo `breast_cancer_wisconsin.csv` já está no repositório (569 amostras). Fontes oficiais:

- [Kaggle — Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Variável alvo:** `diagnosis` — `B` = benigno, `M` = maligno (no notebook/API: 0 = benigno, 1 = maligno).

As features são medidas de **núcleos celulares** extraídas de imagens de PAAF. Cada uma das 10 características abaixo aparece em três versões:

| Sufixo | Significado |
|--------|-------------|
| `mean` | Média dos valores por núcleo |
| `error` | Erro padrão |
| `worst` | Média dos três maiores valores |

> Na API (`POST /predict`), envie as **30 features** (sem `id` nem `diagnosis`). Exemplo em `api/example_request.json`. Lista completa das colunas: ver Fase 1 (`desafio/README.md`) — inalterada nesta fase.

## 7. Relatório técnico

Documentação completa — Fase 1 (EDA, baseline) + Fase 2 (GA, LLM, comparativos): `RELATORIO_TECNICO.md`.

## 8. Entrega Fase 2 (PDF)

Arquivo de entrega: **`ENTREGA_FASE_2.pdf`** — inclui: mapa dos entregáveis, diagrama de
arquitetura atualizado, gráficos de convergência do GA e comparativo otimizado vs.
original, prints reais da API (Swagger, `/predict`, `/predict/explain`, `/metrics`).

> Ferramenta de **apoio à decisão**. Não substitui avaliação médica nem biópsia.
