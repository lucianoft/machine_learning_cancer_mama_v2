# Relatório Técnico — Tech Challenge Fase 1 + Fase 2
## Predição de Câncer de Mama (Wisconsin Diagnostic)

> Este documento cobre as duas fases: as seções 1–5 são a entrega da Fase 1 (EDA,
> modelos, arquitetura da API), inalteradas. As seções 6–9 são a entrega da Fase 2 —
> Projeto 1 (otimização via algoritmo genético + interpretação via LLM).

**Projeto:** Sistema de apoio à decisão clínica com Machine Learning  
**Repositório Git:** https://github.com/lucianoft/machine_learning_cancer_mama  
**Dataset:** `breast_cancer_wisconsin.csv` (incluso no repositório)  
**Download alternativo:** [Kaggle - Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
**Variável alvo:** `diagnosis` (0 = benigno, 1 = maligno)  
**Notebook:** `Tech_Challenge_Breast_Cancer.ipynb`

### Mapa dos entregáveis (Fase 1)

| Exigência do PDF | Onde está |
|------------------|-----------|
| Link do repositório Git | https://github.com/lucianoft/machine_learning_cancer_mama |
| Código-fonte completo | Repositório (`api/`, notebook, `models/`) |
| README com instruções | `README.md` |
| Dockerfile | `Dockerfile` + `docker-compose.yml` |
| Dataset ou link | `breast_cancer_wisconsin.csv` + links Kaggle/UCI acima |
| Resultados (prints, gráficos) | `Tech_Challenge_Breast_Cancer.ipynb` (seções 2, 6 e 7) + `ENTREGA_FASE_1.pdf` |
| Arquitetura do sistema | `ARQUITETURA.md` + `docs/arquitetura_sistema.png` |
| Relatório técnico | Este arquivo (`RELATORIO_TECNICO.md`) |

### Contexto clínico

O dataset **Breast Cancer Wisconsin (Diagnostic)** contém **569 amostras** de massas mamárias analisadas por **PAAF** (Punção Aspirativa por Agulha Fina). Para cada núcleo celular foram calculadas **30 medidas numéricas** (média, erro padrão e pior valor de 10 características morfológicas), permitindo classificar o tumor como **benigno** ou **maligno**.

---

## 1. Discussões da Análise Exploratória

### 1.1 Contexto do problema

O conjunto foi curado por Wolberg, Street e Mangasarian (University of Wisconsin) e está disponível no repositório UCI. É um benchmark clássico em classificação médica e saúde da mulher — tema central do Tech Challenge.

A variável alvo é **`diagnosis`**: `B` (benigno) ou `M` (maligno), codificada como 0 e 1 no pipeline.

### 1.2 Dimensão e balanceamento

| Aspecto | Valor |
|---------|-------|
| Registros | 569 |
| Atributos | 32 colunas (id + diagnosis + 30 features) |
| Benigno (0) | 357 (62,7%) |
| Maligno (1) | 212 (37,3%) |

O dataset é **moderadamente balanceado** em comparação com problemas de triagem por questionário, o que permite métricas mais estáveis (recall, F1 e accuracy).

### 1.3 Valores ausentes

**Nenhum valor ausente** no dataset diagnostic. Não foi necessária imputação agressiva; o `SimpleImputer` no pipeline permanece por consistência e robustez em produção.

### 1.4 Variáveis relevantes

As 30 features derivam de 10 medidas de núcleos celulares:

- **Geométricas:** raio, perímetro, área
- **Textura:** desvio padrão de tons de cinza
- **Forma:** suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal

Cada medida aparece em três versões: **mean** (média), **error** (erro padrão) e **worst** (média dos três maiores valores).

### 1.5 Correlação

A matriz de correlação mostra forte associação entre raio, perímetro e área — esperado por definição geométrica. Features `worst_*` tendem a correlacionar fortemente com o diagnóstico maligno.

### 1.6 Conclusões da EDA

1. Dataset adequado ao desafio de classificação em saúde feminina.
2. Balanceamento razoável permite usar accuracy, recall e F1 de forma complementar.
3. Features derivadas diretamente da amostra citológica — contexto de **apoio ao diagnóstico pós-PAAF**, não triagem populacional por questionário.

---

## 2. Estratégias de Pré-processamento

### 2.1 Seleção de features (30 colunas)

| Excluída | Motivo |
|----------|--------|
| `id` | Identificador da amostra |
| `diagnosis` | Variável alvo |

### 2.2 Pipeline sklearn

```
SimpleImputer(median) → StandardScaler → Classificador
```

Encapsulado em `ColumnTransformer` + `Pipeline` para uso idêntico no treino e na API REST.

### 2.3 Divisão dos dados

- **80% treino** (455) / **20% teste** (114)
- Estratificação por `diagnosis`
- `random_state=42`

### 2.4 Balanceamento

`class_weight='balanced'` em Regressão Logística, Árvore e Random Forest.

---

## 3. Modelos Usados e Porquê

| Modelo | Justificativa |
|--------|---------------|
| **Regressão Logística** | Baseline interpretável; excelente em dados linearmente separáveis |
| **Árvore de Decisão** | Captura regras não lineares entre medidas morfológicas |
| **Random Forest** | Ensemble robusto; referência em benchmarks tabulares |
| **KNN** | Compara amostras similares; referência do notebook de ML avançado |

Critério de seleção para deploy: **maior F1-Score** no teste.

---

## 4. Resultados e Interpretação dos Dados

### 4.1 Métricas comparativas (teste — 114 amostras)

| Modelo | Accuracy | Recall | F1-Score |
|--------|----------|--------|----------|
| **Logistic Regression** | **97,4%** | **95,2%** | **96,4%** |
| Random Forest | 97,4% | 92,9% | 96,3% |
| K-Nearest Neighbors | 95,6% | 90,5% | 93,8% |
| Decision Tree | 90,4% | 83,3% | 86,4% |

### 4.2 Regressão Logística — detalhamento

```
              precision    recall  f1-score   support

     Benigno       0.97      0.99      0.98        72
     Maligno       0.98      0.95      0.96        42

    accuracy                           0.97       114
```

### 4.3 Interpretação

- **Regressão Logística** apresenta melhor **F1-Score (96,4%)** e foi salva como pipeline final para REST.
- **Recall de 95,2%** para malignos — poucos falsos negativos, crítico em contexto oncológico.
- Métricas altas refletem que as features são extraídas do **mesmo material citológico** usado no diagnóstico — problema bem definido e com forte sinal preditivo.
- **Falsos negativos** (maligno classificado como benigno) permanecem o erro mais grave clinicamente.

### 4.4 Uso prático

1. **Apoio à decisão** em análise de PAAF — segunda opinião automatizada.
2. **API REST** — pipeline em `models/pipeline_breast_cancer.pkl`.
3. **Integração** com sistemas de patologia digital (mediante extração das 30 medidas).

### 4.5 Limitações

- Amostra de **569 casos** de uma única origem (Wisconsin, anos 1990).
- Features exigem **imagem digitalizada de PAAF** — não aplicável a triagem só com dados demográficos.
- Não substitui avaliação médica, mamografia nem biópsia definitiva.
- **O profissional de saúde tem a palavra final.**

### 4.6 Conclusão

O projeto atende ao Tech Challenge com classificação em saúde feminina, EDA documentada, pipeline reprodutível e modelo serializado para REST. O dataset Wisconsin oferece base mais estável que problemas fortemente desbalanceados por questionário, com métricas adequadas para demonstração e deploy.

---

## 5. Arquitetura do Sistema

### 5.1 Visão geral

O sistema foi estruturado em **camadas** para separar treino, deploy e consumo:

1. **Dados** — `breast_cancer_wisconsin.csv` (569 amostras, 30 features).
2. **Treino** — notebook Jupyter realiza EDA, compara modelos e serializa a pipeline com `joblib`.
3. **Artefatos** — `pipeline_breast_cancer.pkl` e `pipeline_metadata.pkl` em `models/`.
4. **API** — FastAPI (`api/main.py`) carrega o modelo via `ModelService` e expõe REST.
5. **Deploy** — Docker (`Dockerfile` + `docker-compose.yml`) na porta 8000.
6. **Clientes** — Swagger (`/docs`), Postman ou sistemas clínicos via HTTP.

### 5.2 Fluxo de predição

```
Cliente → POST /predict (JSON, 30 features)
       → Validação Pydantic (schemas.py)
       → ModelService.predict()
       → Pipeline sklearn (imputer + scaler + classificador)
       → JSON { predicao, label, probabilidades }
```

### 5.3 Diagrama

Documentação completa e diagramas em **`ARQUITETURA.md`** e imagem em **`docs/arquitetura_sistema.png`**.

| Componente | Arquivo | Responsabilidade |
|------------|---------|------------------|
| API HTTP | `api/main.py` | Rotas `/health`, `/metadata`, `/predict` |
| Contrato | `api/schemas.py` | Validação das 30 features de entrada |
| ML | `api/model_service.py` | Carrega `.pkl` e executa predição |
| Container | `Dockerfile` | Imagem Python 3.11 + dependências da API |

### 5.4 Justificativa

- **Pipeline única** evita divergência entre treino e produção.
- **FastAPI** gera Swagger automaticamente para documentação e testes.
- **Docker** garante ambiente reproduzível na entrega e avaliação.

---

# Fase 2 — Projeto 1: Otimização via Algoritmo Genético + LLM

## 6. Algoritmo genético

### 6.1 Codificação (representação de genes)

Cada indivíduo é um `dict {hiperparâmetro: valor}` — não uma bitstring. Pra um espaço
de busca misto (inteiro, float em escala log, categórico), um dict tipado é decodificado
direto pros kwargs do classificador sklearn, sem uma etapa de codificação/decodificação
binária que só adicionaria uma fonte de bug sem ganho real. Dois espaços de busca
(`ga/search_space.py`):

| Modelo | Genes | Por quê |
|---|---|---|
| `logistic_regression` (modelo em produção na Fase 1) | `C` (log-float 0.001–100), `penalty` (l1/l2) | Poucos hiperparâmetros relevantes; comparação direta "antes vs. depois" do modelo já servido |
| `random_forest` (2º lugar na Fase 1) | `n_estimators` (30–150), `max_depth` (2–30), `min_samples_split` (2–20), `min_samples_leaf` (1–10), `max_features` (sqrt/log2/None) | Espaço de busca bem mais rico — melhor pra ilustrar a busca genética |

`penalty=l1` só é suportado pelo solver `liblinear` no scikit-learn — o solver é
**derivado** do gene `penalty` na decodificação (`ga/individual.py`), não sorteado
independente, pra nunca gerar uma combinação inválida.

### 6.2 Operadores (`ga/operators.py`)

- **Seleção**: torneio com `k=3` — sorteia 3 indivíduos, o de maior fitness vence.
- **Crossover**: uniforme por gene — cada gene do filho vem de um dos pais com 50% de
  chance; com probabilidade `1 - crossover_rate` os pais passam direto (clonados).
- **Mutação**: resample — cada gene tem `mutation_rate` de chance de ser resorteado
  dentro dos limites do espaço de busca. Preferido a uma perturbação incremental porque
  sempre gera um valor válido, sem precisar de lógica de clipping por tipo de gene.
- **Elitismo**: os 2 melhores indivíduos de cada geração passam direto pra próxima, sem
  crossover/mutação.

### 6.3 Função fitness (`ga/fitness.py`)

Reconstrói a pipeline (mesmo `Imputer → StandardScaler` do notebook da Fase 1) com os
hiperparâmetros do indivíduo e avalia por `StratifiedKFold` (cv=3) **só no treino** —
nunca no teste, senão o GA otimizaria em cima do conjunto que precisa ficar de fora pra
comparação final. Combina três métricas num fitness escalar:

```
fitness = 0.25·accuracy + 0.45·recall + 0.30·f1
```

Recall pesa mais que accuracy porque falso negativo (amostra maligna classificada como
benigna) é o erro mais grave nesse contexto clínico — o mesmo raciocínio já registrado
na seção 4.3 deste relatório (Fase 1). Combinações de hiperparâmetros inválidas pro
scikit-learn são penalizadas com fitness 0 em vez de derrubar o experimento inteiro.

### 6.4 Experimentos (`ga/run_ga_experiments.py`)

Três configurações de GA, aplicadas às duas famílias de modelo (`population_size=16` é
o baseline; população maior e mutação alta testam sensibilidade a esses parâmetros):

| Experimento | População | Mutação | Gerações |
|---|---|---|---|
| `baseline_ga` | 16 | 0,10 | 10 |
| `populacao_maior` | 32 | 0,10 | 10 |
| `mutacao_alta` | 16 | 0,35 | 10 |

**Resultado de cada experimento** (fitness/métricas de cross-validation no treino, e
métricas no teste held-out reajustando o melhor indivíduo no treino completo):

| Modelo | Experimento | Fitness CV | Acc. CV | Recall CV | F1 CV | Acc. teste | Recall teste | F1 teste |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | baseline_ga | 0,9669 | 0,9736 | 0,9647 | 0,9648 | 98,2% | 97,6% | 97,6% |
| logistic_regression | populacao_maior | 0,9684 | 0,9758 | 0,9647 | 0,9676 | 98,2% | 97,6% | 97,6% |
| logistic_regression | mutacao_alta | 0,9684 | 0,9758 | 0,9647 | 0,9676 | 98,2% | 97,6% | 97,6% |
| random_forest | baseline_ga | 0,9519 | 0,9582 | 0,9530 | 0,9449 | 96,5% | 92,9% | 95,1% |
| random_forest | populacao_maior | 0,9518 | 0,9582 | 0,9530 | 0,9447 | 97,4% | 92,9% | 96,3% |
| random_forest | mutacao_alta | 0,9532 | 0,9604 | 0,9530 | 0,9475 | 96,5% | 92,9% | 95,1% |

Gráficos de convergência (melhor fitness e fitness médio por geração) de cada
experimento: `reports/ga_convergence_<modelo>_<experimento>.png`.

### 6.5 Comparativo: otimizado vs. original

Baseline recalculado no mesmo treino/teste (não copiado do relatório da Fase 1, pra
comparação justa), com os hiperparâmetros exatamente como no notebook original:

| Modelo | Accuracy original | Recall original | F1 original | Accuracy GA | Recall GA | F1 GA |
|---|---|---|---|---|---|---|
| **logistic_regression** | 97,4% | 95,2% | 96,4% | **98,2%** | **97,6%** | **97,6%** |
| random_forest | 97,4% | 92,9% | 96,3% | 97,4% | 92,9% | 96,3% |

**Vencedor: `logistic_regression`**, hiperparâmetros `C=0,640067`, `penalty='l2'`
(`solver` derivado: `liblinear`) — ganho real de F1 (96,4% → 97,6%) e de recall
(95,2% → 97,6%, ou seja, menos falsos negativos), não só uma variação de ruído. Pipeline
salva em `models/pipeline_breast_cancer_ga_optimized.pkl`, servida pela API por padrão.

Pro `random_forest`, o GA encontrou hiperparâmetros diferentes do baseline
(`n_estimators=149`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`,
`max_features='sqrt'`, contra `n_estimators=200` e defaults no original) mas o
resultado no teste empatou com o original. Isso é esperado e coerente, não um bug: o
dataset é pequeno (114 amostras de teste) e o baseline da Fase 1 já estava perto do
teto de desempenho possível pra esse problema — o valor do GA aqui está no *processo*
(codificação, operadores, fitness, comparação reprodutível), não em bater recorde a
qualquer custo.

## 7. Integração com LLM

### 7.1 Abordagem

`POST /predict/explain` (rota separada do `/predict` original) recebe a mesma entrada,
roda a predição e manda o resultado pra uma LLM (`llm/explainer.py`), que devolve uma
explicação em linguagem natural voltada a profissional de saúde. O provedor é
configurável via `LLM_PROVIDER=groq|openai` (default `groq` — API compatível com a da
OpenAI, gratuita, servindo Llama 3, um dos modelos sugeridos pelo próprio PDF do
desafio) — o mesmo SDK (`openai`) atende os dois, só trocando `base_url`/chave.

As features "mais relevantes" citadas na explicação não são escolhidas por uma lista
fixa: usamos o `StandardScaler` já treinado dentro da pipeline pra calcular o
`|z-score|` de cada medida da amostra em relação à média da população de treino, e
citamos as 5 que mais se desviam — ou seja, o que há de mais "atípico" naquela amostra
especificamente.

### 7.2 Prompt engineering

`llm/prompts.py` define um *system prompt* fixando papel, regras e formato (nunca
inventar números, citar 2-3 features relevantes, terminar sempre com o mesmo
disclaimer clínico já usado na API desde a Fase 1) e um *user prompt* com o resultado
da predição + as features relevantes selecionadas. Regras explícitas de "nunca inventar
números" e "responder só com os dados fornecidos" reduzem alucinação sem precisar de
técnicas mais caras (RAG, few-shot com exemplos).

### 7.3 Avaliação da qualidade

`llm/evaluation.py` roda a explicação em 8 casos representativos do conjunto de teste
(mistura de benigno/maligno e confiança alta/baixa) e monta uma rubrica
(`reports/llm_evaluation.csv`):

| Critério | Como é avaliado |
|---|---|
| Contém o disclaimer clínico | Automático — checa a string exata no texto gerado |
| Não veio vazia | Automático |
| Clareza (1–5) | Manual — revisão humana |
| Correção clínica (1–5) | Manual — revisão humana |

Preenchido na entrega final após rodar `python -m llm.evaluation` com uma chave de API
configurada (ver `RELATORIO_TECNICO.md` anexo ao PDF de entrega, seção de prints).

## 8. Desafios enfrentados e soluções

- **Overfitting do GA no próprio processo de seleção** — resolvido avaliando fitness
  por cross-validation no treino, nunca no teste, e só usando o teste held-out na
  comparação final (uma única vez, com o indivíduo já escolhido).
- **Combinações inválidas de hiperparâmetros** (ex.: `min_samples_leaf` maior que o
  número de amostras de uma folha) — capturadas em `ga/fitness.py` e penalizadas com
  fitness 0 em vez de derrubar o experimento inteiro.
- **Dataset pequeno e já perto do teto de desempenho** — o `random_forest` não mostrou
  ganho no teste apesar de hiperparâmetros diferentes; documentado como resultado
  honesto (seção 6.5), não escondido nem forçado.
- **Custo de avaliação do GA pro `random_forest`** — reduzido limitando
  `n_estimators` a 30–150 (em vez de um range maior) e `cv=3`, mantendo o tempo total
  dos 3 experimentos × 2 modelos em poucos minutos sem comprometer a qualidade da busca.
- **Alucinação da LLM** — mitigada com regras explícitas no *system prompt* ("nunca
  invente números", "use só os dados fornecidos") e testada em `tests/test_llm_explainer.py`.

## 9. Escalabilidade e observabilidade

Ver `ARQUITETURA.md`, seção "Escalabilidade e observabilidade" — logging estruturado e
métricas Prometheus (`/metrics`) implementados; auto-scaling em nuvem fica documentado
como proposta, não implementado (decisão consciente de foco no obrigatório — a
implementação em nuvem é explicitamente opcional/pontuação extra no PDF do desafio).

---

## Referências

- UCI ML Repository: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Kaggle: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- Tech Challenge IADT — Fase 1 (`IADT - Fase 1 - Tech challenge A.pdf`)
