# Detección de contenido sobre trastornos alimentarios en tuits en español

Proyecto de **Desarrollo de Aplicaciones Avanzadas** (Tec). Clasificación binaria
de tuits en español como **`anorexia`** (contenido que promueve, expresa o gira
en torno a un trastorno de la conducta alimentaria) vs **`control`** (cualquier
otro contenido: comida sana, fitness, recetas, temas no relacionados).

## Equipo 5

- Kevin Santiago Castro Torres
- Jennifer Nahomi Jasso Hernández
- Min Che Kim

---

## 1. Objetivo

Comparar de forma justa **tres familias de enfoques** para la misma tarea, sobre
el mismo conjunto de datos y métrica:

| Familia | Miembros |
|---|---|
| **Representaciones + clasificador clásico** | TF-IDF (n-gramas de caracteres) · BETO (embeddings congelados) · BETO *fine-tuned* |
| **Clasificadores clásicos** | LinearSVC · Random Forest · Regresión Logística |
| **LLMs (locales, vía Ollama)** | Llama 3.2 3B · Qwen 2.5 3B · Gemma 2 2B (zero-/few-shot) |

La pregunta central: *¿qué generaliza mejor a datos no vistos, y a qué costo?*

## 2. Datos

- **Entrenamiento:** `data/raw/data_train.xlsx` — **1500** tuits (804 anorexia /
  696 control ≈ 53.6/46.4 %).
- **Prueba (externo):** `data/raw/data_test_combined.csv` — **375** tuits (201 /
  174), resultado de fusionar `data_test_fold1.csv` + `data_test_fold2.csv`,
  reparar la codificación (`ftfy`) y eliminar duplicados.
- **Sin fuga de datos (verificado):** train y test no comparten ningún
  `tweet_id`, `tweet_text` ni `user_id` (el corpus es **1 tuit por usuario**), así
  que la estimación en test es genuina.

## 3. Estructura del repositorio

```
src/
  config.py              # configuración central (switch FEATURE_METHOD, hiperparámetros)
  main.py                # orquestador: corre el flujo según FEATURE_METHOD
  data/                  # carga, reparación de codificación, limpieza, guardado
  features/              # vectorizer (TF-IDF char_wb) · embeddings (BETO mean-pooled)
  models/                # svm · random_forest · logistic_regression · beto_finetune · llm (Ollama)
  evaluation/            # metrics · compare (tablas) · predictions · plots
  plot_classical_errors.py  # diagnósticos por CV + figuras (comparación, sobreajuste, ROC/AUC)
  plot_eda.py            # EDA (clases, longitud, términos) + matrices de confusión
tests/                   # 61 pruebas unitarias (unittest)
reports/figures/         # figuras generadas
data/raw/                # datasets
requirements.txt
```

## 4. Metodología

**Preprocesamiento (según el modelo):**
- **TF-IDF:** limpieza agresiva — minúsculas, quita URLs/menciones, normaliza
  hashtags, expande jerga, *stemming* (Snowball español).
- **BETO / LLMs:** limpieza ligera (`clean_text_light`) — conserva mayúsculas y
  acentos, sin stemming (BETO es *cased*; los LLMs entienden texto natural).

**Representaciones:**
- **TF-IDF** con `analyzer="char_wb"`, `ngram_range=(2,5)` — n-gramas de
  caracteres que capturan morfología y son robustos a ruido de redes sociales.
- **BETO** (`dccuchile/bert-base-spanish-wwm-cased`): embeddings de 768 dims por
  *mean-pooling* enmascarado de la última capa (encoder congelado).

**Evaluación:** **validación cruzada estratificada de 5 folds** con **F1
weighted** (no accuracy, por el leve desbalance), y una **evaluación final en el
test externo** que confirma la selección por CV (*CV elige → test confirma*).
Para los modelos con score continuo se reporta también **AUC (ROC)**.

## 5. Cómo ejecutar

Requiere el entorno con las dependencias de `requirements.txt` (incluye
`scikit-learn`, `torch`/`transformers` para BETO, `matplotlib`). Para los LLMs,
**Ollama** debe estar corriendo con los modelos descargados
(`ollama pull llama3.2:3b qwen2.5:3b gemma2:2b`).

El flujo se controla con `FEATURE_METHOD` en `src/config.py`:

| `FEATURE_METHOD` | Qué corre |
|---|---|
| `"tfidf"` | TF-IDF + LinearSVC |
| `"beto"` | BETO + clasificadores |
| `"grid"` | **rejilla 2×3**: {TF-IDF, BETO} × {LinearSVC, RandomForest, LogisticReg} |
| `"beto_finetune"` | BETO afinado end-to-end |
| `"llm"` | comparación de los 3 LLMs (zero-/few-shot según `LLM_N_FEW_SHOT`) |

```bash
cd src
python main.py                 # corre el flujo seleccionado en config.py
python plot_classical_errors.py  # figuras de diagnóstico de los clásicos
python plot_eda.py             # figuras de EDA + matrices de confusión
python -m unittest discover -s ../tests   # 61 pruebas
```

- **Few-shot vs zero-shot LLM:** `LLM_N_FEW_SHOT = 4` (few-shot) o `0` (zero-shot).
  El prompt pide un JSON `{"label", "prob_anorexia"}`; la probabilidad habilita
  el AUC de los LLMs (confianza auto-reportada, útil para *ranking*).
