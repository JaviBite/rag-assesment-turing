# INSTRUCTIONS.md — Guía para agentes de código

Contexto rápido para trabajar en este repo sin tener que releer todo el código.
Para el enunciado del ejercicio ver [STATEMENT.md](STATEMENT.md), para las
respuestas teóricas [ANSWERS.md](ANSWERS.md) y para la guía de usuario
[README.md](README.md).

## Resumen del proyecto

Prueba técnica (Turing Challenge) dockerizada en 4 servicios:

| Servicio   | Carpeta     | Rol                                                        | Puerto |
|------------|-------------|-------------------------------------------------------------|--------|
| `chroma`   | (imagen oficial) | Vector DB (modo servidor)                              | 8001   |
| `vllm`     | (imagen oficial) | LLM `gemma-4-E2B-it-qat-w4a16-ct` (multimodal + tools) | 8000   |
| `detector` | `detector/` | FastAPI + YOLO (`person`/`car`)                            | 8002   |
| `app`      | `app/`      | Chainlit (UI) + LangGraph (backend) + ingesta + embeddings | 8501   |

## Arquitectura del grafo (LangGraph)

`app/graph/build.py` ensambla:

```
START → orchestrator → (rag | python | chitchat | detect) → summarize → END
```

- **`orchestrator.py`**: clasifica la última pregunta del usuario en `rag` /
  `python` / `chitchat` / `detect` con salida estructurada (`with_structured_output`).
  Tiene un fallback para cuando el modelo (2B) devuelve la etiqueta como texto
  plano en vez de JSON. La ruta `detect` solo se elige si además hay una imagen
  disponible (subida en este turno vía `image_path`, o la última de
  `retrieved_image_paths` como "imagen en contexto"); en ese caso fija
  `detect_image_path`. Si hay `image_path` en el estado y no se pidió
  detección, va a `rag`.
- **`rag_node.py`**: recupera contexto de Chroma (`get_vectorstore().similarity_search`),
  construye el system prompt con el contexto + resumen previo, e inyecta
  imágenes (subidas o recuperadas del RAG) en el último `HumanMessage` como
  contenido multimodal. También expone `chitchat_node`.
- **`detection_node.py`**: nodo de **ejecución estática (sin LLM)**. Llama
  directamente a `tools.detect_objects_in_image` (servicio `detector`) sobre
  `detect_image_path`, dibuja las cajas y devuelve un `AIMessage` con los
  conteos ya formateados.
- **`python_node.py`**: usa `PythonREPLTool` (langchain-experimental) con
  bucle de tool-calling acotado a 3 iteraciones. **Ejecuta código generado por
  el LLM dentro del contenedor de la app sin sandbox real** — limitación
  conocida y documentada, no "arreglarla" silenciosamente sin avisar.
- **`summarize_node.py`**: memoria dinámica. Si los mensajes superan
  `SUMMARY_TOKEN_THRESHOLD` (heurística ~4 chars/token), resume todo menos los
  últimos `keep_last_messages` y emite `RemoveMessage` para purgarlos del
  estado persistido (`SqliteSaver`, en `data/checkpoints.sqlite`).
- **`state.py`** (`GraphState`): `messages` (reducer `add_messages`), `summary`,
  `route`, `image_path`, `retrieved_image_paths`, `detect_image_path`.

## Estructura del repo

```
app/
  chainlit_app.py     # entrypoint UI (Chainlit), un thread_id por sesión
  config.py           # Settings (pydantic-settings, lee .env y env vars)
  llm.py              # get_chat_model (ChatOpenAI -> vLLM), describe_image, ask_about_image
  embeddings.py       # HuggingFaceEmbeddings BAAI/bge-m3 (CPU/GPU)
  vectorstore.py       # Chroma HttpClient (lru_cache)
  ingest.py           # CLI: docs/*.pdf -> texto+imágenes en Chroma + extracción de formularios
  pdf_processing.py   # PyMuPDF: texto, imágenes embebidas, render de página 1
  extraction.py       # Detección de formulario + extracción estructurada -> data/extractions/*.json
  graph/              # nodos y ensamblado de LangGraph (ver arriba)
detector/
  main.py             # FastAPI + YOLO (yolov8n por defecto, CPU)
tests/                # tests de integración contra servicios EN MARCHA (no mocks)
docs/                 # PDFs a ingestar (gitignored, no versionar)
data/                 # generado: images/, extractions/, checkpoints.sqlite* (mayormente gitignored)
docker-compose.yml        # stack completo (requiere GPU NVIDIA para vllm)
docker-compose.mac.yml     # sin contenedor vllm; usa Ollama nativo del host (Mac/Metal)
Makefile             # atajos: build/up/down/logs, ingest, tests, modo local sin Docker
```

## Cómo arrancar el entorno

- **Docker completo** (requiere GPU NVIDIA + WSL2/NVIDIA Container Toolkit):
  `make build`, `make up`, `make status`, `make logs` / `make logs-vllm`.
- **Mac con Ollama nativo** (sin contenedor vLLM, usa Metal): `make up-mac`
  (requiere `ollama serve` + `ollama pull $OLLAMA_MODEL`, por defecto `gemma4:e2b`).
- **Local sin Docker**: `.venv` con `app/requirements.txt` instalado, luego
  `make chroma-local` + `make start-local` (Chainlit en `:8501`). Las llamadas
  a vLLM/detector fallan con gracia si no están disponibles en localhost.
- **Ingesta**: `make ingest` (incremental) / `make ingest-reset` (vacía
  colección antes). Versión local: `make ingest-local[-reset]`.

El Makefile fuerza `SHELL := bash` (Git Bash en Windows) porque usa sintaxis
POSIX (`mkdir -p`, `export`, `&&`...).

## Configuración

- `.env` (copiar de `.env.example`): `MODEL_ID`, `HUGGING_FACE_HUB_TOKEN`,
  `VLLM_MAX_LEN`, `VLLM_GPU_UTIL`, `SUMMARY_TOKEN_THRESHOLD`.
- `app/config.py` (`Settings`, pydantic-settings): URLs de vLLM/Chroma/detector,
  modelo de embeddings, umbrales de memoria, rutas (`docs_dir`, `data_dir`).
  Todo overrideable por variable de entorno (en docker-compose o `make
  start-local`).

## Tests

Los tests en `tests/` son de **integración**, esperan que los servicios estén
arrancados (Docker o local) y NO mockean APIs externas:

- `make test-detector` — `curl` al servicio YOLO con `image.jpg`.
- `make test-rag` / `make test-python` / `make test-memory` — invocan el grafo
  completo dentro del contenedor `app` (`docker compose run --rm app python
  /srv/app/tests/test_*.py`).
- `make test-all` — los cuatro en secuencia.

## Convenciones de código

- Comentarios, docstrings y mensajes de cara al usuario **en español** —
  mantener esa consistencia en código nuevo.
- `from __future__ import annotations` + type hints (`str | None`, etc.) en
  todos los módulos de `app/`.
- Módulos pequeños y de responsabilidad única; los nodos del grafo viven en
  `app/graph/` y reciben/devuelven `dict` parciales sobre `GraphState`.
- Salidas estructuradas del LLM con `with_structured_output(..., method="json_schema")`
  + modelos Pydantic (ver `orchestrator.py`, `extraction.py`). El modelo es un
  2B y a veces incumple el formato — cuando sea relevante, contempla un
  fallback de parseo como en `orchestrator.route_selector`.

## Cosas a tener en cuenta / limitaciones conocidas

- `python_node` ejecuta código Python generado por el LLM sin sandbox real
  (limitación documentada en README, no introducir como "bug nuevo").
- `detector` corre `yolov8n` en **CPU** por defecto para dejar la VRAM a vLLM.
- No versionar: `docs/*.pdf`, `*.pt` (incluye `yolov8n.pt`), `.env`,
  `data/images/`, `data/extractions/`, `data/*.sqlite*` — todo está en
  `.gitignore`. `data/checkpoints.sqlite-shm/-wal` son artefactos locales del
  checkpointer SQLite (memoria de conversación), no deberían tocarse en commits.
- `STATEMENT.md` es el enunciado del reto (no modificar su contenido salvo que
  se pida explícitamente); `ANSWERS.md` son las respuestas entregadas.
