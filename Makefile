# Makefile — shortcuts para desarrollo y pruebas
# Requiere: docker compose, curl, python3
#
# IMPORTANTE: forzamos bash como shell para que funcionen mkdir -p, export,
# el operador & y demás sintaxis POSIX (Git Bash lo provee en Windows).
SHELL := bash

.PHONY: help build up down logs ingest ingest-reset \
        test-detector test-rag test-python test-memory test-all eval eval-local \
        status clean \
        chroma-local start-local ingest-local ingest-local-reset stop-local

help:
	@echo ""
	@echo "  build              Construye las imágenes Docker (sin arrancar)"
	@echo "  up                 Arranca todos los contenedores en background"
	@echo "  down               Para y elimina los contenedores"
	@echo "  logs               Sigue los logs de todos los servicios"
	@echo "  logs-vllm          Sigue solo los logs de vLLM (descarga del modelo)"
	@echo "  status             Estado + healthchecks de los 4 servicios"
	@echo ""
	@echo "  ingest             Ingesta los PDFs de docs/ (incremental)"
	@echo "  ingest-reset       Vacía la colección y reingesta"
	@echo ""
	@echo "  test-detector      Prueba el servicio YOLO con una imagen de muestra"
	@echo "  test-rag           Prueba una consulta RAG al chatbot"
	@echo "  test-python        Prueba la generación + ejecución de Python"
	@echo "  test-memory        Prueba el resumen automático de conversación"
	@echo "  test-all           Lanza todos los tests en secuencia"
	@echo "  eval               Evalúa el chatbot con el golden set (métricas + LLM-as-judge)"
	@echo ""
	@echo "  clean              Elimina contenedores, volúmenes y caché de HF"
	@echo ""
	@echo "  Desarrollo local (sin Docker):"
	@echo "  chroma-local       Arranca Chroma en ventana nueva (Windows) o background"
	@echo "  start-local        Arranca Chainlit apuntando a servicios localhost"
	@echo "  ingest-local       Ingesta docs/ en la Chroma local"
	@echo "  ingest-local-reset Vacía colección local y reingesta"
	@echo "  stop-local         Para el proceso Chroma lanzado en background"
	@echo ""

# ─── Docker ──────────────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d
	@echo "Esperando healthchecks... (vLLM puede tardar 3-5 min descargando el modelo)"
	@docker compose ps

down:
	docker compose down

logs:
	docker compose logs -f

logs-vllm:
	docker compose logs -f vllm

status:
	docker compose ps

# ─── Ingesta ─────────────────────────────────────────────────────────────────

ingest:
	docker compose run --rm app python -m app.ingest

ingest-reset:
	docker compose run --rm app python -m app.ingest --reset

# ─── Tests ───────────────────────────────────────────────────────────────────

DETECTOR_URL ?= http://localhost:8002
VLLM_URL     ?= http://localhost:8000

url-detector:
	@echo "=== Test: servicio de detección de objetos ==="
	@echo "--- curl de ejemplo (README) ---"
	curl -F "file=@image.jpg" $(DETECTOR_URL)/detect

test-detector:
	@echo "=== Test: servicio de detección de objetos ==="
	@echo "--- curl de ejemplo (README) ---"
	docker compose run --rm app python /srv/app/tests/test_detector.py

test-rag:
	@echo "=== Test: flujo RAG (via LangGraph directo) ==="
	docker compose run --rm app python /srv/app/tests/test_rag.py

test-python:
	@echo "=== Test: nodo Python (ejecución de código) ==="
	docker compose run --rm app python /srv/app/tests/test_python_node.py

test-memory:
	@echo "=== Test: resumen automático de memoria ==="
	docker compose run --rm app python /srv/app/tests/test_memory.py

test-all: test-detector test-rag test-python test-memory
	@echo "=== Todos los tests completados ==="

eval:
	@echo "=== Evaluación: golden set (métricas + LLM-as-judge) ==="
	docker compose run --rm app python /srv/app/tests/eval.py $(ARGS)

# ─── Desarrollo local (sin Docker) ──────────────────────────────────────────
#
# Requiere: .venv con app/requirements.txt instalado
#   Windows:  .venv\Scripts\activate
#   Linux:    source .venv/bin/activate
#
# Servicios opcionales en localhost (las llamadas fallan graciosamente si no están):
#   - vLLM / Ollama en LOCAL_VLLM_URL  (para que el LLM responda)
#   - Detector YOLO en LOCAL_DETECTOR  (para el tool detect_objects)
#
# Sobreescribibles en línea:
#   make start-local LOCAL_VLLM_URL=http://localhost:11434/v1 LOCAL_VLLM_MODEL=gemma4:e2b

LOCAL_VLLM_URL    ?= http://localhost:8000/v1
LOCAL_VLLM_MODEL  ?= gemma
LOCAL_CHROMA_PORT ?= 8001
LOCAL_DETECTOR    ?= http://localhost:8002
CHROMA_PID_FILE   := .chroma_local.pid

# export hace que las vars sean visibles para los subprocesos de make (Python, chainlit…)
# sin necesidad de prefijarlas en cada línea.
export VLLM_BASE_URL  = $(LOCAL_VLLM_URL)
export VLLM_MODEL     = $(LOCAL_VLLM_MODEL)
export CHROMA_HOST    = localhost
export CHROMA_PORT    = $(LOCAL_CHROMA_PORT)
export DETECTOR_URL   = $(LOCAL_DETECTOR)
export DOCS_DIR       = $(CURDIR)/docs
export DATA_DIR       = $(CURDIR)/data
export PYTHONPATH     = $(CURDIR)

# ── start-local ──────────────────────────────────────────────────────────────
start-local:
	@echo ""
	@echo "  UI   http://localhost:8501"
	@echo "  LLM  $(LOCAL_VLLM_URL)  (peticiones fallarán si no está corriendo)"
	@echo ""
	.venv/Scripts/chainlit run app/chainlit_app.py --port 8501

# ── ingest-local ─────────────────────────────────────────────────────────────
ingest-local:
	@echo "Ingestando docs/ en Chroma local (puerto $(LOCAL_CHROMA_PORT))..."
	.venv/Scripts/python -m app.ingest $(ARGS)

ingest-local-reset:
	$(MAKE) ingest-local ARGS=--reset

# ── eval-local ───────────────────────────────────────────────────────────────
eval-local:
	@echo "Evaluando con el golden set (servicios en localhost)..."
	.venv/Scripts/python tests/eval.py $(ARGS)

# ── stop-local ───────────────────────────────────────────────────────────────
stop-local:
	@if [ -f $(CHROMA_PID_FILE) ]; then \
	    PID=$$(cat $(CHROMA_PID_FILE)); \
	    if command -v powershell.exe &>/dev/null; then \
	        powershell.exe -Command "Stop-Process -Id $$PID -Force -ErrorAction SilentlyContinue"; \
	    else \
	        kill $$PID 2>/dev/null; \
	    fi; \
	    rm -f $(CHROMA_PID_FILE); \
	    echo "Chroma (PID=$$PID) parado."; \
	else \
	    echo "No hay PID guardado — Chroma no fue lanzado con 'make chroma-local'."; \
	fi

# ─── Limpieza ────────────────────────────────────────────────────────────────

clean:
	docker compose down -v --remove-orphans
	@echo "Volúmenes eliminados. Los datos locales en data/ permanecen."
