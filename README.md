# Chatbot RAG multimodal + servicio de detección de objetos

Prueba técnica. Solución **dockerizada** con 4 contenedores:
Se ha decidido utilizar 4 contenedores por simular un entorno real con diferentes servicios.

| Servicio   | Rol                                                                 | Puerto host |
|------------|---------------------------------------------------------------------|-------------|
| `chroma`   | Base de datos vectorial (modo servidor)                             | 8001        |
| `vllm`     | vLLM sirviendo `gemma-4-E2B-it-qat-w4a16-ct` (multimodal + tools)   | 8000        |
| `detector` | FastAPI + YOLO, detecta `person` y `car` (Apartado 3)               | 8002        |
| `app`      | Chainlit (UI) + LangGraph (backend) + embeddings CPU + CLI ingesta  | 8501        |

## Requisitos

- Docker + Docker Compose.
- GPU NVIDIA con WSL2 + NVIDIA Container Toolkit (para `vllm`).
- Token de Hugging Face si el modelo es *gated*.

## Arranque

```bash
cp .env.example .env          # rellena HUGGING_FACE_HUB_TOKEN si hace falta
docker compose up -d --build  # la primera vez vLLM descarga el modelo (varios minutos)
docker compose ps             # espera a que vllm y chroma estén healthy
```

## Ingesta de documentos

1. Copia tus PDFs en `docs/` (incluye al menos uno con imágenes y uno tipo formulario).
2. Lanza la ingesta:

```bash
docker compose run --rm app python -m app.ingest          # añade a lo existente
docker compose run --rm app python -m app.ingest --reset  # vacía y reingesta
```

La ingesta:
- Indexa el **texto** (chunks) en Chroma.
- Extrae **imágenes**, las describe con Gemma (visión) e indexa esas descripciones
  (`data/images/`).
- Si la primera página es un **formulario**, hace **extracción estructurada** a
  `data/extractions/<pdf>.json` (no se guarda en la vector DB).

## Uso del chatbot

Abre <http://localhost:8501>. Puedes:

- **Preguntar por los documentos** (RAG con cita de fuente).
- **Preguntar por las imágenes** de los PDFs.
- **Subir una imagen** y preguntar "¿cuántos coches/personas hay?" → invoca el detector.
- **Pedir cálculos** ("calcula la media de…", "grafica…") → genera y ejecuta Python.
- La **memoria** se resume automáticamente al superar `SUMMARY_TOKEN_THRESHOLD` tokens.

## Servicio de detección (Apartado 3) por separado

El servicio es autónomo y se puede llamar desde Postman o Python.
Se ha decidido integrarlo en el agente por añadirlo como herramienta externa y validar asi también su funcionamiento.

```bash
# multipart (Postman: POST form-data, key "file" tipo File)
curl -F "file=@image.jpg" http://localhost:8002/detect
```

```python
import requests
r = requests.post("http://localhost:8002/detect",
                  files={"file": open("image.jpg", "rb")})
print(r.json())
# {"detections": [{"label": "car", "confidence": 0.94, "box": [...]}, ...],
#  "counts": {"car": 1, "person": 0}}
```

## Arquitectura (resumen)

- **Grafo LangGraph**: `orchestrator` (routing por salida estructurada) →
  `rag` | `python` | `chitchat` → `summarize` → END.
- **RAG**: Chroma con una colección que mezcla texto e imágenes (descritas por el LLM),
  diferenciados por metadata.
- **Memoria**: `SqliteSaver` persiste el estado por `thread_id`; un nodo propio resume
  al superar el umbral de tokens.
- **Embeddings**: `BAAI/bge-m3` en CPU (multilingüe).

## Limitaciones conocidas

- El nodo de Python ejecuta código generado por el LLM dentro del contenedor; en
  producción requeriría un sandbox real (gVisor, contenedor efímero, etc.).
- `yolov8n` corre en CPU por defecto para dejar la VRAM a vLLM (configurable a GPU).
