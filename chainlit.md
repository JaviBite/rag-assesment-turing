# 🤖 Chatbot RAG multimodal + detección de objetos

Asistente de la prueba técnica. Combina varias capacidades sobre un grafo de
**LangGraph**, con **Gemma** servido vía **vLLM** y **Chroma** como base vectorial:

- **RAG multimodal** sobre los PDFs ingestados en `docs/` (texto + imágenes
  descritas por el modelo), con cita de la fuente.
- **Generación y ejecución de código Python** para cálculos, tablas y gráficas.
- **Detección de objetos (YOLO)**: cuenta o localiza personas y coches en
  imágenes que subas al chat.
- **Memoria de conversación**: se resume automáticamente al superar el umbral
  de tokens configurado.

## Ejemplos para probar el sistema 🧪

Prueba estas preguntas para ver las distintas capacidades del asistente:

### 🐍 Python
> Muéstrame los 100 primeros números de la secuencia Fibonacci

### 📄 RAG (documentos)
> ¿Cuánta población hay actualmente en diferentes países?

### 🖼️ RAG con imágenes
> Busca una imagen de un escudo de armas o de bandera

### 🔍 Detección de objetos
> Detecta las personas de esta imagen

*(Sube primero una imagen con personas o coches y, en el mismo mensaje o en uno
siguiente, pide la detección)*
