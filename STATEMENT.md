# Reto Turing Challenge — Departamento de Ciencia de Datos

Este reto está orientado a resolver, de una forma aproximada, un proyecto similar a algunos de los desarrollados en Turing Challenge desde el punto de vista del departamento de ciencia de datos.

> Con objeto de simplificar y reducir el tiempo de desarrollo no se tendrá en cuenta, aunque no se prohíbe, el uso de Azure. Eso sí, **no está permitido el uso de servicios cognitivos**.

El reto estará compuesto de 3 apartados: un apartado **técnico**, un apartado **teórico** y un apartado **opcional**.

> Se valorará positivamente enviar un enlace al repositorio Git con el apartado técnico desarrollado para que lo podamos revisar antes de la entrevista.

## Objetivos de cada apartado

- **Apartado técnico**: valorar la capacidad de buscar una solución a un problema en un tiempo limitado y que ésta dé la salida especificada. No se plantea evaluar que la solución sea óptima en ninguna métrica, es decir, ni tiempo de ejecución, ni precisión de la inferencia.
- **Apartado teórico**: realizar un pequeño guion de los pasos necesarios a realizar para resolver el reto, así como detectar y comentar los posibles problemas que podrían aparecer y plantear contramedidas para minimizar los riesgos.
- **Apartado opcional**: simplemente una oportunidad para demostrar el conocimiento en otras tecnologías relevantes.

---

## Apartado 1: Chatbot (Técnico)

Crear un chatbot que tenga las siguientes funcionalidades:

- [X] Una interfaz, por ejemplo la interfaz de chatbot de Gradio.
- [ ] Ingestar varios documentos PDF largos para usarlos como base de conocimiento de un RAG. Se ha de usar una base de datos vectorial a elección.
  - [X] *(Bonus)* Que alguno de los documentos contenga imágenes y éstas sean indexadas para poder preguntar por ellas.
  - [X] *(Bonus)* Que de alguno de los documentos se haga una extracción estructurada de su información (por ejemplo, un formulario con nombre, apellidos, fecha de nacimiento...). Esta extracción de información no tiene por qué guardarse en la base de datos vectorial.
- [ ] Implementar una memoria dinámica que mantenga la conversación y que cuando esta pase de X tokens se resuma de forma automática.
- [X] La implementación ha de estar basada en LangChain/LangGraph.
- [ ] Si se detecta una pregunta que lo necesite, el modelo ha de ser capaz de implementar y ejecutar código Python.

---

## Apartado 2: Preguntas teóricas

Dar respuesta a los siguientes puntos de forma teórica, sin necesidad de desarrollarlos, que guardan relación con las tecnologías utilizadas en el primer apartado:

1. Diferencias entre 'completion' y 'chat' models.
2. ¿Qué diferencias hay entre un modelo de razonamiento y un modelo generalista?
3. ¿Cómo forzar a que el chatbot responda 'sí' o 'no'? ¿Cómo parsear la salida para que siga un formato determinado?
4. RAG vs fine-tuning: ¿para qué sirve cada uno, y qué ventajas e inconvenientes tienen?
5. ¿Qué es un agente?
6. ¿Cómo evaluar el desempeño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG? ¿Cómo evaluar el desempeño de una app de IA Generativa, en general: herramientas y métricas?

---

## Apartado 3 (Opcional): Servicio local de detección de objetos

El objetivo es disponer de un servicio que tenga como entrada una imagen y que como salida proporcione un JSON con detecciones de coches y personas. Se han de cumplir los siguientes puntos:

- [X] No hay necesidad de entrenar un modelo. Se pueden usar preentrenados.
- [X] El servicio ha de estar conteinerizado. Es decir, una imagen Docker que al arrancar exponga el servicio.
- [ ] La petición al servicio se puede hacer desde Postman o herramienta similar o desde código Python.
- [X] La solución ha de estar implementada en Python.

### Extra teórico: entrenamiento de un modelo con nuevas categorías

Además, plantear cuáles serían los pasos necesarios para entrenar un modelo de detección con categorías no existentes en los modelos preentrenados. Los puntos en los que centrar la explicación son:

- Pasos necesarios a seguir.
- Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
- Estimación de cantidad de datos necesarios así como de los resultados/métricas esperadas.
- Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño: las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.