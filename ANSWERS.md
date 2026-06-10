# Respuestas

## Apartado 2: Preguntas teóricas

1. Diferencias entre 'completion' y 'chat' models.
> Aunque ambos se basan en el mismo modelo LLM basado en transformers para completar texto, la diferencia radica en el entrenamiento adicional (SFT, DPO, etc.) de estos modelos: los modelos "completion" simplemente intentan continuar el texto, como un autocompletar avanzado, mientras que los modelos "chat" o "instruct", como se suelen llamar, están enfocados en mantener una conversación con el uso de "chat templates" y roles mediante tokens especiales. Estos últimos son ampliamente utilizados para chatbots, Q&A, agentes, etc.

2. ¿Qué diferencias hay entre un modelo de razonamiento y un modelo generalista?
> Un modelo generalista (ej. GPT-3) genera la respuesta inmediatamente tras el prompt del usuario, mientras que los modelos de razonamiento (como GPT-o1, DeepSeek-R1, etc.) generan unas trazas de razonamiento (texto) internas que les ayudan a planificar la respuesta (Chain-of-thought). Estas trazas ocurren hasta que el modelo decide dejar de razonar mediante un token especial y entonces devuelve la respuesta al usuario. Estos modelos son más lentos y costosos (más tokens, más coste), pero ofrecen respuestas más correctas y elaboradas. Algunos modelos híbridos permiten elegir si usar razonamiento o no (on/off).

3. ¿Cómo forzar a que el chatbot responda 'sí' o 'no'? ¿Cómo parsear la salida para que siga un formato determinado?
> La práctica más extendida es, por ejemplo, mediante LangChain (que ya lo integra), aplicar métodos como "with_structured_output()" que internamente preparan la solicitud al servicio de inferencia vLLM para configurarla aplicando restricciones a los logits (logit bias, salida del modelo en forma de "puntuaciones" matemáticas para cada token) siempre y cuando el modelo lo permita. Luego, gracias a librerías de parseo y tipado de datos como pydantic, podemos verificar que la salida es la deseada.

4. RAG vs fine-tuning: ¿para qué sirve cada uno, y qué ventajas e inconvenientes tienen?
> Un RAG, como sus siglas indican, es un sistema de "generación aumentada por recuperación", lo que significa que se fusiona una pieza de recuperación de información (texto generalmente) y un LLM para presentar y razonar sobre la información recuperada. El fine-tuning consiste en entrenar el propio modelo LLM con datos nuevos. Mientras que el RAG necesita de componentes externos (vectorDB, extractor de embeddings, orquestación), permite modificar su base de conocimiento fácilmente indexando o eliminando ficheros de la DB; el finetuning, en cambio, genera un único modelo LLM estático cuyo conocimiento, una vez aplicado, solo se puede modificar volviendo a entrenarlo, un proceso largo y muy costoso computacionalmente. Ambas técnicas pueden ser complementarias para casos de uso muy específicos (ej. medicina).

5. ¿Qué es un agente?
> Un agente es un wrapper sobre un LLM al cual se le indica cuál es su rol en el sistema para que razone sobre las tareas que tiene que llevar a cabo. Se basa en el bucle Pensar -> Actuar -> Observar. Pueden actuar individualmente (ej. agente resumidor de texto) o en grupo mediante un sistema multiagente donde cada uno está especializado en una tarea concreta (workers-orquestador). Suelen implicar una arquitectura basada en estados con memoria y herramientas (tools, MCPs). Librerías como LangChain/LangGraph facilitan su desarrollo.

6. ¿Cómo evaluar el desempeño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG? ¿Cómo evaluar el desempeño de una app de IA Generativa, en general: herramientas y métricas?
> *Bot de Q&A*: Existen diferentes datasets de conocimiento general para evaluar Q&A como TriviaQA; en este caso estaríamos evaluando el modelo LLM que hay detrás. También se puede aplicar la técnica LLM-as-judge, que utiliza un modelo superior (más parámetros) para evaluar las preguntas y respuestas. Dependiendo del contexto del chatbot, lo ideal sería tener un dataset propio con el que evaluar.

> *RAG*: Para evaluar un RAG se pueden analizar, por una parte, sus componentes, como el recuperador de información (+ embeddings indirectamente), con métricas típicas como Recall@K/Precision@K, para las cuales es necesario un dataset anotado; también hay métricas más centradas en la parte de generación, como la fidelidad (que la respuesta se base en el contexto recuperado) o la relevancia de la respuesta, muy relacionadas con el framework RAGAS. También cabe indicar que se puede evaluar el performance en términos de velocidad de respuesta, tanto del retrieval (tiempo de embeddings) como del LLM (Time to First Token o Tokens per second).

> *GenAI app*: En general, toda aplicación de IA Generativa puede evaluar la pieza general que es el LLM, tanto en performance (Time to First Token, Tokens per second) como en calidad de respuesta con el uso de datasets o benchmarks específicos en función de las tareas de la aplicación: Q&A (TriviaQA), agentes (𝜏²-Bench), lenguaje natural (HellaSwag), problemas matemáticos (GSM8K, MATH), razonamiento científico (ARC, GPQA), coding (SciCode, etc.). Existen frameworks como LangFuse o LangSmith que permiten depurar y monitorizar el flujo de las aplicaciones agénticas. En producción se puede usar telemetría para monitorizar el funcionamiento de los sistemas y aplicar procesos offline para extracción de métricas más exhaustivas. Se suelen incluir opciones de feedback de usuario para mejorar estas aplicaciones continuamente.

## Apartado 3 (Opcional): Extra teórico - entrenamiento de un modelo con nuevas categorías
> Asumo que se trata de entrenar el modelo de detección de objetos del apartado 3 con nuevas clases.

1. Pasos necesarios a seguir.
> 1. Preparar datos de las nuevas clases a detectar: Examinar datos, limpieza, balanceo, preparar splits (train-test-val).
> 2. Elegir el modelo base de detección en función de los requisitos de latencia y calidad (n, m, l, x, etc.).
> 3. Finetuning (transfer-learning) del modelo base elegido y el dataset preparado. Se aconseja utilizar buenas prácticas de MLOps (MLflow, W&B) para mantener la trazabilidad de los experimentos y las métricas generadas durante el entrenamiento (accuracy, loss, validación). Realizar varios experimentos con diferentes configuraciones para encontrar la óptima (grid-search con hyperparámetros, lr, data-augmentation, etc.).
> 4. Elegir el modelo candidato y evaluarlo finalmente contra el split de test. En caso de no obtener buenos resultados, volver a pasos anteriores. Proceso iterativo hasta alcanzar el objetivo adecuado.

2. Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
> Datos insuficientes o desbalanceados para las nuevas clases: Aplicar medidas de data augmentation, enriquecer los datos con datasets públicos compatibles o incluso probar generación sintética (experimental).
> El modelo resultante no cumple con las expectativas: Si las expectativas son relativas al tiempo de inferencia, elegir un modelo más pequeño sacrificando cierta calidad; o, si por el contrario son de calidad, elegir un modelo más pesado a pesar de aumentar los tiempos de inferencia.
> En producción las imágenes de entrada difieren de las del entrenamiento (Data Drift): Si no es posible reentrenar con los nuevos datos, bajar el umbral de detección y guardar un porcentaje de detecciones para alimentar el dataset de entrenamiento (aprendizaje continuo).

3. Estimación de cantidad de datos necesarios así como de los resultados/métricas esperadas.
> Depende en función de las clases a detectar y de cómo difieren del preentrenamiento original. Si son pocas clases y con relativa similitud a las originales (COCO dataset), con 200-500 ejemplos podría funcionar; en cambio, si son muy específicas y difieren del dominio original, se requeriría un volumen mucho mayor (x5-10).

4. Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño: las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.
> Para mejorar el desempeño del modelo se pueden usar técnicas de data augmentation, selección correcta de hyperparámetros, etc.
> Durante el entrenamiento se evalúa la calidad con métricas como el accuracy, precisión, recall, f1-score, mAP@0.5 y mAP@0.5-0.95, además de las funciones de pérdida (loss); en el caso de YOLO: clsloss, boxloss y dflloss.
> Para mejorar en tiempo de inferencia se pueden usar técnicas de cuantización o compresión (pruning, distillation) o exportar a frameworks de inferencia avanzados como TensorRT.
> Durante la inferencia, las métricas de latencia y recursos son las que más peso ganan, como el porcentaje de GPU en uso, FPS (o tiempo de cómputo por frame).