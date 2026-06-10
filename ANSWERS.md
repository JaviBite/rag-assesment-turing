# Respuestas

## Apartado 2: Preguntas teóricas

1. Diferencias entre 'completion' y 'chat' models.
> Aunque abmos se basan en el mismo modelo LLM basado en transformers para completar texto, la diferencia en el entrenamiento adicional de estos modelos, los completion simplemente intentan continuar el texto, como un autocompletar avanzado mientras que el "chat" o instructed como se suelen llamar estan enfocadoes en mantener una conversación. Estos ultimos son apliamente utilizados para chatbots, Q&A, Agentes, etc.

2. ¿Qué diferencias hay entre un modelo de razonamiento y un modelo generalista?
> Un modelo generalista genera la respuesta indemdiatamente tras el prompt del usuario mientras que los modelos de raonamiento generan unas trazas de razonamiento (texto) interno que les ayuda a planificar la respuesta (Chain-of-thought). Estas trazas ocurren hasta que el modelo decide dejar de razonar mediante un token especial y entonces devuelve la respuesta al usuario. Estos modelos son mas lentos pero oferecen respuestas más correctas y elaboradas.

3. ¿Cómo forzar a que el chatbot responda 'sí' o 'no'? ¿Cómo parsear la salida para que siga un formato determinado?
> La practica mas extendida es por ejemplo mediante LangChain que ya lo integra, aplicar métodos como "with_structured_output()" que ya internamente preparan la solicitud al servicio de inferencia vLLM para configuirarla aplicando restricciones a los logits (salida del modelo en forma de "puntuaciones" matématicas para cada token). Luego gracias a librerias parseo y de tipado de datos como pydantic podemos verificar que la salida es la deseada.

4. RAG vs fine-tuning: ¿para qué sirve cada uno, y qué ventajas e inconvenientes tienen?
> Un RAG como sus siglas indican es un sistema de "generación recuperación-aumentado" lo que quiere decir es que se fusiona una pieza de recuperación de información (texto generalmente) y un LLM para presentar y razonar sobre la información recuperada. El fine-tuning consiste en entrenar el propio modelo LLM con datos nuevos. Mientras que el RAG necesita de componentes externos (vectorDB, extractor de embeddings, orquestación), permite modificar su base de conocimiento facilmente indexando o eliminado ficheros de la DB, el finetuning genera un unico modelo LLM estático que una vez se aplica, modificar su conocimiento requeriría volver a entrenar, un proceso largo y muy costoso computacionalmente.

5. ¿Qué es un agente?
> Un agente es un wrapper sobre un LLM al cual se le indica cual es su rol en el sistema para que razone sobre las tareas que tiene que llevar a cabo. Pueden atuar individualmente (ej. Agente resumidor de texto) o en grupo mediante un sistema multiagente donde cada uno esta especializado en una tarea concreta.

6. ¿Cómo evaluar el desempeño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG? ¿Cómo evaluar el desempeño de una app de IA Generativa, en general: herramientas y métricas?
> *Bot de Q&A*: Existen diferentes dataset de conocimiento general para evaluar Q&A como TriviaAQ, en este caso estaríamos evaluando el modelo LLM que hay detrás. Tambien se puede aplicar la técnica LLM-as-judge que utiliza un modelo superior (más parámetros) para evaluar las pregunras y respuestas. Dependiendo del contexto del chatbot lo ideal sería tener un dataset propio con el que evaluar.

> *RAG*: Para evaluar un RAG se puede evaluar por una parte sus componentes como el recuperador de informacion (+ embeddings indirectamente) con métricas típicas como Recall@K/Precision@k para las cuales es necesario un dataset anotado, también hay metricas mas centradas en la parte de generacion como Fidelidad (que la respuesta se base en el contexto recuperado) o la relevancia de la respuesta. También indicar que se puede evaluar el performance en términos de velocidad de respuesta tanto del retrieval (tiempo de embeddings) como del LLM (Time to First token o TOkens per second). Existen frameworks como RAGAS para la evaluación completa de estos sistemas

> *GenAI app*: EN general toda aplicación de IA Generativa puede evaluar la pieza general que es el LLM tanto en performance (Time to first token, Tokens per second) como en calidad de respuesta con el uso de datasets o benchmarks específicos en funcion de las tareas de la aplicación: Q&A TriviaQA, Agentes (𝜏²-Bench), Lenguaje natural (HellaSwage), problemas matemáticos (GSM8K,MATH), Razonamiento científico (ARC,GPQA), COding (SciCoding, etc.)

## Apartado 3 (Opcional): Extra teórico - entrenamiento de un modelo con nuevas categorías
> Asumo que es entrenar un modelo de detección (el del apartado 3) de objetos con nuevas clases

1. Pasos necesarios a seguir.
> 1. Preparar datos de las nuevas clases a detectar: Examinar datos, limpieza, balananceo, preparar splits (train-test-val)
> 2. Elegir el modelo base de detección en función de los requiisitos de latencia y calidad (n,m,l,x, etc)
> 3. Finetuning (transfer-learning) del modelo base elegido y el dataset preparado. Se aconseja utilizar buenas prácticas de MLOps para mantener la trazabilidad de los experimentos y las métricas generadas durante el entrenamiento (accuracy, loss, validación). Realizar varios experimentos con diferentes configuraciones para econtrara la optima (grid-search con hyperparámetros, lr, data-augmentation, etc).
> 4. Elegir el modelo candidato y evaluarlo finalmente contra el split de test. EN caso de no obtener buenos resultados volver a pasos anteriores. Proceso iterativo hasta alcanzar el objetivo adecuado.

2. Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
> Datos insuficientes para las nuevas clases: Aplicar medidas de data augmentation, enriquecer los datos con datasets publicos compatibles o incluso probar generación sintética (experimental).
> El modelo resultante no cumple con las expectativas: Si las expectativas son relativas al tiempo de inferencia, elegir un modelo mas pequeño sacrificando cierta calidad o si por el contrario son de calidad elegir un modelo más pesado a pesar de aumentar los tiempos de inferencia.
> En produccion las imagenes de entrada son difieren de las del entrenamiento (DataDrift): Si no es posible reentrenar con los nuevos datos, bajar el umbral de detección y guardar un procentaje de detecciones para alimentar el dataset de entrenamiento (aprendizaje continuo)

3. Estimación de cantidad de datos necesarios así como de los resultados/métricas esperadas.
> Depende en función de las clases a detectar y como diferen del preentrenamiento original. Si son pocas clases y con relativa similitud a las originales (COCO dataset) con 200-500 ejemplos podría funcionar en cambio si son muy erspeficia y difieren del dominio original se requeriría un volumen mucho mayor x5-10

4. Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño: las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.
> Para mejorar el desemepño del modelo se pueden usar técnicas de data augmentation, selección correcta de hyperparámetros, etc.
> Las métricas relevantes durante el entrenamiento se evalua la calidad con métricas como el accuracy, precisión, recall, f1score, mAP@0.5, mAP@0.5-0.95, además de las funciones de pérdida (loss) durante el entrenamiento, en el caso de yolo: clsloss, boxloss, dflloss.
> Durante la inferencia las métricas de latencia y recursos son las que mas peso ganan, como el procentaje de GPU en uso, FPS (o tiempo de copmuto por frame)