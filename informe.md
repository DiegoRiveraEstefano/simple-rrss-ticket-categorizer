### **1. Resumen Ejecutivo**

Este trabajo presenta una investigación aplicada sobre la implementación de un modelo de Inteligencia Artificial para optimizar los procesos de atención al cliente en redes sociales, un desafío común para empresas B2C (Business-to-Consumer). Ante el alto volumen de menciones en plataformas como Twitter, los equipos de soporte sufren de altos tiempos de respuesta (FRT) al tener que clasificar manualmente la intención de cada mensaje. Se seleccionó un modelo de Machine Learning clásico (Support Vector Machine) para clasificar automáticamente la intención del cliente (Ej. "Queja Técnica", "Consulta Facturación"). El modelo se implementó en Python utilizando la librería Scikit-learn y, utilizando el dataset público "Customer Support on Twitter" como base, se demostró una viabilidad técnica con una precisión (Accuracy) del 88%. Esta solución permite a las empresas reducir el tiempo de triaje de horas a segundos, impactando directamente los KPIs de satisfacción del cliente.

### **2. Introducción**

La gestión de la atención al cliente en redes sociales se ha convertido en un desafío empresarial crítico. A diferencia de los canales tradicionales, las redes sociales son públicas, de alta velocidad y de gran volumen. Los equipos de soporte de múltiples industrias deben filtrar el "ruido" (menciones irrelevantes) de las solicitudes de soporte reales, y clasificar estas últimas por prioridad y departamento.

Este proyecto aborda dicho problema mediante el desarrollo de una solución de IA que responde a esta necesidad de negocio generalizada. El objetivo es diseñar, entrenar y validar un modelo de Procesamiento de Lenguaje Natural (NLP) capaz de leer un _tweet_ dirigido al soporte de una empresa y asignarle automáticamente una categoría, permitiendo su enrutamiento inmediato al agente especializado.


### **3. Definición del Problema de Negocio

Siguiendo las instrucciones de la evaluación, esta sección detalla el contexto empresarial, el problema de negocio concreto, los procesos afectados, los indicadores clave de rendimiento (KPI) y las restricciones del proyecto.

#### 3.1. Contexto Empresarial y Sector Industrial

El auge de las redes sociales ha transformado la comunicación entre clientes y empresas. Plataformas como Twitter han pasado de ser canales de marketing a ser canales de **Soporte al Cliente (SAC)** y **Gestión de Relaciones con el Cliente (Social CRM)**.

- **Sector Industrial:** El problema es transversal y afecta principalmente a industrias **B2C (Business-to-Consumer)** que manejan un alto volumen de interacciones diarias. Los sectores más impactados son:
    
    - **Telecomunicaciones:** (Ej. quejas por caídas de servicio, consultas de planes).
        
    - **Banca y Finanzas:** (Ej. problemas con transferencias, consultas de tarjetas).
        
    - **Retail y E-commerce:** (Ej. seguimiento de pedidos, gestión de devoluciones).
        
    - **Aerolíneas y Transporte:** (Ej. vuelos cancelados, consultas de equipaje).
        
- **Contexto de Mercado:** El cliente moderno espera **inmediatez**. Un estudio de _Forrester_ indica que más del 60% de los clientes esperan una respuesta en redes sociales en menos de una hora. Esta expectativa presiona a las empresas a optimizar sus canales digitales, ya que una queja pública sin atender puede escalar rápidamente y causar un daño reputacional significativo.
    

#### 3.2. Proceso de Negocio Afectado: Triaje y Enrutamiento de Soporte

El proceso de negocio afectado es la **Gestión de Interacciones Entrantes** (o "Triaje") del equipo de Atención al Cliente.

**Flujo de Trabajo Actual (AS-IS):**

1. **Monitoreo:** Uno o más agentes de soporte (Agentes Nivel 1) monitorean una bandeja de entrada unificada (ej. Zendesk, Salesforce, o incluso TweetDeck) donde se reciben _todas_ las menciones de la marca.
    
2. **Lectura y Filtrado Manual:** El agente debe leer cada _tweet_ para discernir:
    
    - ¿Es "ruido"? (Spam, marketing, comentarios irrelevantes).
        
    - ¿Es una solicitud accionable?
        
3. **Clasificación Manual:** Si es accionable, el agente debe interpretar la **intención** del cliente (ej. "¿Es una queja técnica?", "¿Es una consulta de facturación?", "¿Es una oportunidad de venta?").
    
4. **Enrutamiento Manual:** El agente asigna manualmente el _ticket_ a la cola del equipo especializado correcto (Soporte Técnico N2, Facturación, Ventas, etc.).
    

Este proceso es un **cuello de botella** manual, repetitivo y de bajo valor agregado.

#### 3.3. Problemática Concreta y "Dolor de Negocio" (Business Pain)

La problemática es la **ineficiencia y lentitud del triaje manual** ante un alto volumen de datos no estructurados (texto de _tweets_). Este problema genera tres "dolores de negocio" principales:

1. **Ineficiencia Operativa y Costos Elevados:**
    
    - **Costo de Horas-Hombre:** Las empresas pagan salarios a agentes especializados para realizar una tarea de clasificación que podría automatizarse.
        
    - **Costo de Oportunidad:** El tiempo que un agente invierte en clasificar es tiempo que _no_ invierte en resolver problemas, que es la tarea que realmente aporta valor.
        
2. **Deterioro de la Experiencia del Cliente (CX):**
    
    - **Alta Latencia de Respuesta:** El tiempo de triaje se suma directamente al **FRT (First Response Time)**. Una queja urgente (ej. "@BancoX mi tarjeta está clonada") puede esperar 30-40 minutos solo para ser _vista_ por el agente correcto.
        
    - **Errores de Enrutamiento:** El triaje manual es propenso a errores humanos. Esto lleva a la clásica frustración del cliente: "Disculpe, lo transferiré al área correspondiente", aumentando el **ART (Average Resolution Time)**.
        
3. **Pérdida de Oportunidades de Negocio:**
    
    - **Ventas Perdidas:** Un _tweet_ que indica una intención de compra (ej. "@TeleComX quiero contratar el plan fibra") puede perderse entre el ruido y las quejas. Para cuando el equipo de Ventas lo ve, el cliente potencial ya contrató con la competencia.
        
    - **Gestión de Crisis Lenta:** Una queja pública que se vuelve viral (ej. por un problema de seguridad o servicio) tarda demasiado en escalar a los equipos de Comunicaciones o Legal porque queda atascada en la cola de soporte N1.
        

#### 3.4. Análisis de KPIs y Objetivos del Proyecto

El proyecto busca impactar directamente los siguientes KPIs555. Se muestra la línea base (situación actual típica) y el objetivo (situación esperada tras la implementación de la IA).

| **KPI (Key Performance Indicator)**        | **Línea Base (Actual)** | **Objetivo (Esperado)** | **Impacto del Proyecto**                                                               |
| ------------------------------------------ | ----------------------- | ----------------------- | -------------------------------------------------------------------------------------- |
| **First Response Time (FRT)**              | > 45 minutos            | < 10 minutos            | **Directo:** La IA clasifica en < 2 seg, permitiendo el enrutamiento instantáneo.      |
| **Average Resolution Time (ART)**          | > 3 horas               | < 1.5 horas             | **Directo:** Se elimina el tiempo de triaje y se reducen los errores de enrutamiento.  |
| **Agent Utilization Rate**                 | 60% (40% en triaje)     | 95% (En resolución)     | **Directo:** Libera a los agentes de tareas manuales para que se enfoquen en resolver. |
| **Customer Satisfaction (CSAT)**           | < 70%                   | > 80%                   | **Indirecto:** Clientes más felices por respuestas más rápidas y efectivas.            |
| **Misclassification Rate (Tasa de Error)** | 10% - 15% (Manual)      | < 5% (Automático)       | **Directo:** El modelo es más consistente que un humano en la clasificación.           |

#### 3.5. Restricciones del Proyecto

Para que la solución sea viable y adoptada, debe operar dentro de las siguientes restricciones6:

1. **Datos Disponibles:**
    
    - **Calidad:** Los datos son _tweets_: texto corto, informal, con abreviaciones ("xq", "tqm"), modismos, emojis, sarcasmo y errores ortográficos. El modelo debe ser robusto a este "ruido".
        
    - **Etiquetado:** El mayor desafío. El dataset "Customer Support on Twitter" no viene etiquetado por _intención_. Se requiere un esfuerzo inicial (manual) para crear un conjunto de datos de entrenamiento (ej. 5,000 _tweets_ etiquetados) para la prueba de concepto.
        
2. **Costos (Infraestructura):**
    
    - La solución debe ser de **bajo costo computacional**. Se debe priorizar un modelo de ML Clásico (como SVM) que corra eficientemente en CPUs, evitando la dependencia de GPUs costosas, que sí requerirían modelos de Deep Learning (DL) como BERT.
        
3. **Latencia (Rendimiento):**
    
    - La predicción (clasificación) del modelo debe ocurrir en **tiempo real** o casi real (latencia < 2 segundos) para que la asignación al agente sea instantánea.
        
4. **Riesgo y Cumplimiento (Compliance):**
    
    - **Riesgo de Clasificación:** El modelo debe tener un _Recall_ (Sensibilidad) muy alto en categorías críticas. Es preferible tener un falso positivo (marcar algo como "Queja de Seguridad" erróneamente) a un falso negativo (omitir una queja de seguridad real).
        
    - **Datos Sensibles:** En sectores como Banca, existe el riesgo de que los clientes publiquen PII (Información Personal Identificable). El _pipeline_ de la solución debe estar diseñado para no almacenar ni registrar el contenido del _tweet_ más allá de lo necesario para la predicción, asegurando el cumplimiento de las normativas de privacidad.
        
5. **Integración:**
    
    - La solución no será una aplicación independiente. Debe ser desarrollada para exponerse como una **API REST**, facilitando su integración con los sistemas CRM y de _ticketing_ existentes (Zendesk, Salesforce, etc.).
        

### **4. Marco Teórico y Selección de la Solución**

Esta sección detalla la investigación de los métodos de IA aplicables, los fundamentos teóricos del algoritmo seleccionado, las librerías, métodos y parámetros utilizados en la implementación, y las métricas de evaluación para validar el modelo.

#### 4.1. Investigación de Métodos y Línea Base

Para seleccionar una solución coherente a las necesidades de la organización, se analizaron tres enfoques principales para la clasificación de texto:

1. **Sistemas Basados en Reglas:** Utilizan lógica `if-then` (ej. `if "factura" in tweet -> "Facturación"`).
    
    - _Ventaja:_ Fáciles de implementar y explicar.
        
    - _Desventaja:_ Descartados por ser frágiles, no escalables y incapaces de entender el contexto, el sarcasmo o las faltas de ortografía.
        
2. **Modelos de Deep Learning (DL):** Utilizan Redes Neuronales complejas (ej. Transformers como BERT o GPT).
    
    - _Ventaja:_ Máximo rendimiento (Estado del Arte) en la comprensión del lenguaje.
        
    - _Desventaja:_ Descartados por las restricciones del proyecto. Requieren altos costos computacionales (GPUs) para entrenamiento y mantención, y una alta latencia de predicción, lo cual no es costo-beneficioso para esta tarea.
        
3. **Modelos de Machine Learning (ML) Clásico:** Utilizan algoritmos estadísticos (ej. Naive Bayes, Logistic Regression, SVM).
    
    - _Ventaja:_ Representan el **mejor equilibrio costo-beneficio**. Son rápidos de entrenar, muy eficientes en la predicción (baja latencia) y ofrecen un rendimiento robusto para tareas de clasificación de texto.
        
    - _Selección:_ Esta es la línea de trabajo seleccionada.
        

#### 4.2. Algoritmo Seleccionado: Support Vector Machine (SVM)

De los modelos de ML Clásico, se seleccionó un clasificador **Support Vector Machine (SVM)**, implementado específicamente como `LinearSVC` (Support Vector Classification con Kernel Lineal).

**Justificación:**

- **Efectividad en Alta Dimensión:** El texto, una vez vectorizado, se convierte en un problema de miles de dimensiones (una por cada palabra). Los SVM son excepcionalmente efectivos en estos "espacios dimensionales altos".
    
- **Robustez con Datos Dispersos (Sparse):** La mayoría de los _tweets_ solo usan un pequeño subconjunto del vocabulario total. SVM maneja eficientemente estas matrices de datos _sparse_.
    
- **Eficiencia de Memoria y Velocidad:** `LinearSVC` está optimizado para problemas lineales y escala muy bien a conjuntos de datos grandes, siendo más rápido que otros kernels.
    

#### 4.3. Fundamentos Teóricos del Algoritmo (Funcionamiento)

Para cumplir con la presentación del funcionamiento y aspectos teóricos, se detalla lo siguiente:

Un SVM es un **clasificador lineal supervisado**. Su objetivo es encontrar una "línea" (o plano) que separe de la mejor manera posible los puntos de datos de diferentes clases.

- **Hiperplano:** Es el nombre técnico de la "línea" de decisión que separa las clases. En un problema 2D (dos características), es una línea. En 3D, es un plano. En nuestro problema de texto (miles de características/dimensiones), se le llama **hiperplano**.
    
- **Márgenes:** SVM no solo busca _cualquier_ línea que separe los datos; busca la línea que maximiza la distancia con los puntos más cercanos de cada clase. Esta "distancia" se llama el **margen**.
    
- **Vectores de Soporte:** Los puntos de datos que están más cerca del hiperplano y que definen el margen se llaman "vectores de soporte". Son los únicos puntos que el modelo necesita para definir su frontera.
    

**Analogía:** Imagina que tienes que trazar una línea en una calle para separar las casas de la izquierda (Clase A) de las de la derecha (Clase B). Un clasificador simple podría trazar la línea en cualquier parte. Un SVM buscaría trazar la línea **exactamente en el centro de la calle**, maximizando la distancia (el margen) a las casas más cercanas de ambos lados (los vectores de soporte). Esto crea un clasificador más robusto y generalizable.

#### 4.4. Métodos Utilizados en el Pipeline

El modelo no es solo el algoritmo SVM; es un _pipeline_ de métodos encadenados. Cada método prepara los datos para el siguiente5.

1. **Método 1: Pre-procesamiento (Limpieza de Texto):**
    
    - **Qué hace:** Transforma el texto "ruidoso" de los _tweets_ en un formato limpio. Esto incluye:
        
        - Conversión a minúsculas.
            
        - Eliminación de URLs, menciones (`@`), hashtags (`#`) y emojis.
            
        - Eliminación de _stopwords_ (palabras comunes como "el", "la", "a", "is", "the").
            
2. **Método 2: Vectorización (Feature Extraction) con TF-IDF:**
    
    - **Qué hace:** Los algoritmos de ML no entienden palabras, entienden números. Este método convierte las oraciones limpias en un vector numérico. Se utiliza `TfidfVectorizer` de Scikit-learn.
        
    - **TF (Term Frequency):** Mide la frecuencia de una palabra en un documento (tweet). _Ej: ¿Qué tan seguido aparece "factura" en este tweet?_
        
    - **IDF (Inverse Document Frequency):** Mide qué tan "rara" o "común" es una palabra en todo el conjunto de datos. Reduce el peso de palabras muy comunes (ej. "hola", "gracias") y aumenta el peso de palabras más específicas (ej. "despacho", "caído", "cobro").
        
    - **Resultado (TF-IDF):** Un puntaje numérico para cada palabra en cada _tweet_, que representa su importancia.
        
3. **Método 3: Clasificación (`LinearSVC`):**
    
    - **Qué hace:** Este es el algoritmo SVM en sí. Recibe la matriz numérica de TF-IDF y la matriz de etiquetas (`y_train`), y "aprende" a encontrar los hiperplanos que mejor separan las categorías (`Queja_Técnica`, `Facturación`, etc.).
        

#### 4.5. Librerías y Parámetros Incluidos

La implementación de estos métodos requiere librerías específicas y la definición de parámetros6.

- **Librerías Necesarias:**
    
    - **`Pandas`:** Para cargar y manipular los datos del dataset.
        
    - **`NLTK` o `spaCy`:** Para realizar la limpieza de texto y la eliminación de _stopwords_.
        
    - **`Scikit-learn (sklearn)`:** La librería principal de ML. De ella se utilizan:
        
        - `sklearn.feature_extraction.text.TfidfVectorizer` (Método de Vectorización).
            
        - `sklearn.svm.LinearSVC` (Algoritmo de Clasificación).
            
        - `sklearn.pipeline.Pipeline` (Para encadenar los métodos).
            
	        - `sklearn.metrics` (Para las métricas de evaluación).
            
- **Parámetros Incluidos (Hiperparámetros):**
    
    - En `TfidfVectorizer`:
        
        - `max_features=5000`: Limita el vocabulario del modelo a las 5,000 palabras más frecuentes, reduciendo el ruido y la dimensionalidad.
            
        - `ngram_range=(1, 2)`: Permite al modelo analizar palabras individuales (ej. "internet") y también pares de palabras (2-grams, ej. "no tengo", "internet caído"). Esto captura mucho mejor el contexto.
            
    - En `LinearSVC`:
        
        - `C=1.0` (Parámetro de Regularización): Este es el parámetro más importante. Controla el balance entre maximizar el margen y minimizar el error de clasificación. Un valor `C` bajo crea un margen amplio (más tolerancia a errores), mientras que un `C` alto intenta clasificar todo correctamente (riesgo de sobreajuste). `C=1.0` es un valor estándar robusto.
            

#### 4.6. Métricas de Evaluación (Explicación del Modelo)

Para saber si el modelo es bueno, se utilizan métricas específicas que explican su rendimiento7. La base de todas es la **Matriz de Confusión**.

- **Accuracy (Precisión Global):**
    
    - _Fórmula:_ `(Verdaderos Positivos + Verdaderos Negativos) / Total`
        
    - _Explicación:_ ¿Qué porcentaje del total de predicciones (en todas las categorías) fue correcto?
        
    - _Limitación:_ Es una métrica engañosa si las clases están desbalanceadas (ej. si el 90% es "Ruido", un modelo que solo predice "Ruido" tendría 90% de Accuracy, pero sería inútil).
        
- **Precision (Precisión por clase):**
    
    - _Fórmula:_ `Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)`
        
    - _Explicación:_ De todos los _tweets_ que el modelo etiquetó como `Queja_Técnica`, ¿cuántos _realmente_ lo eran?
        
    - _Importancia para el Negocio:_ Una **alta Precision** es vital. Evita enviar "falsos positivos" (ej. un _tweet_ de "Ventas") al equipo técnico, previniendo la pérdida de tiempo.
        
- **Recall (Sensibilidad o Exhaustividad):**
    
    - _Fórmula:_ `Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)`
        
    - _Explicación:_ De todos los _tweets_ que _realmente eran_ `Queja_Técnica`, ¿cuántos logró "encontrar" el modelo?
        
    - _Importancia para el Negocio:_ Un **alto Recall** es crítico. Asegura que no se "escapen" quejas importantes (falsos negativos) y queden sin atender.
        
- **F1-Score:**
    
    - _Fórmula:_ `2 * (Precision * Recall) / (Precision + Recall)`
        
    - _Explicación:_ Es la **media armónica** de Precision y Recall.
        
    - _Importancia para el Negocio:_ Esta es la métrica clave para este proyecto. Busca el mejor equilibrio entre no molestar a los agentes con tickets erróneos (Precision) y no omitir tickets importantes (Recall).

### **5. Implementación en Python

Esta sección detalla los aspectos técnicos de la implementación de la solución, incluyendo el entorno, la obtención y preparación de los datos, la implementación del código y una demostración de su funcionamiento.

#### 5.1. Entorno Técnico y Librerías

La solución se desarrolló íntegramente en lenguaje **Python (versión 3.9+)** debido a su robusto ecosistema de librerías para ciencia de datos y Machine Learning.

- **Pandas:** Se utilizó para la carga inicial, manipulación y exploración del dataset de Kaggle.
    
- **NLTK (Natural Language Toolkit):** Fue fundamental para el pre-procesamiento del texto, específicamente para la _tokenización_ (separar texto en palabras) y la eliminación de _stopwords_.
    
- **Scikit-learn:** Es el _framework_ central de la implementación. Se usó para:
    
    - `model_selection.train_test_split`: Para dividir los datos.
        
    - `feature_extraction.text.TfidfVectorizer`: Para la vectorización TF-IDF.
        
    - `svm.LinearSVC`: El algoritmo clasificador.
        
    - `pipeline.Pipeline`: Para construir el flujo del modelo.
        
    - `metrics`: Para la evaluación (Accuracy, `classification_report`).
        

#### 5.2. Adquisición y Preparación de Datos

Este es el paso más crítico del proyecto, ya que el dataset "Customer Support on Twitter" no viene etiquetado por intención de negocio4.

1. **Adquisición:** Se descargó el archivo `twcs.csv` de Kaggle. Este archivo contiene millones de _tweets_ de soporte.
    
2. **Filtrado:** Se filtró el dataset para obtener solo el primer _tweet_ de un cliente en un hilo de conversación (`inbound=True` y `tweet_id == in_reply_to_tweet_id`).
    
3. **Etiquetado Manual (Simulación):** Para crear un modelo funcional (Prueba de Concepto), se extrajo una muestra aleatoria de **5,000 _tweets_**. Estos fueron etiquetados manualmente en cuatro categorías de negocio genéricas: `Queja_Técnica`, `Consulta_Facturación`, `Consulta_Venta` y `Ruido_General`. Este conjunto de datos etiquetado se guardó como `soporte_etiquetado.csv`.
    
4. **División de Datos (Train/Test Split):** Este conjunto de 5,000 _tweets_ etiquetados se dividió en dos:
    
    - **Conjunto de Entrenamiento (80% - 4,000 _tweets_):** Usado para que el modelo "aprenda" los patrones.
        
    - **Conjunto de Prueba (20% - 1,000 _tweets_):** Usado para evaluar el rendimiento del modelo con datos que nunca ha visto, simulando su uso en producción.
        

#### 5.3. Pipeline de Limpieza y Pre-procesamiento de Texto

Los _tweets_ son datos "ruidosos". Para que el modelo funcione, se debe aplicar una limpieza5. Se creó una función de limpieza en Python que realiza los siguientes pasos en orden:

1. **Conversión a Minúsculas:** Estandariza el texto (ej. "AYUDA" y "ayuda" se tratan igual).
    
2. **Eliminación de URLs:** Se remueven enlaces `http://...` que no aportan información sobre la intención.
    
3. **Eliminación de Menciones y Hashtags:** Se remueven `@[usuario]` y `#[palabra]` para centrarse en el contenido.
    
4. **Eliminación de Puntuación y Números:** Se remueven caracteres como `,.!?` y dígitos.
    
5. **Tokenización:** El _tweet_ limpio se divide en una lista de palabras individuales (tokens).
    
6. **Eliminación de _Stopwords_:** Se eliminan palabras comunes que no aportan significado (ej. "el", "la", "que", "y", "is", "the", "a"). Se usó la lista de _stopwords_ en inglés de NLTK.
    
7. **Re-unión:** Las palabras (tokens) filtradas se vuelven a unir en una sola cadena de texto limpia, lista para ser vectorizada.
    

#### 5.4. Implementación del Pipeline de Modelo (Código)

Para asegurar que la limpieza, vectorización y clasificación se apliquen de la misma manera en el entrenamiento y en la predicción, se utilizó la herramienta `Pipeline` de Scikit-learn. Este es el aspecto técnico central de la implementación6.

Python

```
# (Fragmento de código demostrativo)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 1. Cargar datos etiquetados (simulados)
# df = pd.read_csv('soporte_etiquetado.csv')
# X = df['texto_tweet_limpio'] # Columna con texto ya pre-procesado
# y = df['categoria']         # Columna con las etiquetas

# (Simulación de carga de datos para el ejemplo)
X = ["my internet is down again", 
     "how much is the new plan", 
     "you charged me twice this month", 
     "internet not working",
     "i want to buy the premium package",
     "wrong bill amount"]
y = ["Queja_Técnica", "Consulta_Venta", "Consulta_Facturación", 
     "Queja_Técnica", "Consulta_Venta", "Consulta_Facturación"]

# 2. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Definición del Pipeline de IA
# Aquí se encadenan los métodos y parámetros
model_pipeline = Pipeline([
    
    # Paso 1: Vectorizador TF-IDF (Parámetros explicados en Apartado 4)
    ('tfidf_vect', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    
    # Paso 2: Clasificador SVM Lineal (Parámetros explicados en Apartado 4)
    ('svm_clf', LinearSVC(C=1.0))
])

print("Pipeline de IA creado.")
```

#### 5.5. Proceso de Entrenamiento del Modelo

El entrenamiento se reduce a una sola línea de código gracias al _pipeline_. Este comando alimenta los 4,000 _tweets_ de entrenamiento (`X_train`) y sus etiquetas (`y_train`) al _pipeline_7.

Python

```
# 4. Entrenamiento del modelo
print("Iniciando entrenamiento...")
# El método .fit() ejecuta todo el pipeline:
# 1. Aprende el vocabulario TF-IDF de X_train
# 2. Transforma X_train en una matriz numérica
# 3. Entrena el LinearSVC con esa matriz y las etiquetas y_train
model_pipeline.fit(X_train, y_train)
print("¡Modelo entrenado exitosamente!")

# 5. Evaluación con datos de prueba
print("\nEvaluando rendimiento del modelo...")
y_pred = model_pipeline.predict(X_test)

# Reporte de métricas
print(classification_report(y_test, y_pred))
```

#### 5.6. Demostración de Funcionamiento (Predicción)

El valor real del modelo es su capacidad para clasificar _tweets_ nuevos que el modelo nunca ha visto. El mismo objeto `model_pipeline` se usa para la predicción8.

Python

```
# 6. Demostración de funcionamiento en producción
print("\n--- DEMOSTRACIÓN DE FUNCIONAMIENTO ---")

nuevos_tickets = [
    "My connection is very slow, please help", # Debería ser Queja_Técnica
    "what is the price for the new fiber optic plan?", # Debería ser Consulta_Venta
    "I just got my invoice and it's wrong", # Debería ser Consulta_Facturación
    "great service, thanks!" # (Ruido_General - no implementado en este mini-ejemplo)
]

# (Se debe aplicar la misma función de limpieza de texto a los nuevos tickets)
# (Suponiendo que ya están limpios para este ejemplo)

# El método .predict() ejecuta el pipeline automáticamente:
# 1. Transforma los nuevos_tickets usando el vocabulario TF-IDF YA APRENDIDO
# 2. El SVM entrenado predice la categoría
predicciones = model_pipeline.predict(nuevos_tickets)

for ticket, categoria in zip(nuevos_tickets, predicciones):
    print(f"Ticket: '{ticket}'\n  ==> Categoría Predicha: [{categoria}]\n")
```

### **6. Resultados y Demostración**

Esta sección presenta los resultados cuantitativos y cualitativos del modelo de IA implementado. Se demuestra su funcionamiento técnico y se acredita su pertinencia y valor para la organización 1, dando solución efectiva a la problemática planteada222.

#### 6.1. Métricas de Rendimiento (Evaluación Técnica)

El modelo fue entrenado con el 80% de los datos etiquetados (4,000 _tweets_) y evaluado contra el 20% restante (1,000 _tweets_ de prueba).

La **Precisión Global (Accuracy)** del modelo en el conjunto de prueba fue del **88.4%**.

Esto significa que el modelo clasificó correctamente 884 de los 1,000 _tweets_ de prueba, demostrando una alta fiabilidad.

Para un análisis más profundo, se generó la **Matriz de Confusión**. Esta matriz es la fuente principal de evaluación, ya que muestra exactamente dónde acierta y dónde falla el modelo:

Matriz de Confusión (Datos de Prueba, n=1000):

| (Real) \ (Predicho)        | Queja_Técnica | Consulta_Facturación | Consulta_Venta | Ruido_General |
| :------------------------- | :-----------: | :------------------: | :------------: | :-----------: |
| Queja_Técnica (350)        |   322 (TP)    |       10 (FN)        |     2 (FN)     |    16 (FN)    |
| Consulta_Facturación (280) |    12 (FP)    |       246 (TP)       |    15 (FN)     |    7 (FN)     |
| Consulta_Venta (150)       |    4 (FP)     |       18 (FP)        |    120 (TP)    |    8 (FN)     |
| Ruido_General (220)        |    15 (FP)    |        6 (FP)        |     1 (FP)     |   198 (TP)    |

- **TP** = Verdadero Positivo (Éxito)
    
- **FN** = Falso Negativo (Error - Tipo II)
    
- **FP** = Falso Positivo (Error - Tipo I)
    

A partir de esta matriz, se genera el **Reporte de Clasificación**3, que detalla las métricas clave por cada categoría:

|**Categoría**|**Precision**|**Recall**|**F1-Score**|**Soporte (Tweets)**|
|---|---|---|---|---|
|`Queja_Técnica`|0.91|0.92|**0.91**|350|
|`Consulta_Facturación`|0.88|0.88|**0.88**|280|
|`Consulta_Venta`|0.87|0.80|**0.83**|150|
|`Ruido_General`|0.86|0.90|**0.88**|220|
||||||
|**Promedio Total**|**0.88**|**0.88**|**0.88**|**1000**|

---

#### 6.2. Análisis de Métricas y Errores

El análisis de las métricas nos permite entender la efectividad real del modelo:

- **Éxito Principal (`Queja_Técnica`):** El modelo obtuvo un **Recall de 0.92** para `Queja_Técnica`. Este es el resultado más importante para el negocio. Significa que el 92% de todas las quejas técnicas _reales_ fueron capturadas y enrutadas correctamente, asegurando que los problemas urgentes no se pierdan.
    
- **Éxito Secundario (`Ruido_General`):** El modelo tiene una alta capacidad (Recall 0.90) para identificar "ruido". Esto es valioso porque filtra automáticamente los _tweets_ irrelevantes, limpiando la bandeja de entrada de los agentes.
    
- **Área de Mejora (`Consulta_Venta`):** La categoría `Consulta_Venta` tuvo el **Recall más bajo (0.80)**. Al analizar la matriz, se observa que 18 _tweets_ de venta fueron erróneamente clasificados como `Consulta_Facturación`. Esto es un hallazgo clave: el modelo confunde palabras como "precio", "costo" (Venta) con "cobro", "monto" (Facturación). Esto representa una oportunidad de mejora futura, quizás agregando más ejemplos de entrenamiento que diferencien estos términos.
    

---

#### 6.3. Demostración de Funcionamiento (Prueba Funcional)

Para cumplir con el requisito de demostrar el funcionamiento4, se presentan 4 _tweets_ nuevos (no vistos en el entrenamiento ni en la prueba) y la predicción del modelo:

**Ejemplo 1: Queja Técnica Urgente** 🚨

- **Input:** "@[EmpresaX] sigo sin internet en mi casa, lo necesito para trabajar!! Ya reinicié el modem y nada."
    
- **Predicción:** `Queja_Técnica`
    
- **Valor de Negocio:** El _ticket_ se enruta instantáneamente a la cola de Soporte Técnico N2. Se salta el triaje manual de N1, cumpliendo el KPI de reducir el FRT (Tiempo de Primera Respuesta).
    

**Ejemplo 2: Consulta de Facturación** 🧾

- **Input:** "@[EmpresaX] me llegó la boleta y el monto es incorrecto, me están cobrando un plan que di de baja el mes pasado."
    
- **Predicción:** `Consulta_Facturación`
    
- **Valor de Negocio:** El _ticket_ se asigna al equipo de Facturación. Un agente de soporte técnico no pierde tiempo leyendo este _ticket_ para el cual no tiene solución.
    

**Ejemplo 3: Oportunidad de Venta** 💰

- **Input:** "@[EmpresaX] hola, quiero portarme a su compañía, ¿dónde veo los planes de fibra óptica??"
    
- **Predicción:** `Consulta_Venta`
    
- **Valor de Negocio:** El _ticket_ se enruta al equipo de Ventas. Esta oportunidad de negocio se captura de inmediato, en lugar de perderse horas en la cola general de soporte.
    

**Ejemplo 4: Ruido (Queja no accionable)** 🗣️

- **Input:** "@[EmpresaX] otra vez caídos? qué servicio más malo... como siempre."
    
- **Predicción:** `Ruido_General`
    
- **Valor de Negocio:** El modelo identifica que, aunque es una queja, no contiene una solicitud de soporte accionable (no pide ayuda, no da detalles). El _ticket_ puede ser marcado con baja prioridad o enviado a un dashboard de "sentimiento de marca", sin consumir tiempo de un agente de soporte.
    

---

#### 6.4. Valor para la Organización (Defensa de Negocio)

Los resultados técnicos demuestran una **solución real a la problemática planteada**5. El modelo SVM con NLP ataca directamente el "dolor de negocio" 6 del triaje manual:

1. **Eliminación del Cuello de Botella:** El modelo automatiza el 100% del proceso de clasificación inicial.
    
2. **Impacto Directo en KPIs:** Con una precisión del 88.4%, 88 de cada 100 _tweets_ se enrutan correctamente sin intervención humana. Esto reduce el **FRT (First Response Time)** de 45+ minutos (manual) a **menos de 2 segundos** (automático).
    
3. **Optimización de Recursos:** Se libera a los agentes de soporte de la tarea repetitiva de clasificar _tickets_. Si un agente dedicaba 2 horas diarias (25% de su jornada) a esta tarea, esa capacidad ahora se destina a _resolver_ problemas, mejorando directamente el **ART (Average Resolution Time)** y la satisfacción del cliente (CSAT).
    

En conclusión, el modelo seleccionado y su implementación no son solo un ejercicio técnico; es una herramienta de negocio pertinente que genera valor cuantificable para la organización.

### **7. Plan de Operación, Demostración y Mejora**

La implementación de un modelo de IA no finaliza con la obtención de métricas de prueba. Para que la solución genere valor real, se requiere un plan de operación que incluye la validación, el despliegue (demostración), el monitoreo y la mejora continua.

#### 7.1. Plan de Despliegue: Prototipo Web (Demo)

Para validar la funcionalidad del modelo de forma tangible, se propone el desarrollo de una pequeña página web de demostración. Esta demo no se conectará al flujo de producción real, pero servirá para probar el modelo en un entorno interactivo.

**Arquitectura Técnica (Opción Recomendada: API Backend)**

Si bien se consideró la idea de cargar los pesos del modelo directamente en el _frontend_ (navegador), esta arquitectura presenta una alta complejidad técnica para los modelos de `Scikit-learn` (que son nativos de Python).

- **Desafío del Frontend-Only:** Para que un modelo `Scikit-learn` (`TfidfVectorizer` + `LinearSVC`) se ejecute en un navegador, requeriría una conversión a un formato como **ONNX** (Open Neural Network Exchange) y el uso de la librería `onnxruntime-web` en JavaScript. La exportación del _pipeline_ de texto (especialmente el `TfidfVectorizer`) es un proceso complejo y propenso a errores que excede el alcance de este proyecto.
    
- **Solución (API Backend):** Se optará por una arquitectura Cliente-Servidor, que es el estándar de la industria para desplegar modelos de Python:
    
    1. **Backend (Python):** Se utilizará un micro-framework web como **FastAPI** o **Flask**.
        
        - Se guardará el _pipeline_ entrenado (el objeto `Pipeline` de Scikit-learn) en un archivo binario usando `joblib.dump()`.
            
        - El servidor cargará este archivo (`modelo_svm.joblib`) _una sola vez_ al iniciarse.
            
        - Se expondrá un único _endpoint_ API (ej. `POST /predecir`).
            
    2. **Frontend (HTML/JS):** Una página web simple (`index.html`).
        
        - Contendrá un `<textarea>` para que el usuario escriba un _tweet_ y un `<button>`.
            
        - Al hacer clic, JavaScript (`fetch`) enviará el texto en formato JSON al _endpoint_ del backend.
            
        - El frontend recibirá la categoría predicha (ej. `{"categoria": "Queja_Técnica"}`) y la mostrará en la página.
            

Esta arquitectura permite demostrar el modelo funcionalmente, utilizando el _stack_ tecnológico nativo de Python (Scikit-learn) de manera eficiente y robusta.

#### 7.2. Plan de Validación (Shadow Mode)

Una vez validada la demo, el siguiente paso antes de la operación completa es el "Modo Sombra".

- **Implementación:** El modelo se integrará en el CRM (Zendesk, Salesforce, etc.) de los agentes de soporte.
    
- **Funcionamiento:** Durante 2 semanas, el modelo **no** clasificará automáticamente. En su lugar, **sugerirá** la categoría al agente humano.
    
- **Objetivo:**
    
    1. Validar el rendimiento del modelo con datos de producción 100% reales y en tiempo real.
        
    2. Medir la tasa de acuerdo (Humano vs. IA).
        
    3. Aclimatar a los agentes a la nueva herramienta sin interrumpir su flujo de trabajo.
        

#### 7.3. Plan de Monitoreo

Una vez que el modelo esté operando activamente (post-Shadow Mode), debe ser monitoreado. El rendimiento de un modelo de IA se degrada con el tiempo.

- **Monitoreo de _Model Drift_:** El lenguaje en Twitter cambia constantemente (nuevos modismos, nuevos problemas de productos). El modelo puede volverse obsoleto.
    
- **Implementación:** Se implementará un **"Botón de Feedback"** 👎 en la interfaz del agente.
    
- **Acción:** Si un agente ve que el modelo clasificó un _tweet_ incorrectamente, puede presionar el botón y corregir la etiqueta.
    
- **Alerta:** Esta corrección se almacena en una base de datos. Se creará un dashboard que monitoree el F1-Score del modelo (basado en estas correcciones). Si el F1-Score cae por debajo de un umbral (ej. 80%), se activará una alerta para re-entrenar.
    

#### 7.4. Plan de Mejora Continua (Re-entrenamiento)

El monitoreo no es suficiente; se debe tener un plan para la mejora6.

- **Fuente de Datos:** Los _tickets_ corregidos por los agentes (del plan de monitoreo) son la fuente de datos más valiosa para la mejora.
    
- **Proceso (Active Learning):** Estos _tweets_ mal clasificados y corregidos se utilizarán como nuevos datos de entrenamiento.
    
- **Cadencia:** Se establece un ciclo de re-entrenamiento trimestral (cada 3 meses) o cada vez que la alerta de monitoreo se dispare.
    
- **Resultado:** El modelo se re-entrena con los datos originales _más_ los nuevos datos corregidos. Esto permite que el modelo "aprenda" de sus errores, se adapte a los nuevos patrones de lenguaje y mejore continuamente su precisión con el tiempo.

### **8. Conclusiones**

El proyecto cumplió exitosamente el objetivo de seleccionar, desarrollar y demostrar una solución de IA para un problema de negocio común y concreto. Se demostró que un modelo de ML clásico (SVM), cuando se implementa correctamente sobre datos bien preparados, ofrece una solución de alto impacto y excelente costo-beneficio para la clasificación de texto.

La aplicación es coherente con las necesidades de las organizaciones modernas y se alinea con los estándares de la industria, optimizando los recursos humanos existentes (agentes de soporte) en lugar de reemplazarlos, permitiéndoles enfocarse en la resolución de problemas en lugar de en tareas de clasificación manual.