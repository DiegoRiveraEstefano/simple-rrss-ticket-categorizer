import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Carga de Datos ---
# Simularemos 3 departamentos de soporte usando 3 categorías del dataset.
# El dataset está en inglés, por lo que el modelo aprenderá inglés.
categories = [
    'comp.sys.ibm.pc.hardware', # Simula: "Soporte Técnico - Hardware"
    'misc.forsale',             # Simula: "Ventas y Facturación"
    'sci.electronics'           # Simula: "Soporte Técnico - Electrónica"
]
target_names = ["Hardware", "Ventas", "Electrónica"] # Nombres personalizados

print("Cargando datasets de entrenamiento y prueba...")
# Cargar datos de entrenamiento
# Se eliminan 'headers', 'footers' y 'quotes' para que el modelo
# aprenda del contenido del texto, no de los metadatos.
train_data = fetch_20newsgroups(
    subset='train', 
    categories=categories, 
    shuffle=True, 
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)
X_train = train_data.data
y_train = train_data.target

# Cargar datos de prueba (test)
test_data = fetch_20newsgroups(
    subset='test', 
    categories=categories, 
    shuffle=True, 
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)
X_test = test_data.data
y_test = test_data.target

print(f"Datos de entrenamiento cargados: {len(X_train)} tickets")
print(f"Datos de prueba cargados: {len(X_test)} tickets")


# --- 2. Definición del Pipeline (Modelo) ---
# Un Pipeline es una cadena de pasos. Es la forma correcta
# de implementar un flujo de ML en scikit-learn.

print("\nCreando el Pipeline de IA...")

[cite_start]# [cite: 38] Se definen las librerías, métodos y parámetros
text_classifier_pipeline = Pipeline([
    
    # Paso 1: 'vect' (Vectorizador)
    # Convierte el texto en una matriz de números usando TF-IDF.
    # TF-IDF mide la importancia de una palabra en un documento.
    ('vect', TfidfVectorizer(stop_words='english', max_features=5000)),

    # Paso 2: 'clf' (Clasificador)
    # Usa el algoritmo Naive Bayes Multinomial, que es muy
    # eficiente y funciona bien para la clasificación de texto.
    ('clf', MultinomialNB(alpha=0.01)) # alpha es un parámetro de suavizado
])


# --- 3. Entrenamiento del Modelo ---
print("Entrenando el modelo...")
text_classifier_pipeline.fit(X_train, y_train)
print("¡Modelo entrenado!")


# --- 4. Evaluación del Modelo ---
print("\nEvaluando el modelo con los datos de prueba...")
y_pred = text_classifier_pipeline.predict(X_test)

# Calcular la precisión (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión (Accuracy) del modelo: {accuracy * 100:.2f}%")

# Mostrar reporte detallado (Precisión, Recall, F1-Score)
print("\nReporte de Clasificación (Métricas Detalladas):")
print(classification_report(y_test, y_pred, target_names=target_names))


# --- 5. Demostración de Funcionamiento ---
print("\n--- DEMOSTRACIÓN: Predicción de nuevos tickets ---")

# Creamos nuevos "tickets" de soporte (deben estar en inglés)
new_tickets = [
    "My computer power supply is making a loud noise and smells like burning.",
    "I want to buy 15 units of your product, can I get a volume discount?",
    "The main circuit board of my device has a blown capacitor. I need a replacement.",
    "The new video card driver is not working with my monitor."
]

# El pipeline se encarga de todo el proceso (vectorizar y predecir)
predicted_categories_indices = text_classifier_pipeline.predict(new_tickets)

# Mapeamos los índices (0, 1, 2) a los nombres de las categorías
for ticket, category_index in zip(new_tickets, predicted_categories_indices):
    print(f"\nTicket: '{ticket}'")
    print(f"==> Predicción: {target_names[category_index]}")