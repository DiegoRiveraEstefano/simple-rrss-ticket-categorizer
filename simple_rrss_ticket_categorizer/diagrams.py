import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|mailto:\S+", "", text)  # URLs/correos
    text = re.sub(r"\d+", "", text)  # números
    text = re.sub(r"[^\w\s]", " ", text)  # Puntuación por espacio
    text = re.sub(r"\s+", " ", text).strip()  # Colapsar espacios

    return text


def load_data(path="data/processed/tickets_simple_categories.csv"):
    """Carga y prepara el dataset limpio."""
    df = pd.read_csv(path)
    # Asegurar que no haya NaNs en las columnas clave
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    return df


def plot_confusion_matrix(y_true, y_pred, labels):
    """Genera y guarda la matriz de confusión."""
    print("Generando Matriz de Confusión...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matriz de Confusión")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Matriz de Confusión guardada en 'confusion_matrix.png'")


def plot_classification_report(y_true, y_pred, labels):
    """Genera y guarda un heatmap del reporte de clasificación."""
    print("Generando Reporte de Clasificación...")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
    )

    # Excluir las métricas promedio para el heatmap de clases
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"])

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        report_df[["precision", "recall", "f1-score"]],
        annot=True,
        cmap="viridis",
        fmt=".2f",
    )
    plt.title("Reporte de Clasificación (Precisión, Recall, F1-Score)")
    plt.tight_layout()
    plt.savefig("classification_report.png")
    print("✅ Reporte de Clasificación guardado en 'classification_report.png'")


def plot_precision_recall_curve(model, X_test, y_test, labels):
    """Genera y guarda las curvas PR para cada clase (One-vs-Rest)."""
    print("Generando Curvas Precision-Recall...")
    try:
        # Binarizar las etiquetas para el gráfico multi-clase
        y_test_bin = label_binarize(y_test, classes=labels)

        # Obtener las puntuaciones (no probabilidades) de LinearSVC
        y_score = model.decision_function(X_test)

        # Configurar el gráfico
        plt.figure(figsize=(11, 8))
        ax = plt.gca()
        colors = plt.cm.get_cmap("viridis", len(labels))

        # Dibujar una curva PR para cada clase
        for i, (label, color) in enumerate(zip(labels, colors(range(len(labels))))):
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, i],
                y_score[:, i],
            )
            display = PrecisionRecallDisplay(precision=precision, recall=recall)
            display.plot(
                ax=ax,
                name=f"Clase: {label}",
                color=color,
            )

        plt.title("Curvas Precision-Recall (One-vs-Rest)")
        plt.xlabel("Recall (Exhaustividad)")
        plt.ylabel("Precision (Precisión)")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("precision_recall_curves.png")
        print("✅ Curvas P-R guardadas en 'precision_recall_curves.png'")

    except Exception as e:
        print(f"Error al generar curvas P-R: {e}")
        print("Asegúrate de que el modelo sea el pipeline final con ColumnTransformer.")


if __name__ == "__main__":
    # --- 1. Cargar Artefactos ---
    print("Cargando modelo y datos...")
    model = joblib.load("models/simple_ticket_classifier.pkl")
    df = load_data()

    # --- 2. Recrear el Test Set ---
    # Es crucial usar el mismo random_state y test_size que en el entrenamiento
    # para obtener las mismas métricas.

    # Asumiendo que el pipeline guardado usa ColumnTransformer
    X = df[["subject", "body"]]
    y = df["Simple_Target"]

    # Obtener la lista de etiquetas únicas en el orden correcto
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Datos de prueba cargados: {len(y_test)} muestras.")

    # --- 3. Obtener Predicciones ---
    y_pred = model.predict(X_test)

    # --- 4. Generar Gráficos ---
    plot_confusion_matrix(y_test, y_pred, labels)
    plot_classification_report(y_test, y_pred, labels)
    plot_precision_recall_curve(model, X_test, y_test, labels)

    print("\n--- Reporte de Texto (Consola) ---")
    print(classification_report(y_test, y_pred, labels=labels))
