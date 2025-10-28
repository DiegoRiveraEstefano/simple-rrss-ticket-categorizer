import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay  # <--- NUEVO
from sklearn.metrics import auc  # <--- NUEVO
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve  # <--- NUEVO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# --- DEFINICIÓN DE PREPROCESS_TEXT ---
# Esta función debe estar aquí para que joblib.load() funcione
# (Como solucionamos en el paso anterior)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|mailto:\S+", "", text)  # URLs/correos
    text = re.sub(r"\d+", "", text)  # números
    text = re.sub(r"[^\w\s]", " ", text)  # Puntuación por espacio
    text = re.sub(r"\s+", " ", text).strip()  # Colapsar espacios
    # NOTA: Si tu modelo se entrenó con STEMMING, esa lógica
    # debe estar aquí también para que las predicciones sean correctas.
    return text


def load_data(path="data/processed/tickets_simple_categories.csv"):
    """Carga y prepara el dataset limpio."""
    df = pd.read_csv(path)
    # Asegurar que no haya NaNs en las columnas clave
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    return df


def plot_class_distribution(y, labels, filename="class_distribution.png"):
    """(NUEVO) Genera un gráfico de barras de la distribución de clases."""
    print("Generando Distribución de Clases...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y, order=labels, palette="viridis")
    plt.title("Distribución de Clases en el Dataset Completo")
    plt.ylabel("Número de Tickets")
    plt.xlabel("Categoría")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Distribución de Clases guardada en '{filename}'")


def plot_confusion_matrices(y_true, y_pred, labels):
    """
    (ACTUALIZADO) Genera y guarda DOS matrices de confusión:
    1. Con conteos absolutos.
    2. Normalizada por porcentaje.
    """
    print("Generando Matriz de Confusión (Conteos)...")
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matriz de Confusión (Conteos Absolutos)")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.savefig("confusion_matrix_counts.png")
    print("✅ Matriz de Confusión (Conteos) guardada en 'confusion_matrix_counts.png'")

    print("Generando Matriz de Confusión (Normalizada)...")
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",  # Formato de porcentaje
        cmap="Greens",  # Diferente color
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matriz de Confusión (Normalizada por % de Verdadera Etiqueta)")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.savefig("confusion_matrix_normalized.png")
    print(
        "✅ Matriz de Confusión (Normalizada) guardada en 'confusion_matrix_normalized.png'",
    )


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
        y_test_bin = label_binarize(y_test, classes=labels)
        y_score = model.decision_function(X_test)

        plt.figure(figsize=(11, 8))
        ax = plt.gca()
        colors = plt.cm.get_cmap("viridis", len(labels))

        for i, (label, color) in enumerate(zip(labels, colors(range(len(labels))))):
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, i],
                y_score[:, i],
            )
            display = PrecisionRecallDisplay(precision=precision, recall=recall)
            display.plot(ax=ax, name=f"Clase: {label}", color=color)

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


def plot_roc_curves(model, X_test, y_test, labels):
    """(NUEVO) Genera y guarda las curvas ROC/AUC para cada clase."""
    print("Generando Curvas ROC/AUC...")
    try:
        y_test_bin = label_binarize(y_test, classes=labels)
        y_score = model.decision_function(X_test)

        plt.figure(figsize=(11, 8))
        ax = plt.gca()
        colors = plt.cm.get_cmap("plasma", len(labels))

        # Añadir la línea de "azar"
        ax.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.50)")

        for i, (label, color) in enumerate(zip(labels, colors(range(len(labels))))):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            display.plot(
                ax=ax,
                name=f"Clase: {label} (AUC = {roc_auc:.2f})",
                color=color,
            )

        plt.title("Curvas ROC (One-vs-Rest)")
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_auc_curves.png")
        print("✅ Curvas ROC/AUC guardadas en 'roc_auc_curves.png'")

    except Exception as e:
        print(f"Error al generar curvas ROC: {e}")


if __name__ == "__main__":
    # --- 1. Cargar Artefactos ---
    print("Cargando modelo y datos...")
    # Asegúrate de que la ruta a tu modelo sea correcta
    model = joblib.load("models/simple_ticket_classifier.pkl")
    df = load_data()

    # --- 2. Recrear el Test Set ---
    # Asumiendo que el pipeline guardado usa ColumnTransformer
    X = df[["subject", "body"]]
    y = df["Simple_Target"]
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

    # (NUEVO) Gráfico de distribución del dataset COMPLETO
    plot_class_distribution(y, labels)

    # (ACTUALIZADO) Genera ambas matrices de confusión
    plot_confusion_matrices(y_test, y_pred, labels)

    # Gráficos existentes
    plot_classification_report(y_test, y_pred, labels)
    plot_precision_recall_curve(model, X_test, y_test, labels)

    # (NUEVO) Gráfico de curvas ROC
    plot_roc_curves(model, X_test, y_test, labels)

    print("\n--- Reporte de Texto (Consola) ---")
    print(classification_report(y_test, y_pred, labels=labels))
