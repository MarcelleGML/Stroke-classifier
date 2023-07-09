import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def RUN(y_test,predictions, models):
    # Comparison
    table_data = []
    for model, pred in zip(models, predictions):
        report = classification_report(y_test, pred, output_dict=True)
        precision_1 = report["1"]["precision"]
        recall_1 = report["1"]["recall"]
        f1_score_1 = report["1"]["f1-score"]
        precision_0 = report["0"]["precision"]
        recall_0 = report["0"]["recall"]
        f1_score_0 = report["0"]["f1-score"]
        table_data.append([model, precision_1, precision_0, recall_1, recall_0, f1_score_1, f1_score_0])

    columns = ["Model", "Precision (1)", "Precision (0)", "Recall (1)", "Recall (0)", "F1-score (1)", "F1-score (0)"]
    idx = models
    df = pd.DataFrame(table_data, columns=columns, index=idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center", colColours=["lightgray"] * len(columns))

    plt.title("Model Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
