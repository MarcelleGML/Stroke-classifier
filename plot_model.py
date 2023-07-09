import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve


def RUN(y_test, predictions, title):
    # Calculate the confusion matrix
    cm_model = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(title)
    plt.subplot(2,1,1)
    sns.heatmap(cm_model, annot=True, cmap="gray")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")



    # Calculate the ROC curve
    plt.subplot(2, 1, 2)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.tight_layout()
    plt.show()