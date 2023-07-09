import pandas as pd
import data_preparation as PREPARATION
import data_visualization as VISUALIZATION
import model_train as MODEL
import plot_model as PLOT
import model_compare as COMPARE

# Specify the path to the stroke dataset file
dataset_path = r'D:\Desktop\Marcelle\Portifolio\kaggle\healthcare-dataset-stroke-data.csv'

# Read the dataset into a DataFrame
df = pd.read_csv(dataset_path)

# Run data_preparation
X_train, X_test, y_train, y_test = PREPARATION.RUN(df, "over")

# Run data_visualization
VISUALIZATION.RUN(df)

# Run classification
y_test, RF_predictions, DT_predictions, LR_predictions, linear_predictions, gaussian_predictions  = MODEL.RUN(X_train, X_test, y_train, y_test)
predictions = [RF_predictions, DT_predictions, LR_predictions, linear_predictions, gaussian_predictions]
models = ["Random Florest", "Decion Tree", "Linear Regression", "Linear SVM", "SVR ( Gaussian Kernel)"]

# PLOT
for title, data in zip(models, predictions):
    PLOT.RUN(y_test, data, title)

# Comparison
COMPARE.RUN(y_test,predictions,models)
