import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



def RUN(X_train, X_test, y_train, y_test):
    # Fit a linear SVM model
    linear_svc = SVC(kernel='linear')
    linear_svc.fit(X_train, y_train)
    linear_predictions = linear_svc.predict(X_test)
    linear_predictions = np.array(linear_predictions)
    #print("Linear SVM Model Performance:")
    #print(classification_report(y_test, linear_predictions))

    # Fit a Gaussian kernel SVM model
    gaussian_svc = SVC(kernel='rbf')
    gaussian_svc.fit(X_train, y_train)
    gaussian_predictions = gaussian_svc.predict(X_test)
    gaussian_predictions = np.array(gaussian_predictions)
    #print("Gaussian Kernel SVM Model Performance:")
    #print(classification_report(y_test, gaussian_predictions))

    # Decision Tree Classifier
    DT_MODEL = DecisionTreeClassifier()
    DT_PIPE = StandardScaler()
    DT_MODEL.fit(DT_PIPE.fit_transform(X_train), y_train)
    DT_predictions = DT_MODEL.predict(X_test)
    #print("Decision Tree Model Performance:")
    #print(classification_report(y_test, DT_predictions))


    # Random Forest Classifier
    RF_MODEL = RandomForestClassifier()
    RF_PIPE = StandardScaler()
    RF_MODEL.fit(RF_PIPE.fit_transform(X_train), y_train)
    RF_predictions = RF_MODEL.predict(X_test)
    #RF_predictions = cross_val_score(RF_MODEL, RF_PIPE.fit_transform(X_test), y_test, cv=5)
    #print("Random Forest Model Performance:")
    #print(classification_report(y_test, RF_predictions))

    # Logistic Regression Classifier
    LR_MODEL = LogisticRegression()
    LR_PIPE = StandardScaler()
    LR_MODEL.fit(LR_PIPE.fit_transform(X_train), y_train)
    LR_predictions = LR_MODEL.predict(X_test)
    #print("Linear Regression Model Performance:")
    #print(classification_report(y_test, LR_predictions))

    return y_test, RF_predictions, DT_predictions, LR_predictions, linear_predictions, gaussian_predictions
