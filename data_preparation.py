import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler




def RUN(df, SAMPLE):
    # Check for missing values in each column
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Replace missing values with the median (or other appropriate values)
    df['bmi'].fillna(df['bmi'].median(), inplace=True)


    # Encode categorical variables
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1}).astype(np.uint8)
    df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)

    # Separate the features and target variable
    X  = df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
    y = df['stroke']

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Instantiate the random oversampler and undersampler
    oversampler = RandomOverSampler(random_state=42)
    undersampler = RandomUnderSampler(random_state=42)


    if SAMPLE == "over":
        # Apply oversampling and undersampling to the data
        X_balanced, y_balanced = oversampler.fit_resample(X_scaled, y)
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    elif SAMPLE == "under":
        # Apply oversampling and undersampling to the data
        X_balanced, y_balanced = oversampler.fit_resample(X_scaled, y)
        X_balanced, y_balanced = undersampler.fit_resample(X_balanced, y_balanced)
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test

