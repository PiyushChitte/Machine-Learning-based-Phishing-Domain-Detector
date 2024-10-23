"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

   
data=pd.read_csv('uci_dataset.csv')
data=data.drop('id',1)

x=data.drop('Result',axis=1)
y=data['Result']

    
#print(x.shape)
#print(x.isnull().sum())

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rfc=RandomForestClassifier(random_state=42)
rfc=rfc.fit(x_train_scaled, y_train)

score=rfc.score(x_test_scaled , y_test)
#print("Score = ",score)
print("Accuracy score = ",score*100)

joblib.dump(rfc, open('rfc_model.pkl', 'wb'))
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('uci_dataset.csv')

# Feature engineering and preprocessing
# ... (Feature selection, handling missing values, encoding categorical features, etc.)

# Split data into training and testing sets
X = data.drop('Result', axis=1)
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Random Forest classifier
rfc = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get best hyperparameters and train the final model
best_params = grid_search.best_params_
final_rfc = RandomForestClassifier(**best_params, random_state=42)
final_rfc.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = final_rfc.predict(X_test_scaled)
#print(classification_report(y_test, y_pred))

joblib.dump(rfc, open('rfc_model.pkl', 'wb'))
