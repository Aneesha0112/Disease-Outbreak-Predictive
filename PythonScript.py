
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/content/Dataset.csv'
data = pd.read_csv(dataset_path)

# Handle null values
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode 'Yes' and 'No' to 1 and 0 for all columns
label_encoder = LabelEncoder()

for column in data_imputed.columns:
    if data_imputed[column].dtype == 'O':  
        data_imputed[column] = label_encoder.fit_transform(data_imputed[column])

data_imputed

target_variable = 'COVID-19'

# Bar chart for Symptom Frequency
symptom_columns = data_imputed.columns[:-1]
symptom_frequencies = data_imputed[symptom_columns].sum()

plt.figure(figsize=(12, 6))
sns.barplot(x=symptom_frequencies.index, y=symptom_frequencies.values, palette='viridis')
plt.title('Symptom Frequency')
plt.xlabel('Symptoms')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Heatmap or Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data_imputed.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pie chart for Disease Distribution
disease_distribution = data_imputed[target_variable].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(disease_distribution, labels=disease_distribution.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('Covid Test Distribution')
plt.show()

# Stacked Bar Chart for Symptom-Disease Relationship
symptom_disease_counts = pd.crosstab(data_imputed[target_variable], data_imputed[symptom_columns].sum(axis=1))

symptom_disease_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
plt.title('Symptom-Disease Relationship')
plt.xlabel('Disease Presence')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

"""Visualization"""

sns.countplot(x='COVID-19', data=data_imputed)

plt.title('Number of Confirmed Covid-19 Cases')
plt.xlabel('Covid-19 Test Result')
plt.ylabel('No of Patients')

sns.countplot(x='COVID-19', data=data_imputed)

plt.title('Number of Confirmed SwineFlu Cases')
plt.xlabel('SwineFlu Test Result')
plt.ylabel('No of Patients')

sns.countplot(x='Breathing Problem',data=data_imputed)

plt.title('Number of Patients with Breathing Problem')
plt.xlabel('Breathing Problem')
plt.ylabel('No of Patients')

sns.countplot(x='Sore throat',data=data_imputed)

plt.title('Number of Patients with Sore throat')
plt.xlabel('Sore throat')
plt.ylabel('No of Patients')

data_imputed.dtypes.value_counts()

data_imputed.describe(include='all')

data_imputed.hist(figsize=(20,15));

"""Corelation"""

cor=data_imputed.corr()
cor.style.background_gradient(cmap='coolwarm',axis=None)

"""Feature Selection"""

data_imputed=data_imputed.drop('Running Nose',axis=1)
data_imputed=data_imputed.drop('Chronic Lung Disease',axis=1)
data_imputed=data_imputed.drop('Headache',axis=1)
data_imputed=data_imputed.drop('Heart Disease',axis=1)
data_imputed=data_imputed.drop('Diabetes',axis=1)
data_imputed=data_imputed.drop('Gastrointestinal ',axis=1)
data_imputed=data_imputed.drop('Asthma',axis=1)
data_imputed=data_imputed.drop('Fatigue ',axis=1)

cor=data_imputed.corr()
cor.style.background_gradient(cmap='coolwarm',axis=None)

Y = pd.DataFrame()
Y['COVID-19'] = data_imputed['COVID-19'].copy()
Z = pd.DataFrame()
Z['SwineFlu'] = data_imputed['SwineFlu'].copy()
X = data_imputed.drop('COVID-19', axis=1).copy()
X = X.drop('SwineFlu', axis=1)

Y.columns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


X_train_covid, X_test_covid, Y_train_covid, Y_test_covid = train_test_split(X, Y, test_size=0.2, random_state=42)

svm_model_covid = SVC(kernel='linear', random_state=0)
svm_model_covid.fit(X_train_covid, Y_train_covid.values.ravel())

svm_predictions_covid = svm_model_covid.predict(X_test_covid)


svm_accuracy_covid = accuracy_score(Y_test_covid, svm_predictions_covid)
print(f'SVM Accuracy for Covid: {svm_accuracy_covid}')


x_train_swineflu, x_test_swineflu, Z_train_swineflu, Z_test_swineflu = train_test_split(X, Z, test_size=0.3, random_state=35)


svm_model_swineflu = SVC(kernel='linear', random_state=0)
svm_model_swineflu.fit(x_train_swineflu, Z_train_swineflu.values.ravel())

svm_predictions_swineflu = svm_model_swineflu.predict(x_test_swineflu)

svm_accuracy_swineflu = accuracy_score(Z_test_swineflu, svm_predictions_swineflu)
print(f'SVM Accuracy for SwineFlu: {svm_accuracy_swineflu}')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X_train_covid, X_test_covid, Y_train_covid, Y_test_covid = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_model_covid = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model_covid.fit(X_train_covid, Y_train_covid.values.ravel())

rf_predictions_covid = rf_model_covid.predict(X_test_covid)

rf_accuracy_covid = accuracy_score(Y_test_covid, rf_predictions_covid)
print(f'Random Forest Accuracy for Covid: {rf_accuracy_covid}')

x_train_swineflu, x_test_swineflu, Z_train_swineflu, Z_test_swineflu = train_test_split(X, Z, test_size=0.3, random_state=35)

rf_model_swineflu = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model_swineflu.fit(x_train_swineflu, Z_train_swineflu.values.ravel())

rf_predictions_swineflu = rf_model_swineflu.predict(x_test_swineflu)

rf_accuracy_swineflu = accuracy_score(Z_test_swineflu, rf_predictions_swineflu)
print(f'Random Forest Accuracy for SwineFlu: {rf_accuracy_swineflu}')

feature_importances_covid = rf_model_covid.feature_importances_

feature_df_covid = pd.DataFrame({'Feature': X_train_covid.columns, 'Importance': feature_importances_covid})

feature_df_covid = feature_df_covid.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_df_covid['Feature'], feature_df_covid['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance Chart for Covid')
plt.xticks(rotation=45, ha='right')
plt.show()

feature_importances_swineflu = rf_model_swineflu.feature_importances_

feature_df_swineflu = pd.DataFrame({'Feature': x_train_swineflu.columns, 'Importance': feature_importances_swineflu})

feature_df_swineflu = feature_df_swineflu.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_df_swineflu['Feature'], feature_df_swineflu['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance Chart for SwineFlu')
plt.xticks(rotation=45, ha='right')
plt.show()

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train, x_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.3, random_state=35)

ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
ann_model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))
ann_model.fit(x_train, Z_train, epochs=10, batch_size=32, validation_data=(x_test, Z_test))
# Predict on the test set
ann_predictions = (ann_model.predict(X_test) > 0.5).astype("int32")

# Calculate and print accuracy
ann_accuracy = accuracy_score(Y_test, ann_predictions)
print(f'ANN Accuracy for Covid: {ann_accuracy}')

ann_predictions = (ann_model.predict(x_test) > 0.5).astype("int32")

# Calculate and print accuracy
ann_accuracy = accuracy_score(Z_test, ann_predictions)
print(f'ANN Accuracy for SwineFlu: {ann_accuracy}')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

X_train_covid, X_test_covid, Y_train_covid, Y_test_covid = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_model_covid = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model_covid.fit(X_train_covid, Y_train_covid.values.ravel())

feature_names_covid = X_train_covid.columns
print("Enter values for each feature:")
user_input_covid = {}
for feature_name in feature_names_covid:
    value = float(input(f"{feature_name}: "))
    user_input_covid[feature_name] = value

user_input_df_covid = pd.DataFrame([user_input_covid], columns=feature_names_covid)
prediction_covid = rf_model_covid.predict(user_input_df_covid)

if prediction_covid == 0:
    print('The model predicts that you do not have Covid.')
else:
    print('The model predicts that you have Covid.')