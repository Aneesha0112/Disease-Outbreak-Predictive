
import numpy as np
from mrjob.job import MRJob
import csv
from io import StringIO
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class HadoopJob(MRJob):

    def mapper(self, _, line):

        rows = next(csv.reader(StringIO(line)))
        df = pd.DataFrame([rows])
        df.columns = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
                      'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue', 'Gastrointestinal', 'Abroad travel', 'Contact with COVID Patient',
                      'Attended Large Gathering', 'Visited Public Exposed Places', 'Family working in Public Exposed Places', 'Wearing Masks',
                      'Sanitization from Market', 'COVID-19']
        df = df.fillna(df.mean())
        json_data = df.to_dict(orient='records')[0]
        yield None, json.dumps(json_data)

    def reducer(self, _, values):

        data = []
        for json_data in values:
            df = pd.DataFrame.from_dict([json.loads(json_data)])
            data.append(df)
        final_df = pd.concat(data, ignore_index=True)
        processed_df = pd.DataFrame(final_df)

        train_test_data = self.get_split_data(processed_df)

        SVM_accuracy = self.get_SVM_accuracy(train_test_data)
        RF_accuracy = self.get_RF_accuracy(train_test_data)
        ANN_accuracy = self.get_ANN_accuracy(train_test_data)

        json_accuracies = {"Decision Tree Accuracy": SVM_accuracy,
                           "Bagging Accuracy": RF_accuracy,
                           "Random Forest Accuracy": ANN_accuracy}

        json_object = json.dumps(json_accuracies, indent=4)

        local_output_path = "/Users/Ani/Desktop/prediction.json"
        with open(local_output_path, "w") as outfile:
            outfile.write(json_object)

        hdfs_output_path = '/Ani/outputs/prediction.json'

        os.system(f'hadoop fs -rm -f {hdfs_output_path}')
        os.system(f'hadoop fs -put {local_output_path} {hdfs_output_path}')


    def get_split_data(self, processed_df):
        X_train, X_test, Y_train, Y_test = train_test_split(processed_df.drop('COVID-19', axis=1), processed_df['COVID-19'], test_size=0.2, random_state=100)
        return [X_train, X_test, Y_train, Y_test]

    def SVM_accuracy(self, train_test_data):
        X_train, X_test, Y_train, Y_test = train_test_data
        svm_model_covid = SVC(kernel='linear', random_state=0)
        svm_model_covid.fit(X_train, Y_train.values.ravel())
        svm_predictions_covid = svm_model_covid.predict(X_test)
        accuracy = accuracy_score(Y_test, svm_predictions_covid) * 100
        return float("{:.2f}".format(accuracy))

    def RF_accuracy(self, train_test_data):
        X_train, X_test, Y_train, Y_test = train_test_data
        rf_model_covid = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        rf_model_covid.fit(X_train, Y_train.values.ravel())
        rf_predictions_covid = rf_model_covid.predict(X_test)
        accuracy = accuracy_score(Y_test, rf_predictions_covid) * 100
        return float("{:.2f}".format(accuracy))

    def ANN_accuracy(self, train_test_data):
        X_train, X_test, Y_train, Y_test = train_test_data
        ann_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=6, activation='relu', input_dim=X_train.shape[1]),
            tf.keras.layers.Dense(units=6, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')])
        ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        ann_model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))
        ann_predictions = (ann_model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(Y_test, ann_predictions) * 100
        return float("{:.2f}".format(accuracy))

if __name__ == '__main__':
    input_path = "/Users/Ani/Desktop/Dataset.csv"
    HadoopJob.run()