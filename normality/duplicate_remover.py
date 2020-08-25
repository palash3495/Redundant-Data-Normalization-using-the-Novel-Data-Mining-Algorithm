import numpy as np
import pandas as pd

from Levenshtein import distance as levenshtein_distance
from time import time

from sklearn import metrics
from sklearn.metrics import *
from tqdm import tqdm


class DuplicateRemover:
    """
    Class for removing duplicates
    """

    def __init__(self, data_path, folder_to_save, threshold=2):
        """
        Init function for duplicate remover.
        :param data_path: path to the file with data.
        :param folder_to_save: path to the folder where the processed file has to be saved.
        :param threshold: maximum value for returned Levenshtein distance between two samples in data
        """
        # Read data from source
        self.data = pd.read_csv(data_path)
        # Replace missed values by empty string
        self.data = self.data.fillna("")
        self.unique_data = None
        self.data_without_duplicates = None
        self.threshold = threshold
        self.folder_to_save = folder_to_save

    def process(self):
        """
        Function for process data and save new data without duplicates.
        1. Preprocess source data.
        2. Find unique values in source data.
        3. Drop duplicate names from source data.
        4. Drop extra columns.
        5. Save data without duplicates in names.
        """
        self.preprocess()
        self.find_unique()
        self.remove_duplicates()
        self.data_without_duplicates = self.data_without_duplicates.drop(columns=['full_name', 'find_unique'])
        self.data_without_duplicates.to_csv(f"{self.folder_to_save}/relateddata.csv")

        # Importing the dataset
        training_dataset = pd.read_csv('Datasets/ConferenceName_non_standard.csv')
        test_dataset = pd.read_csv('Datasets/ConferenceName_non_standard.csv')

        # Slicing and Dicing the dataset to separate features from predictions
        X = training_dataset.iloc[:, 0:132].values
        Y = training_dataset.iloc[:, -1].values

        # Dimensionality Reduction for removing redundancies
        dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()

        # Encoding String values to integer constants
        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(Y)

        # Splitting the dataset into training set and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        # Implementing the Decision Tree Classifier
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

        print(classifier)
        # make predictions
        expected = y_train
        predicted = classifier.predict(X_train)
        np.savetxt(
            'C:/Users/Godwit Tech/PycharmProjects/Recommendation_based_healthcare_system/staging/predictedDT.txt',
            predicted, fmt='%01d')
        # summarize the fit of the model
        GFS_accuracy = accuracy_score(expected, predicted)
        GFS_recall = recall_score(expected, predicted, average="weighted")
        GFS_precision = precision_score(expected, predicted, average="weighted")
        f1 = f1_score(expected, predicted, average="weighted")

        cm = metrics.confusion_matrix(expected, predicted)
        print(cm)
        tpr = float(cm[0][0]) / np.sum(cm[0])
        fpr = float(cm[1][1]) / np.sum(cm[1])
        print("%.3f" % tpr)
        print("%.3f" % fpr)
        print("4")
        print(GFS_accuracy)
        print(GFS_precision)
        print(GFS_recall)

        # Saving the information of columns
        cols = training_dataset.columns
        cols = cols[:-1]

    def preprocess(self):
        """
        Function for preprocess data.
        Merge given_name and surname to validate names.
        Merge [date_of_birth, sex, full_Name] columns to find right unique_values.
        Print number of records in source data,
        """
        self.data['full_name'] = self.data["given_name"] + " " + self.data["surname"]
        self.data['find_unique'] = self.data['date_of_birth'] + " " + self.data['sex'] + " " + self.data['full_name']


    def find_unique(self):
        """
        Function for finding unique values in data.
        Print number of unique records in source data.
        """
        self.unique_data = self.data.drop_duplicates(subset="find_unique")

    def remove_duplicates(self):
        """
        Function for removing duplicate names in data.
        Print number of different people in source data.
        """
        start = time()
        self.data_without_duplicates = self.function_for_remove_duplicates("full_name", self.threshold)

    def function_for_remove_duplicates(self, similar_column='full_name', threshold=2):
        """
        Function for process data, remove duplicate people records.
        :param similar_column: column by which find duplicates in data
        :param threshold: maximum value for returned Levenshtein distance between two samples in data
        :return: data without duplicates in similar_column.
        """
        dupl_indexes = []
        rows_number = self.unique_data.shape[0]
        for i in tqdm(range(rows_number - 1)):
            distances = np.array(
                [levenshtein_distance(self.unique_data[similar_column].values[i],
                                      self.unique_data[similar_column].values[j]) for j in
                 range(i + 1, rows_number)])
            matching_indexes = np.where(distances <= threshold)[0]
            matching_indexes = matching_indexes + i + 1

            d_b1 = self.unique_data['date_of_birth'].iloc[i]
            dupl_indexes += [self.unique_data.index[match] for match in matching_indexes if
                             self.unique_data['date_of_birth'].iloc[match] == d_b1]
        clean_df = self.unique_data.copy()
        for index_list in dupl_indexes:
            clean_df = clean_df.drop(index_list)

        return clean_df
