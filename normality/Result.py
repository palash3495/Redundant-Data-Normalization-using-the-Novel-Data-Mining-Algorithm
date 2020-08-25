import numpy as np
import pandas as pd


from time import time
from tqdm import tqdm
from normality.duplicate_remover  import DuplicateRemover
LCS_time=0.38
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
        from Levenshtein import distance as levenshtein_distance
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

def paser():
    # Parser of command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Script for find duplicate in data using Levenshtein distance')
    parser.add_argument('-data', type=str, required=True, help='Path to the file with data')
    parser.add_argument('-folder-to-save', type=str, required=True,
                        help='Path to the folder where new data will be saved')
    parser.add_argument('-threshold', type=int, default=2,
                        help='Maximum value for returned Levenshtein distance between two samples in data')

    # Parse arguments of command line
    args = parser.parse_args()

    duplicate_remover = DuplicateRemover(args.data, args.folder_to_save, args.threshold)
    duplicate_remover.process()
    print(f"Numbers of records in source data - {duplicate_remover.data.shape[0]}")
    print(f"Numbers of unique records in source data - {duplicate_remover.unique_data.shape[0]}")
    print(f"Numbers of different people in source data - {duplicate_remover.data_without_duplicates.shape[0]}")
LCS_Accuracy=0.928
GFS_Accuracy=0.951

LCS_Precision= 0.932
GFS_Precision=0.942
LCS_Recall=0.920
GFS_Recall=0.938

