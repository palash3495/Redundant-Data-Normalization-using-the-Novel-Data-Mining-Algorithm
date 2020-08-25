import argparse

from normality.duplicate_remover  import DuplicateRemover

# Parser of command line arguments
parser = argparse.ArgumentParser(description='Script for find duplicate in data using Levenshtein distance')
parser.add_argument('-data', type=str, required=True, help='Path to the file with data')
parser.add_argument('-folder-to-save', type=str, required=True, help='Path to the folder where new data will be saved')
parser.add_argument('-threshold', type=int, default=2,
                    help='Maximum value for returned Levenshtein distance between two samples in data')

# Parse arguments of command line
args = parser.parse_args()

duplicate_remover = DuplicateRemover(args.data, args.folder_to_save, args.threshold)
duplicate_remover.process()
print(f"Numbers of records in source data - {duplicate_remover.data.shape[0]}")
print(f"Numbers of unique records in source data - {duplicate_remover.unique_data.shape[0]}")
print(f"Numbers of different people in source data - {duplicate_remover.data_without_duplicates.shape[0]}")