from pathlib import Path

from dataset import Dataset
# from preprocessor import Preprocessor

if __name__ == '__main__':
    DTA_FOLDER_PATH = Path("dataset")
    TRAIN_FILE_NAME = Path("mlchallenge_set_2021.tsv")
    TRAIN_FILE_PATH = DTA_FOLDER_PATH / TRAIN_FILE_NAME
    VALID_FILE_NAME = Path("mlchallenge_set_validation.tsv")
    VALID_FILE_PATH = DTA_FOLDER_PATH / VALID_FILE_NAME

    p = Preprocessor()
    p = Preprocessor(TRAIN_FILE_PATH)
    dataset = p.parse_data_file()
    dataset.print_random_data()