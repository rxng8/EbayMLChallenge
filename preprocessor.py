# import bs4
# import requests
import codecs
import sys

from dataset import Dataset
from data import Data

class Crawler:
    def __init__(self):
        pass

class Preprocessor:
    def __init__(self, data_path=None, valid_path=None):
        self.data_path = data_path
        self.valid_path = valid_path
        self.raw_data = []

    def parse_data_file(self, head=2000000):
        """
            TODO: Do we actually need a list? We had dataset class
        """
        dataset = Dataset()
        if self.data_path != None:
            # with codecs.open(self.data_path, 'r', encoding="unicode_escape") as f:
            with codecs.open(self.data_path, 'r', encoding="utf-8") as f:
                line = None
                # 10 first line is the header
                for i in range(11):
                    try:
                        line = f.readline()
                        print("This is header")
                    except:
                        print("Header parse error")
                cnt = 0
                total = 0
                while line:
                    total += 1
                    try:
                        # print("asdda")
                        # Logic
                        line = f.readline()
                        self.raw_data.append(line)
                        dataset.add_data(line)
                        # Counter
                        cnt += 1
                        if cnt >= head:
                            break
                    except IndexError:
                        print(f"Wrong index at line data {cnt} and value: {line}")
                    except:
                        # print("something wrong!")
                        print("Unexpected error:", sys.exc_info()[0])
                print(f"Read {cnt} lines out of {total} lines.")
        else:
            print("No data_path specified.")
        return dataset

    def parse_valid_fie(self):
        if self.valid_path != None:
            pass
        else:
            pass
        pass

    




############################    Test     ####################################

from pathlib import Path
from dataset import Dataset
from preprocessor import Preprocessor

DTA_FOLDER_PATH = Path("dataset")
TRAIN_FILE_NAME = Path("mlchallenge_set_2021.tsv")
TRAIN_FILE_PATH = DTA_FOLDER_PATH / TRAIN_FILE_NAME
VALID_FILE_NAME = Path("mlchallenge_set_validation.tsv")
VALID_FILE_PATH = DTA_FOLDER_PATH / VALID_FILE_NAME

def test_parse_data():
    p = Preprocessor()
    p = Preprocessor(TRAIN_FILE_PATH)
    dataset = p.parse_data_file()
    dataset.print_random_data()

if __name__ == '__main__':
    test_parse_data()
    