import pandas as pd
from config import GZ_DATA, EMBEDDINGS
import gzip

def return_data_for_cbow(filepath):
    data = pd.read_csv(filepath)

    # drop duplicated lines
    data = data.drop_duplicates(['sentence'], keep='last')

    # drop any empty row
    data = data.dropna()

    # convert the column into a list
    data = data.sentence.tolist()

    #split data for Word2Vec model
    splitted = [i.split() for i in data]

    return splitted

def return_line_sentence_data_format(filepath):
    data = pd.read_csv(filepath)

    # drop duplicated lines
    data = data.drop_duplicates(['sentence'], keep='last')

    # drop any empty row
    data = data.dropna()

    return data


def return_gz_format(filepath):
    fp = open(filepath, "rb")
    data = fp.read()
    bindata = bytearray(data)
    with gzip.open(GZ_DATA, "wb") as f:
        f.write(bindata)
    return GZ_DATA
