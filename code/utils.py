import pandas as pd
from config import DATA_FOLDER

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
