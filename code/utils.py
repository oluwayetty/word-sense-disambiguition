import pandas as pd
import ast
from config import FINAL_DATA, LEMMAIDS_DATA

def final_clean_up(filepath):
    data = pd.read_csv(filepath)
    data.drop_duplicates(['sentence'], keep='last')
    data['anchor'] = data.anchor.apply(lambda x: ast.literal_eval(x))
    data['len_anch'] = data['anchor'].apply(lambda x: len(x))
    data['lemma_IDs'] = data.lemma_IDs.apply(lambda x: ast.literal_eval(x))
    data = data[data['len_anch'] != 0]
    return data

def build_lemmas_into_sentence(sentence):
  data = pd.read_csv(LEMMAIDS_DATA)
  # data.drop(['Unnamed: 0'], axis=1, inplace=True)
  # import ipdb; ipdb.set_trace()
  row = data[data['sentence'] == sentence]
  anchor = row['anchor'].tolist()[0]
  lemma_id = row['lemma_IDs'].tolist()[0]
  anchor_lemma = dict(zip(anchor, lemma_id))
  sent_list = sentence.split(' ')
  uni = list(set(sent_list).intersection(anchor))
  a = []

  for i, v in enumerate(sent_list):
    if v in uni:
      sent_list[i] = anchor_lemma.get(v)
      a.append(sent_list)
  try:
    needed = a[-1]
    return " ".join(needed)
  except IndexError as error:
    print(error)

def apply_function():
    dataset = final_clean_up(LEMMAIDS_DATA)
    # build_lemmas_into_sentence(dataset['sentence'])
    dataset['sentences'] = dataset['sentence'].apply(build_lemmas_into_sentence)
    dataset.to_csv(FINAL_DATA)
    return dataset

# apply_function()
