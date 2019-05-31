import pandas as pd
from config import GZ_DATA, EMBEDDINGS
import gzip
from collections import defaultdict


def return_data_for_cbow(filepath):
    data = pd.read_csv(filepath, sep='\n')

    # drop any empty row
    data = data.dropna()

    # convert the column into a list
    data = data.sentence.tolist()

    #split data for Word2Vec model
    splitted = [i.split() for i in data]

    return splitted

def return_gz_format(filepath):
    fp = open(filepath, "rb")
    data = fp.read()
    bindata = bytearray(data)
    with gzip.open(GZ_DATA, "wb") as f:
        f.write(bindata)
    return GZ_DATA

def build_lemma(vecs):
  temp = defaultdict(list)
  dict_ = {}

  for x, y in vecs:
     temp[x].append(y)

  for k, v in temp.items():
    dict_[k] = v[0]
  return dict_

def duplicate_lemmas(lst, item):
  return [i for i, x in enumerate(lst) if x == item]

def padding(x):
 return x if type(x[0]) == list or type(x)== str else [x]

def pair_words(word,lemmas,id, dict_):
  if word in lemmas:
    if lemmas.count(word) > 1:
      hold = []
      #####Get index of occurence of the repeated words
      dups = duplicate_lemmas(lemmas, word)
      output = [dict_.get(word + id[i]) for i in dups]
      return output
    else:
      build_word = dict_.get(word + id[lemmas.index(word)])
      return build_word
  else:
      return "OVT"
