import pandas as pd
from config import COMBINED_TAB, EMBEDDINGS
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import itertools
from utils import build_lemma, duplicate_lemmas, padding, pair_words

'''
VARIABLES DECLARATION
'''

def setup_dataframe(embeddings_path, gold_data_path):
    model = KeyedVectors.load_word2vec_format(embeddings_path)
    df = pd.read_csv(gold_data_path, sep='\t')
    df.columns = ['word_a', 'word_b', 'gold_score']
    print("Setting up data frame......")
    df['word_a'] = df.word_a.str.lower()
    df['word_b'] = df.word_b.str.lower()

    embeddings_list = []
    for key, value  in model.wv.vocab.items():
       embeddings_list.extend([(key, list(model.wv.word_vec(key)))])

    embeddings_dict = build_lemma(embeddings_list)
    lemmas = [sense.split('_bn')[0] for sense in embeddings_dict.keys()]
    babel_id = ["_bn" + sense.split('_bn')[1] for sense in embeddings_dict.keys()]


    df['a_vec'] = df.word_a.map(lambda x: pair_words(x, lemmas,babel_id, embeddings_dict))
    df['b_vec'] = df.word_b.map(lambda x: pair_words(x, lemmas,babel_id,embeddings_dict))


    df['a_vec'] = df['a_vec'].apply(padding)
    df['b_vec'] = df['b_vec'].apply(padding)
    print("Dataframe has been setup......")
    return df

def cosine_sim_for_pairs(word_a, word_b):
    cosine_values = []
    for i in itertools.product(word_a, word_b):
        cos = 1 - cosine(i[0], i[1])
        cosine_values.append(cos)
    return max(cosine_values)

def cosine_sim_for_unfound(row):
  if row.a_vec == "OVT" or row.b_vec == "OVT":
    return -1
  else:
    return cosine_sim_for_pairs(row.a_vec, row.b_vec)

def spearman(dataframe):
    print("Computing cosine similarities for all pairs")
    dataframe['cosine_sim'] = dataframe.apply(lambda x: cosine_sim_for_unfound(x), axis=1)
    # dataframe = dataframe[dataframe['cosine_sim'] != -1]

    assert len(dataframe.gold_score) == len(dataframe.cosine_sim)
    print("Computing spearmans rank correlation for all gold scores and embeddings cosine scores")
    spearmans_rank_correlation = spearmanr(dataframe.gold_score, dataframe.cosine_sim)[0]

    return spearmans_rank_correlation

def main():
    df = setup_dataframe(EMBEDDINGS,COMBINED_TAB)
    spearman_score = spearman(df)
    print("The Spearman correlation is:",spearman_score)


if __name__ == '__main__':
    main()
