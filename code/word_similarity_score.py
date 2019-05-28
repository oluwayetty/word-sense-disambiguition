import pandas as pd
from config import COMBINED_TAB, EMBEDDINGS
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

'''
VARIABLES DECLARATION
'''
df = pd.read_csv(COMBINED_TAB, sep='\t')
df.columns = ['word_a', 'word_b', 'gold']

# convert words from combined.tab to lower case
df['word_a'] = df.word_a.str.lower()
df['word_b'] = df.word_b.str.lower()

loaded_model = KeyedVectors.load_word2vec_format(EMBEDDINGS, binary=False)
words_in_vocab = loaded_model.wv.vocab.keys()

lem_list = [i for i in words_in_vocab if "bn:" in i]
lem_ = dict([i.split('_bn:') for i in lem_list])

normal = {}
exist_as_lemma = {}
def get_word_embeddings(a):
  if a in list(lem_.keys()):
        i_lemma = lem_.get(a)
        build_i = a + '_bn:' + i_lemma
        exist_as_lemma[a] = build_i
        i_vectors = loaded_model.wv.word_vec(build_i)
        return i_vectors

df['new_word_a'] = df.word_a.apply(get_word_embeddings)
df['new_word_b'] = df.word_b.apply(get_word_embeddings)

# replace word not found with zero and verify no row is nu
df = df.fillna(0)
df[(df['new_word_a'].notnull()) & (df['new_word_b'].notnull())]

# converting  words to a list
word_cos_list = df.gold.tolist()
word_a_list = df.new_word_a.tolist()
word_b_list = df.new_word_b.tolist()

def calc_cosine_similarity(array_A, array_B):
    cos_sim = []
    for k, v in enumerate(array_A):
       if type(v) == int or type(array_B[k]) == 0:
         cos_sim.append(-1)
       else:
         sim = 1 - cosine(v, array_B[k])
         max_ = max(-1, sim)
         cos_sim.append(max_)
    return cos_sim

embeddings_cosine_list = calc_cosine_similarity(word_a_list,word_b_list)

assert len(word_cos_list) == len(embeddings_cosine_list)

spearmans_rank_correlation = spearmanr(word_cos_list, embeddings_cosine_list)[0]

print("The Spearman correlation is: ",spearmans_rank_correlation)
