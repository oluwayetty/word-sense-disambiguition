import pandas as pd
from utils, import COMBINED_TAB, EMBEDDINGS
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

cos = pd.read_csv(COMBINED_TAB, sep='\t')
loaded_model = KeyedVectors.load_word2vec_format(EMBEDDINGS)
cos.columns = ['word_a', 'word_b', 'cos']
words_in_vocab = model.wv.vocab.keys()


lem_list = [i for i in words_in_vocab if "bn:000" in i]
lem_ = dict([i.split('_') for i in lem_list])
words_in_vocab = list(words_in_vocab)

def run_on(a):
  if a in words_in_vocab:
      i_vectors = loaded_model.wv.word_vec(a)
      return i_vectors
  elif a in list(lem_.keys()):
        i_lemma = lem_.get(a)
        build_i = a + '_' + i_lemma
        i_vectors = model.wv.word_vec(build_i)
        return i_vectors
  else:
        return "oov"

cos['new_word_a'] = cos.word_a.apply(run_on)
