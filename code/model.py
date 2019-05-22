import string
# import swifter
from config import LEMMAIDS_DATA, FINAL_DATA, EMBEDDINGS
from utils import final_clean_up, build_lemmas_into_sentence, apply_function
from gensim.models import Word2Vec
# from swifter import swiftapply

def replaceMultiple(main, replaces, new):
    for elem in replaces :
        if elem in main :
            main = main.replace(elem, new)
    return main

def prepare_data_for_cbow(data):
    cst_punct = list(string.punctuation.replace(':', '').replace('_', ''))
    # import ipdb; ipdb.set_trace()
    training = data.sentences.tolist()
    training = [i for i in training if i]
    training_data = [replaceMultiple(i, cst_punct, '') for i in training]
    return training_data

def train_cbow_model(training_data):
    # training_data =
    # print(training_data)

    splitted = [i.split() for i in training_data]
    model = Word2Vec(splitted, min_count=1, ns_exponent=10e-3, window=4, sg=0, size=1)
    model.wv.save_word2vec_format(EMBEDDINGS)

def main():
    training_data = prepare_data_for_cbow(apply_function())
    train_cbow_model(training_data)

if __name__ == '__main__':
    main()
