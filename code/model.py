from config import DATA_FILE, EMBEDDINGS, GZ_DATA
from utils import return_data_for_cbow, return_line_sentence_data_format, return_gz_format
from gensim.models import Word2Vec


def train_cbow_model(training_data):
    print("Training started.....")
    model = Word2Vec(sentences = training_data, min_count=1, ns_exponent=10e-3, window=4, sg=0, size=1)
    model.wv.save_word2vec_format(EMBEDDINGS)


def train_cbow_model_line_sentence(training_data):
    print("Training started.....")
    model = Word2Vec(corpus_file = training_data, min_count=1, ns_exponent=10e-3, window=4, sg=0, size=1)
    model.wv.save_word2vec_format(EMBEDDINGS)


def main():
    training_data = return_data_for_cbow(DATA_FILE)
    train_cbow_model(training_data)

    # training_data = return_gz_format(DATA_FILE)
    # train_cbow_model_line_sentence(training_data)

    print("Training done, check "+ EMBEDDINGS + " for the embeddings")

if __name__ == '__main__':
    main()
