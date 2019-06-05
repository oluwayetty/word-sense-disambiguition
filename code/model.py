from config import DATA_FILE, EMBEDDINGS, GZ_DATA
from utils import *
from pre_process import remove_context_words
from gensim.models import Word2Vec
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--format", default="true",
    help="If true, it uses the gz version of the data which boosts the \
    performance in training time. If false - it uses the txt version of the data")
    return parser.parse_args()

def train_cbow_model(training_data):
    print("Training started.....")
    model = Word2Vec(sentences = training_data, min_count=1, ns_exponent=10e-3, window=4, sg=0, hs=1, size=100,iter=10)
    model.wv.save_word2vec_format(EMBEDDINGS)
    print("Training done")


def train_cbow_model_line_sentence(training_data):
    print("Training started.....")
    model = Word2Vec(corpus_file = training_data, min_count=1, ns_exponent=10e-3, window=5, sg=0, hs=1,size=100,iter=10)
    model.wv.save_word2vec_format(EMBEDDINGS)
    print("Training done")

def main():
    if parse_args().format == 'true':
        training_data = return_gz_format(DATA_FILE)
        train_cbow_model_line_sentence(training_data)
        print("Creating embeddings")
        remove_context_words(EMBEDDINGS)
    else:
        training_data = return_data_for_cbow(DATA_FILE)
        train_cbow_model(training_data)
        print("Creating embeddings")
        remove_context_words(EMBEDDINGS)

    print("Embeddings saved, check "+ EMBEDDINGS + " for the embeddings")

if __name__ == '__main__':
    main()
