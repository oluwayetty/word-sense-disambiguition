"""
script for all visualizations shown in report,
was converted from jupyter notebook to a py file
"""


from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as plt2
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.io as pio
import random
import numpy as np
plt2.init_notebook_mode(connected=True)


EMBEDDINGS = 'resources/embeddings.vec'
loaded_model = KeyedVectors.load_word2vec_format(EMBEDDINGS, binary=False)
words_in_vocab = loaded_model[loaded_model.wv.vocab]


# Examples of senses similarities explored
loaded_model.similar_by_word('number_bn:00058290n')
loaded_model.similar_by_word('biomedicine_bn:00010558n')
loaded_model.similar_by_word('transplantation_bn:00059470n')
loaded_model.similar_by_word('anger_bn:00004086n')

"""
BEGINNING OF FIRST VISUALIZATION
"""
def display_closestwords_tsnescatterplot(model, sense_id, window_size, filename):

    vector_dim = model.vector_size
    arr = np.empty((0,window_size), dtype='f')
    word_labels = [sense_id]

    # get close words
    close_words = model.similar_by_word(sense_id,topn=15)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[sense_id]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.figure(figsize=(6, 5))

    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    if filename:
        plt.savefig(filename, format='png', dpi=100, bbox_inches='tight')
    plt.show()

vector_dim = loaded_model[loaded_model.wv.vocab].shape[1]
display_closestwords_tsnescatterplot(loaded_model,'mouse_bn:00056119n',vector_dim, 'plots/closestwords_tsnescatterplot.png')
"""
END OF FIRST VISUALIZATION
"""


"""
BEGINNING OF SECOND VISUALIZATION
"""
keys = ['software_bn:00021497n', 'computer_bn:00021464n', 'hardware_bn:00021480n', 'programmer_bn:00020358n']
embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in loaded_model.most_similar(word, topn=10):
        words.append(similar_word)
        embeddings.append(loaded_model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=100, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Nearest neighbour senses from embeddings', keys, embeddings_en_2d, word_clusters, 0.5,
                        'plots/similar_senses.png')

"""
END OF SECOND VISUALIZATION
"""

"""
BEGINNING OF THIRD VISUALIZATION
"""
words_wp = []
embeddings_wp = []
sampled_vocab = dict(random.sample(loaded_model.wv.vocab.items(), 3000))
for word in list(sampled_vocab):
    embeddings_wp.append(loaded_model.wv[word])
    words_wp.append(word)

tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)


def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c ='green', alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.savefig('plots/3D.png', format='png', dpi=100)
    plt.show()

tsne_plot_3d('3D t-SNE visualization', 'Sense embeddings', embeddings_wp_3d, a=0.1)
"""
END OF THIRD VISUALIZATION
"""


"""
BEGINNING OF FOURTH VISUALIZATION
"""
sampled_vocab = dict(random.sample(loaded_model.wv.vocab.items(), 500))
def get_coordinates(model, words):
    arr = np.empty((0,vector_dim), dtype='f')
    labels = []
    for wrd_score in words:
        try:
            wrd_vector = model[wrd_score]
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
            labels.append(wrd_score)
        except:
            pass
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    z_coords = Y[:, 2]
    return x_coords, y_coords, z_coords


words = list(sampled_vocab.keys())
x, y, z = get_coordinates(loaded_model, words)

plot = [go.Scatter3d(x = x,
                    y = y,
                    z = z,
                    mode = 'markers+text',
                    text = words,
                    textposition='bottom center',
                    hoverinfo = 'text',
                    marker=dict(size=15,opacity=0.8))]

layout = go.Layout(title='Scatter plot of Sense embeddings')
fig = go.Figure(data=plot, layout=layout)
plt2.iplot(fig)
pio.write_image(fig, 'plots/3dscatterplot.png')
"""
END OF FOURTH VISUALIZATION
"""
