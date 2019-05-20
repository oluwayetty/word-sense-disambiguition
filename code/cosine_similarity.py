from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy import spatial
from numpy import dot
from numpy.linalg import norm


dataSetI = [3, 45, 7, 2]
dataSetII = [2, 54, 13, 15]

spatial = 1 - spatial.distance.cosine(dataSetI, dataSetII)
cos_sim = cos([dataSetI], [dataSetII])
dot_sim = dot(dataSetI, dataSetII)/(norm(dataSetI)*norm(dataSetII))
#
# print(spatial,cos_sim,dot_sim)
unique = []
array = []
word = []
embeddings = []
with open('/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework2/test.txt') as file:
    next(file)
    file = file.readlines()
    for x in file:
        # word, embeddings = x.split('_bn:'
        word.append(x.split('_bn:')[0])
        embeddings.append(x.split('_bn:')[1])
        array.append([word,embeddings])

# print(word,embeddings)
for i in word:
    if i not in unique:
        unique.append(i)

count = [None]*len(unique)
i = 0
for x in range(0,len(unique)):
    Y = []
    for y in range(0,len(word)):
        if unique[x] == word[y]:
            Y.append(embeddings[y].strip())
    count[i] = Y
    i = i + 1

s1,s2,s3,s4 = count[0],count[1],count[2],count[3]
# print(s2)
for i in s2[0].split():
    print(i)
