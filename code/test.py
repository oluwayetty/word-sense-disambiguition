word = []
embeddings = []
array = []

with open('/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework2/test.vec') as file:
    next(file)
    file = file.readlines()
    for x in file:
        word.append(x.split('_bn:')[0])
        e = x.strip().split('_bn:')[1].split(' ')
        embeddings.append(e[1:len(e)])
        array.append([word,embeddings])

#get unique list
print(array)
