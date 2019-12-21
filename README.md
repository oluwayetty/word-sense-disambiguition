![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png)

## Problem 
One of the limitation of word embeddings is the meaning conflation deficiency, which arises from representing a word with all its possible meanings as a single vector. This project is one step to address this; representation of meanings. Given a corpus, we will train a Word2Vec model to obtain its lemma_synsetID given a context of words.

## Dataset Description 
I made use of the compulsory [Eurosense corpus](http://lcl.uniroma1.it/eurosense/) and [Trainomatic dataset](https://www.aclweb.org/anthology/D17-1008/) using just the english sentences, there were many inconsistencies found in the corpus which I cleaned up. For nomenclature, babelnetIDs will be represented as bnIDs in this section.

## Model Evaluation using similarity score
Word and sense embeddings can be evaluated using word similarities algorithms. I explored the cosine similarity measure according to the slide and also the gensim similarity function, both gave same results. To penalize words not found in embeddings vocabulary against [the WordSimilarity-353 [8.4](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) annotated score, I used a cosine similarity score of -1 instead of skipping them. Eventually, each cosine score of senses found from my embeddings and the annotated scores(gold) were eventually used to compute my spearman rank score.

## Repository skeleton
```
- code               # this folder contains all the code related to this project
- resources
  |__ embeddings.vec # this file contains the sense embeddings from our trained model
- README.md          # this file
- Homework_2_nlp.pdf # the slides for the course homework instruction
- report.pdf         # my report which basically analyzed the code and the results obtained.
```
