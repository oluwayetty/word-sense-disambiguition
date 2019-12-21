## Problem One of the limitation of word embeddings is the meaning conflation deficiency, which arises from representing a word with all its possible meanings as a single vector. This project is one step to address this; representation of meanings. Given a corpus, we will train a Word2Vec model to obtain its lemma_synsetID given a context of words.

## Dataset Description I made use of the compulsory Eurosense corpus and Trainomatic dataset using just the english sentences, there were many inconsistencies found in the corpus which I cleaned up. For nomenclature, babelnetIDs will be represented as bnIDs in this section

## Repository skeleton
```
- code               # this folder contains all the code related to this project
- resources
  |__ embeddings.vec # this file contains the sense embeddings from our trained model
- README.md          # this file
- Homework_2_nlp.pdf # the slides for the course homework instruction
- report.pdf         # my report which basically analyzed the code and the results obtained.
```
## Instructions for datasets used 
- Download EuroSense from http://lcl.uniroma1.it/eurosense/ . 
- Download WordSimilarity-353 from http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/ .
