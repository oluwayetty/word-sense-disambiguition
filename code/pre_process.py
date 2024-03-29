import string
import pandas as pd
import os
import re
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
from lxml import etree as ET
from argparse import ArgumentParser
from config import EUROSENSE_XML_FILEPATH, DATA_FOLDER, TRAINOMATIC_FOLDERPATH, MAPPING


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--nltk", default="false",
    help="If true, it runs the stopwords function, which removes stopwords from our data.\
          If false - it does not remove stop words function")
    parser.add_argument("--data", default="eurosense",
    help="If eurosense, it runs the parse functions for eurosense corpus,\
          If otherwise - it assumes the data is trainomatic")
    return parser.parse_args()

class Error(Exception):
    """Base class for other exceptions"""
    pass

class EmptyTagError(Error):
    """Raised when the text lang tag is empty"""
    print("Empty tag detected")

def parse_eurosense_xml(filepath):
    """
    :filepath - takes an xml filepath of our corpus
    :returns four multidimensional arrays of anchors,
    lemmas, sentences, & babelnetID found in the file.
    """
    english_texts, anchor_lists, lemma_lists, babelnetIDs = [], [], [], []

    xml_content = ET.iterparse(filepath, events=('end',), tag='sentence')
    print("Parsing xml file.......")

    for event, element in xml_content:

        anchor, lemma, babelnetID = [], [], []
        try:
            for elem in element.iter():
                if elem.tag == 'text' and elem.attrib['lang'] == 'en' and elem.text == None:
                    raise EmptyTagError()
                else:
                    if elem.tag == 'text' and elem.attrib['lang'] == 'en' and elem.text != None:
                        english_texts.append(elem.text)
                    if elem.tag == 'annotations':
                        for annotation in elem.iter():
                            if annotation.tag == 'annotation' and annotation.attrib['lang']  == 'en':
                                lemma.append("_".join(annotation.attrib['lemma'].split(' ')))
                                anchor.append(annotation.attrib['anchor'])
                                babelnetID.append(annotation.text.strip())
            babelnetIDs.append(babelnetID)
            lemma_lists.append(lemma)
            anchor_lists.append(anchor)

        except EmptyTagError:
            continue

        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    print("Parsing all done.......")
    return english_texts, babelnetIDs, lemma_lists, anchor_lists

def parse_trainomatic_xml(filepath):
    """
    :filepath - takes an xml filepath of our corpus
    :returns three multidimensional arrays of
    lemmas, sentences, & wordnetID found in the file.
    """

    xml_content = ET.iterparse(filepath, events=('end',), tag='lexelt')
    print("Parsing xml file.......")

    for event, element in xml_content:
        texts, lemmas, netIDs = [], [], []
        try:
            for elem in element.iter():
                if elem.tag == 'instance':
                    for ele in elem.iter():
                        if ele.tag == 'answer':
                            lemmas.append(ele.attrib['instance'].split('.')[0])
                            netIDs.append(ele.attrib['senseId'].split('.')[0])
                        if ele.tag == 'context':
                            texts.append(ele.text)
        except EmptyTagError:
            continue

    print("Parsing all done.......")

    length = len(lemmas)
    if all(len(lst) == length for lst in [netIDs, texts]):
        return texts, netIDs, lemmas
    else:
        print("Parse error")

def get_wanted_IDs(babelnetIDs):
    """
    :babelnetIDs - A list of all babelnetIDs from our Corpus
    :returns - A list that contains ONLY the babelnetIDs that are found in wordnet based on bn2wn_mapping.txt
    """
    flat_list = [item for sublist in babelnetIDs for item in sublist]
    set_flat_list = list(set(flat_list))
    mapping_IDS =[]

    with open(MAPPING, 'r') as f:
        lines = f.readlines()
        mapping_IDS = [line.split('\t')[0] for line in lines]
    return mapping_IDS

def join_eurosense_lemmas(lemmas, babelnetIDs):
    """
    :lemmas - arrays of lemmas e.g [ 'area', 'president']
    :babelnetIDs arrays of correspondent babelnetIDs to the lemmas
    This joins each lemma to its corresponding babelnetID
    :returns their concatenation e.g "cathedral" + "bn:00016759n" ==> cathedral_bn:00016759n
    """
    lemma_IDs = []
    print("Combining lemmas to their babelnetIDs")
    for x in range(0, len(lemmas)):
        lemma_ids = [m.lower() + '_' + str(n) for m, n in zip(lemmas[x], babelnetIDs[x])]
        lemma_IDs.append(lemma_ids)
    return lemma_IDs

def remove_unwanted_lemmas(check, IDs):
    """
    :check - lemma_ID e.g area_bn:03404559n and check if its synsetID i.e bn:03404559n
    is found in our wanted ID's list.
    :returns the lemma_ID itself if found, else returns just the lemma e.g area
    """
    word, id_ = check.split("_bn")
    id_ = 'bn' + id_
    mapping_IDS = get_wanted_IDs(IDs)
    if id_ in mapping_IDS:
        check
    else:
        check = check.replace(id_, '')
        check = check[:-1]
    return check


def remove_id_from_unwanted(array,IDs):
    """
    Uses the function(remove_unwanted_lemmas) above to map across our entire data
    """
    all_lemmas = []
    for element in array:
        final_lemmas = [remove_unwanted_lemmas(i, IDs) for i in element]
        all_lemmas.append(final_lemmas)
    return all_lemmas


def replace_anchors_with_lemmas(lemma_ids, anchors, sentences):
    """
    Goes through our english text, find the position of every anchor that is found in the sentence.
    if the anchor is found, replace each equivalent anchor with it's equivalent lemma_id
    e.g area in a sentence will become area_:bn:00016759n
    """
    all_sentences = []
    print("Replacing anchors with lemma_IDs")
    for i, sentence in enumerate(sentences):
      lemmas = dict(zip(anchors[i], lemma_ids[i]))
      all_sentences.append([" ".join(map(lambda x:lemmas.get(x, x), sentence.split()))])
    return all_sentences


def remove_context_words(filepath):
    """
    The task is to return just the sense embeddings, this function removes\
    all context words leaving only the sense embeddings in the vec file.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    with open(filepath, "w") as f:
        lines_embeddings = [i for i in lines if '_bn:' in i]
        embeddings_length = str(len(lines_embeddings))
        embeddings_dimsize = str(len(lines_embeddings[1].split(' ')) - 1)
        f.write('{} {}\n'.format(embeddings_length, embeddings_dimsize))
        for line in lines_embeddings:
            f.write(line)


def replaceMultiple(main, replaces, new):
    """
    This split each sentence into an array of each element,
    which helps to prepare our data for gensim Word2Vec expected input
    """
    for elem in replaces :
        if elem in main :
            main = main.replace(elem, new)
    return main


def remove_stop_words(data):
    print("Removing stop words using nltk toolkit")
    all_clean = []
    for each in data:
        clean = [word for word in each.split() if word not in stoplist]
        clean = " ".join(clean)
        all_clean.append(clean)
    return all_clean


def write_data_to_file(data):
    with open(DATA_FOLDER+ '/trainomatic_data_nltk.txt', 'w') as f:
        f.write('sentence'+'\n')
        for item in data:
            f.write("%s\n" % item)


def mapping_to_dict(filepath):
    """
    takes a filepath that contains wordnetID mapping to babelnetID
    returns a dictionary of the mapping
    """
    dict_ = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            new_line = line.split('\t')
            dict_[new_line[-1].replace('\n','')] = new_line[0]
        return dict_


def join_trainomatic_lemma(filename):
    """
    uses the parse_trainomatic_xml() function to process
    xml files from trainomatic
    """
    english_texts, all_netIDs, all_lemmas = parse_trainomatic_xml(filename)
    dict_map = mapping_to_dict(MAPPING)
    reformed_IDs = [i.split(':')[-1] for i in all_netIDs]
    for index, item in enumerate(reformed_IDs):
        if item in dict_map:
            reformed_IDs[index] = dict_map[item]
        else:
            reformed_IDs[index] = 'OVT'

    lemma_IDs = []
    for x in range(0, len(all_lemmas)):
        lemma_IDs = [m.lower() + '_' + str(n) for m, n in zip(all_lemmas, reformed_IDs)]
    lemma_IDs = [[x] for x in lemma_IDs]
    all_lemmas = [[x] for x in all_lemmas]

    length = len(english_texts)
    if all(len(lst) == length for lst in [lemma_IDs, all_lemmas]):
        return english_texts, lemma_IDs, all_lemmas
    else:
        print("Length do not match")


texts = []
def write_file(name):
    english_texts, lemma_ids, anchors = join_trainomatic_lemma(name)
    texts_en_copy = english_texts
    for j in range(len(anchors)):
        for a,b in zip(anchors[j], lemma_ids[j]):
            if a in texts_en_copy[j]:
                texts_en_copy[j] = re.sub(r'\b'+ re.escape(a)+r'\b', b, texts_en_copy[j])
        texts.append(texts_en_copy[j])
    return texts


def folders(path):
    all_texts = []
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            filename = os.path.join(directory, file)
            text = write_file(filename)
        all_texts.append(text)
    return all_texts


def trainomatic_nmain():
    training_data = folders(TRAINOMATIC_FOLDERPATH)
    if parse_args().nltk == 'true':
        all_clean = remove_stop_words(training_data[0])
        write_data_to_file(all_clean)
    else:
        write_data_to_file(training_data[0])


def eurosense_main():
    cst_punct = list(string.punctuation.replace(':', '').replace('_', ''))

    all_english_texts, all_babelnetIDs, all_lemmas, all_anchors = parse_eurosense_xml(EUROSENSE_XML_FILEPATH)
    all_lemmaIDs = join_eurosense_lemmas(all_lemmas,all_babelnetIDs)

    length = len(all_english_texts)
    assert all(len(lst) == length for lst in [all_babelnetIDs, all_lemmas, all_anchors])

    all_lemmaIDs = remove_id_from_unwanted(all_lemmaIDs,all_babelnetIDs)
    training = replace_anchors_with_lemmas(all_lemmaIDs,all_anchors,all_english_texts)
    flat_list = [item for sublist in training for item in sublist]
    training_data = [replaceMultiple(i, cst_punct, '') for i in flat_list]

    if parse_args().nltk == 'true':
        all_clean = remove_stop_words(training_data)
        write_data_to_file(all_clean)
    else:
        write_data_to_file(training_data)

def main():
    if parse_args().data == 'eurosense':
        eurosense_main()
    else:
        trainomatic_nmain()


if __name__ == '__main__':
    main()
    print('Final data was written to '+ DATA_FOLDER)
