import string
import pandas as pd
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
from config import XML_FILEPATH, DATA_FOLDER
from lxml import etree as ET

class Error(Exception):
    """Base class for other exceptions"""
    pass

class EmptyTagError(Error):
    """Raised when the text lang tag is empty"""
    print("Empty tag detected")

def parse_xml(filepath):
    """
    Takes the xml filepath and returns four multidimensional
    of anchors, lemmas, sentences, & babelnetID
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

def get_wanted_IDs(babelnetIDs):
    flat_list = [item for sublist in babelnetIDs for item in sublist]
    set_flat_list = list(set(flat_list))
    mapping_IDS =[]

    with open('resources/bn2wn_mapping.txt', 'r') as f:
        lines = f.readlines()
        mapping_IDS = [line.split('\t')[0] for line in lines]
    return mapping_IDS

def join_lemma_to_IDs(lemmas, babelnetIDs):
    """
    This joins each lemma to its corresponding babelnetID
    e.g "cathedral" + "bn:00016759n" ==> cathedral_bn:00016759n
    """
    lemma_IDs = []
    print("Combining lemmas to their babelnetIDs")
    for x in range(0, len(lemmas)):
        lemma_ids = [m.lower() + '_' + str(n) for m, n in zip(lemmas[x], babelnetIDs[x])]
        lemma_IDs.append(lemma_ids)
    return lemma_IDs

def remove_unwanted_lemmas(check, IDs):
    print(check.split("_bn"))
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
    all_lemmas = []
    for element in array:
        final_lemmas = [remove_unwanted_lemmas(i, IDs) for i in element]
        all_lemmas.append(final_lemmas)
    return all_lemmas


def replace_anchors_with_lemmas(lemma_ids, anchors, sentences):
  all_sentences = []
  print("Replacing anchors with lemma_IDs")
  for i, sentence in enumerate(sentences):
      lemmas = dict(zip(anchors[i], lemma_ids[i]))
      all_sentences.append([" ".join(map(lambda x:lemmas.get(x, x), sentence.split()))])
  return all_sentences


def remove_context_words(filepath):
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
    for elem in replaces :
        if elem in main :
            main = main.replace(elem, new)
    return main

def remove_stop_words(data,outfile):
    print("Removing stop words")
    all_clean = []
    for each in data:
        clean = [word for word in each.split() if word not in stoplist]
        clean = " ".join(clean)
        all_clean.append(clean)
    with open(outfile, 'w') as f:
        f.write('sentence'+'\n')
        for item in all_clean:
            f.write("%s\n" % item)


def main():
    cst_punct = list(string.punctuation.replace(':', '').replace('_', ''))

    all_english_texts, all_babelnetIDs, all_lemmas, all_anchors = parse_xml(XML_FILEPATH)
    all_lemmaIDs = join_lemma_to_IDs(all_lemmas,all_babelnetIDs)

    length = len(all_english_texts)
    assert all(len(lst) == length for lst in [all_babelnetIDs, all_lemmas, all_anchors])

    all_lemmaIDs = remove_id_from_unwanted(all_lemmaIDs,all_babelnetIDs)
    training = replace_anchors_with_lemmas(all_lemmaIDs,all_anchors,all_english_texts)
    flat_list = [item for sublist in training for item in sublist]
    training_data = [replaceMultiple(i, cst_punct, '') for i in flat_list]
    remove_stop_words(training_data, DATA_FOLDER+ '/training.txt')


if __name__ == '__main__':
    main()
    print('Final data was written to '+ DATA_FOLDER)
