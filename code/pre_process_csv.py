import csv
import string
import pandas as pd
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
                                lemma.append(annotation.attrib['lemma'])
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

def join_lemma_to_IDs(lemmas, babelnetIDs):
    """
    This joins each lemma to its corresponding babelnetID
    e.g "cathedral" + "bn:00016759n" ==> cathedral_bn:00016759n
    """
    lemma_IDs = []
    print("Combining lemmas to their babelnetIDs")
    for x in range(0, len(lemmas)):
        lemma_ids = [m.replace(' ', '') + '_' + str(n) for m, n in zip(lemmas[x], babelnetIDs[x])]
        lemma_IDs.append(lemma_ids)
    return lemma_IDs


def replace_anchors_with_lemmas(lemma_ids, anchors, sentences):
  all_sentences = []
  print("Replacing anchors with lemma_IDs")
  for i, sentence in enumerate(sentences):
      lemmas = dict(zip(anchors[i], lemma_ids[i]))
      all_sentences.append([" ".join(map(lambda x:lemmas.get(x, x), sentence.split()))])
  return all_sentences


def replaceMultiple(main, replaces, new):
    for elem in replaces :
        if elem in main :
            main = main.replace(elem, new)
    return main

def write_per_row(filename, list_, header_name):
    with open(filename, mode='w') as csv_file:
        fieldnames = [header_name]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in list_:
            element = element.replace('.', '').replace('...', '')
            writer.writerow({header_name: element})


def main():
    cst_punct = list(string.punctuation.replace(':', '').replace('_', ''))

    all_english_texts, all_babelnetIDs, all_lemmas, all_anchors = parse_xml(XML_FILEPATH)
    all_lemmaIDs = join_lemma_to_IDs(all_lemmas,all_babelnetIDs)

    length = len(all_english_texts)
    assert all(len(lst) == length for lst in [all_babelnetIDs, all_lemmas, all_anchors])

    training = replace_anchors_with_lemmas(all_lemmaIDs,all_anchors,all_english_texts)
    flat_list = [item for sublist in training for item in sublist]
    training_data = [replaceMultiple(i, cst_punct, '') for i in flat_list]
    write_per_row(DATA_FOLDER+ '/training.csv', training_data, 'sentence')


if __name__ == '__main__':
    main()
    print('Final data was written to '+ DATA_FOLDER)
