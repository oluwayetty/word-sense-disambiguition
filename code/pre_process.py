import csv
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

    for x in range(0, len(lemmas)):
        lemma_ids = [m.replace(' ', '') + '_' + str(n) for m, n in zip(lemmas[x], babelnetIDs[x])]
        lemma_IDs.append(lemma_ids)
    return lemma_IDs

def write_per_row(filename, list_, header_name):
    """
    create a unique csv file for each arrays from parse_xml,
    for easy manipulation before concatenation
    """
    with open(filename, mode='w') as csv_file:
        fieldnames = [header_name]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for element in list_:
            writer.writerow({header_name: element})

def write_lists_into_array():
    filenames = []
    all_sentences, all_babelnetIDs, all_lemmas, all_anchors = parse_xml(XML_FILEPATH)
    all_lemma_IDs = join_lemma_to_IDs(all_lemmas,all_babelnetIDs)
    # Verify data is properly parsed by validating all lengths of lists from parse_xml() is same
    length = len(all_sentences)
    assert all(len(lst) == length for lst in [all_babelnetIDs, all_lemmas, all_anchors])
    print("Writing each lists into a new csv.........")
    array_of_contents = [
        [all_sentences, 'sentence'],
        [all_anchors,'anchor'],
        [all_lemmas, 'lemma'],
        [all_babelnetIDs, 'babel_netID'],
        [all_lemma_IDs, 'lemma_IDs']]

    for value in array_of_contents:
        header_name = value[1]
        filename = DATA_FOLDER + '/' + header_name + '.csv'
        write_per_row(filename, value[0] , header_name)
        filenames.append(filename)
    return filenames

def concatenate():
    pieces = []
    all_filenames = write_lists_into_array()

    print("Combining each csv's file one new csv........")

    for file in all_filenames:
        s = pd.read_csv(file) # directory
        pieces.append(s)

    data = pd.concat(pieces, axis=1) # this will yield multiple columns
    filepath = DATA_FOLDER + '/combined.csv'
    data.to_csv(filepath)

if __name__ == "__main__":
    concatenate()
    print('Final data was written to combined.csv')
