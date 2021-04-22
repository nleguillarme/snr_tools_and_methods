import os
import shutil
from math import ceil
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
from pandas.errors import EmptyDataError
import sys
import csv
import re
import codecs

# Utility function : copy and apply a function to the train, test, and dev subsets

def process_dataset(input_dir, output_dir, function, train="train", test="test", dev="dev"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    if dev:
        shutil.copytree(os.path.join(input_dir, dev), os.path.join(output_dir, 'dev'))
        function(os.path.join(output_dir, 'dev'))
    if train:
        shutil.copytree(os.path.join(input_dir, train), os.path.join(output_dir, 'train'))
        function(os.path.join(output_dir, 'train'))
    if test:
        shutil.copytree(os.path.join(input_dir, test), os.path.join(output_dir, 'test'))
        function(os.path.join(output_dir, 'test'))

    if train and dev:
        shutil.copytree(os.path.join(output_dir, train), os.path.join(output_dir, 'train_dev'))
        for item in os.listdir(os.path.join(output_dir, 'dev')):
            s = os.path.join(output_dir, 'dev', item)
            d = os.path.join(output_dir, 'train_dev', item)
            shutil.copy2(s, d)
                        
# Read annotations from CSV file in Standoff format

def get_ann(ann_filename):
    def split_offsets(row):
	    items = row["offsets"].split(" ")
	    row["type"] = items[0]    
	    row["start"] = int(items[1])
	    row["end"] = int(items[-1])
	    return row
	    
    try:
        doc_ann = pd.read_csv(ann_filename, sep = "\t", header=None)
    except EmptyDataError:
        return pd.DataFrame()
    else:
        if not doc_ann.empty:
            doc_ann.columns = ["index", "offsets", "text"]
            doc_ann = doc_ann.set_index("index")
            doc_ann = doc_ann.dropna()
            doc_ann = doc_ann.apply(split_offsets, axis=1)
            doc_ann = doc_ann.drop(["offsets"], axis=1)
            doc_ann = doc_ann[(doc_ann['type'] == "Taxon") | (doc_ann['type'] == "Microorganism") | (doc_ann['type'] == "LIVB")]
            doc_ann = doc_ann.replace("Taxon", "LIVB")
            doc_ann = doc_ann.replace("Microorganism", "LIVB")
    return doc_ann

# Remove \r\n newlines

def remove_newlines(txtfile):
    with open(txtfile, "r",  newline="") as f:
        data = f.read().replace("\r\n", "\n")
    with open(txtfile, "w") as f:
        f.write(data)
    return data

# Replace non-ascii characters in corpus
# See : https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space

def utf8_to_ascii(input_dir):
    for document in glob(os.path.join(input_dir, "*.txt")):
        with open(document, "r") as f:
            data = f.read()
            asciidata = re.sub(r'[^\x00-\x7F]+',' ', data)
        with open(document, "w") as f:
            f.write(asciidata)
    
# Trim whitespaces from the start and the end of entities

def trim_whitespaces(start, end, text):
    invalid_span_tokens = re.compile(r'\s')
    valid_start = start
    valid_end = end
    ent_start = 0
    ent_end = len(text)
    while ent_start < len(text) and invalid_span_tokens.match(
        text[ent_start]):
        ent_start += 1
    while ent_end > 1 and invalid_span_tokens.match(
        text[ent_end - 1]):
        ent_end -= 1
    return valid_start+ent_start, valid_start+ent_end
    
# From a list of intervals (boundaries), replace a set of overlapping entities by their union

def get_max_non_overlapping(intervals):
    i = 0
    while i < len(intervals)-1:
        extended = None
        j = i+1
        while j < len(intervals):
            extended = overlap(intervals[i], intervals[j])
            if extended:
                break
            else:
                j += 1
        if extended:
            intervals[i] = extended
            intervals.pop(j)
        else:
            i += 1
    return intervals
            
# Return True if the boundaries of two entities overlap

def overlap(ival, jval):
    if jval[0] <= ival[0] and ival[1] <= jval[1]:
        return jval
    if ival[0] <= jval[0] and jval[1] <= ival[1]:
        return ival
    if ival[0] < jval[0] and jval[0] < ival[1] and ival[1] < jval[1]:
        return (ival[0], jval[1])
    if jval[0] < ival[0] and ival[0] < jval[1] and jval[1] < ival[1]:
        return (jval[0], ival[1])
    return None
    
def clean_corpus(input_dir):
    data = []
    for document in glob(os.path.join(input_dir, "*.txt")):

        filename = os.path.basename(document)
        ann_filename = document[:-3]+"ann"
        doc_ann = get_ann(ann_filename)
        
        text = remove_newlines(document)
        
        offsets = [(row['start'], row['end']) for _, row in doc_ann.iterrows()]
        offsets = get_max_non_overlapping(offsets)
        
        entities = []
        for offset in offsets:
            start, end = trim_whitespaces(offset[0], offset[1], text[offset[0]:offset[1]])
            entities += [{"offsets":" ".join(["LIVB", str(start), str(end)]), "text":text[start:end].replace("\n"," ")}] 
                
        df = pd.DataFrame(entities)
        df = df.rename('T{}'.format)

        df.to_csv(ann_filename, sep = "\t", header=None, quoting=csv.QUOTE_NONE)


def split_brat_standoff(corpra_dir, train_size, test_size, valid_size, random_seed=42):
    """
    Randomly splits the corpus into train, test and validation sets.
    Args:
        corpus_dir: path to corpus
        train_size: float, train set parition size
        test_size: float, test set parition size
        valid_size: float, validation set parition size
        random_seed: optional, seed for random parition
    """
    assert (
        train_size < 1.0 and train_size > 0.0
    ), "TRAIN_SIZE must be between 0.0 and 1.0"
    assert test_size < 1.0 and test_size > 0.0, "TEST_SIZE must be between 0.0 and 1.0"
    assert (
        valid_size < 1.0 and valid_size > 0.0
    ), "VALID_SIZE must be between 0.0 and 1.0"
    assert (
        ceil(train_size + test_size + valid_size) == 1
    ), "TRAIN_SIZE, TEST_SIZE, and VALID_SIZE must sum to 1"

    print("[INFO] Moving to directory: {}".format(corpra_dir))
    with cd(corpra_dir):
        print("[INFO] Getting all filenames in dataset...", end=" ")
        # accumulators
        text_filenames = []
        ann_filenames = []

        # get filenames
        for file in os.listdir():
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and not filename.startswith("."):
                text_filenames.append(filename)
            elif filename.endswith(".ann") and not filename.startswith("."):
                ann_filenames.append(filename)

        assert len(text_filenames) == len(
            ann_filenames
        ), """Must be equal
            number of .txt and .ann files in corpus_dir"""

        # hackish way of making sure .txt and .ann files line up across the two lists
        text_filenames.sort()
        ann_filenames.sort()

        print("DONE")
        print(
            "[INFO] Splitting corpus into {}% train, {}% test, {}% valid...".format(
                train_size * 100, test_size * 100, valid_size * 100
            ),
            end=" ",
        )
        # split into train and all other, then split all other into test and valid
        X_train, X_test_and_valid = train_test_split(
            text_filenames, train_size=train_size, random_state=random_seed
        )
        y_train, y_test_and_valid = train_test_split(
            ann_filenames, train_size=train_size, random_state=random_seed
        )
        X_test, X_valid = train_test_split(
            X_test_and_valid,
            train_size=test_size / (1 - train_size),
            random_state=random_seed,
        )
        y_test, y_valid = train_test_split(
            y_test_and_valid,
            train_size=test_size / (1 - train_size),
            random_state=random_seed,
        )

        # leads to less for loops
        X_train.extend(y_train)
        X_test.extend(y_test)
        X_valid.extend(y_valid)

        print("Done.")
        print(
            "[INFO] Creating train/test/valid directories at {} if they do not already exist...".format(
                corpra_dir
            )
        )
        # if they do not already exist
        os.makedirs("train", exist_ok=True)
        os.makedirs("test", exist_ok=True)
        os.makedirs("valid", exist_ok=True)

        for x in X_train:
            shutil.move(x, "train/" + x)
        for x in X_test:
            shutil.move(x, "test/" + x)
        for x in X_valid:
            shutil.move(x, "valid/" + x)


## UTILITY METHODS/CLASSES


class cd:
    """Context manager for changing the current working directory."""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
