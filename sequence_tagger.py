# PhraseMatcher.py
# import necessary modules

from __future__ import unicode_literals, print_function


from pathlib import Path
from spacy.util import minibatch, compounding
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy import displacy
from collections import Counter
from spacy.matcher import PhraseMatcher #import PhraseMatcher class
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from time import sleep
from progressbar import progressbar
from spacy.tokens import Span
from spacy.util import minibatch, compounding
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import random
import plac
import warnings
import pandas as pd
import numpy as np
import xlrd
import spacy
import en_core_web_sm
import json 
import re
import glob

plt.rcParams["figure.figsize"] = (15,10)

nltk.download('punkt')

# Language class with the English model 'en_core_web_sm' is loaded
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 7000000






def train_spacy(data, n_iter = 10, load = False, train = True, output_dir = "output_dir"):
    
    if train:
        TRAIN_DATA = data
        # load space model
        if load:
            nlp = spacy.load("en_core_web_sm")
        else:
            nlp = spacy.blank('en')



        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        else:
            ner = nlp.get_pipe("ner")

        # add labels
        ner.add_label('AstroTerm')
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

        # Resume training
        #optimizer = nlp.resume_training()
        #move_names = list(ner.move_names)

        nlp.begin_training()

        # Begin training by disabling other pipeline components
        with nlp.disable_pipes(*other_pipes):
            # show warnings for misaligned entity spans once
            optimizer = nlp.begin_training()

            sizes = compounding(1.0, 5.0, 1.001)
            # Training for n_iter iterations     
            for itn in progressbar(range(n_iter)):
                sleep(0.02)
            # shuffle examples before training
                random.shuffle(TRAIN_DATA)
                # batch up the examples using spaCy's minibatch
                batches = minibatch(TRAIN_DATA, size=sizes)
                # dictionary to store losses
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    # Calling update() over the iteration
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.25, losses=losses)
                    #print("Losses", losses)
        print("Saving the trained model...")            
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)
            
        
    else:
        return 0

    
    
def extract_entities(doc):
    dict_new_ents = {}
    list_new_ents = []
    for ent in doc.ents:
        # Only check
        if ent.label_ == "AstroTerm":
            
            
            list_new_ents.append((ent.start_char, ent.end_char, ent.label_))      
    
    dict_new_ents['entities'] = list_new_ents
            
    return (doc.text,dict_new_ents )











print("Load term's list")
terms_corpus = pd.read_excel('astronomy.xls')

# the list containing the pharses to be matched
terminology_list = []
for term in terms_corpus['key']:
    terminology_list.append(term[term.find(':')+2:])
    
    
print("Read the corpus files...")    
read_files = glob.glob("corpus/Astromony_*.txt")
with open("corpus/result.txt", "wb") as outfile:
    for f in progressbar(read_files):
        with open(f, "rb") as infile:
            outfile.write(infile.read())

# the input text string is converted to a Document object

file = open('corpus/result.txt')
text = file.read()


nlp_rule_based = English()
ruler = EntityRuler(nlp_rule_based)

# create patterns
patterns = []

for term in terminology_list:
    dct = {}
    temp = term.split()
    if len(temp) == 1:
        dct["label"] = "AstroTerm"
        dct["pattern"] = temp[0]
        patterns.append(dct)
    else:
        lst = []
        for item in temp:
            dct_temp = {}
            dct_temp["lower"] = item
            
            lst.append(dct_temp)
            
        dct["label"] = "AstroTerm"
        dct["pattern"] = lst
        patterns.append(dct)
        
        
# add patterns and pipe
ruler.add_patterns(patterns)
nlp_rule_based.add_pipe(ruler)



# generate annotaeted data
print("Generate annotated data...")
train_data = []
for doc in progressbar(nltk.tokenize.sent_tokenize(text)):
    doc = nlp_rule_based(doc)
    train_data.append(extract_entities(doc))
    



# to train the model set 'train' to true 
print("Model training...")
train_spacy(train_data, 10, train = True)



# save the model
# output_dir = "output_dir"
# if output_dir is not None:
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir()
#     nlp.to_disk(output_dir)
#     print("Saved model to", output_dir)