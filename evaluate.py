#!/usr/bin/env python3
import argparse
import os
import pickle
import spacy
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from HmmTagger import HMMTagger
from read_tags import *


def main(args): 
    nlp = spacy.load("en_core_web_sm")
   
    tagger = pickle.load(args.hmm)
    
    total_right = 0
    total_size = 0 

    right = []
    size = []

    for fname in get_files(args.dir):
        with open(fname) as fp:
                for words, tags in parse_file(fp, do_universal=args.universal): 
                    doc = nlp.tokenizer.tokens_from_list(list(words))
                    tagger(doc)

                    if args.universal and not(tagger.do_universal): 
                        for token in doc:
                            token.tag_ = ptb_to_universal[token.tag_]

                    right.append(sum([spacy_token.tag_ == ref_tag 
                                for spacy_token, ref_tag in zip(doc, tags)]))
                    total_right += sum([spacy_token.tag_ == ref_tag 
                                for spacy_token, ref_tag in zip(doc, tags)])
                    
                    size.append(len(doc))
                    total_size += len(doc)
        
    print("Accuracy: {:.2%}".format(total_right/total_size))
    if args.output is not None:
        plt.scatter(size, right)
        plt.xlabel("Document Size")
        plt.ylabel("Percent Correct")
        plt.savefig(args.output)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='POS Tag, then evaluate')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True,
                        help="Read data to tag from DIR")
    parser.add_argument("--hmm", metavar="FILE", 
                        type=argparse.FileType('rb'), required=True,
                        help="Read hmm model from FILE")
    parser.add_argument("--universal", "-u", action="store_true", 
                        help="If set, uses universal tags instead of PTB tag set")
    parser.add_argument("--output", "-o", default=None)
    
    args = parser.parse_args()
    main(args)
 
