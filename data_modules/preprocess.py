from collections import defaultdict
import json
import os
from sklearn.model_selection import train_test_split
import tqdm
from data_modules.datapoint_formats import get_datapoint
from data_modules.reader import tsvx_reader
import random
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, datapoint, intra=True, inter=False):
        self.dataset = dataset
        self.intra = intra
        self.inter = inter
        self.datapoint = datapoint
        self.register_reader(self.dataset)

    def register_reader(self, dataset):
        if self.dataset == 'HiEve':
            self.reader = tsvx_reader
        # elif dataset == 'ESL':
        #     self.reader = cat_xml_reader
        # elif dataset == 'Causal-TB':
        #     self.reader = ctb_cat_reader
        else:
            raise ValueError("We have not supported this dataset {} yet!".format(self.dataset))

    def load_dataset(self, dir_name):
        corpus = []
        if self.dataset == 'ESL':
            topic_folders = [t for t in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, t))]
            for topic in tqdm.tqdm(topic_folders):
                topic_folder = os.path.join(dir_name, topic)
                onlyfiles = [f for f in os.listdir(topic_folder) if os.path.isfile(os.path.join(topic_folder, f))]
                for file_name in onlyfiles:
                    file_name = os.path.join(topic, file_name)
                    if file_name.endswith('.xml'):
                        my_dict = self.reader(dir_name, file_name, inter=self.inter, intra=self.intra)
                        if my_dict != None:
                            corpus.append(my_dict)
        else:
            onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            i = 0
            for file_name in tqdm.tqdm(onlyfiles):
                # if i == 11:
                #     break
                # i = i + 1
                my_dict = self.reader(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        
        return corpus
    
    def process_and_save(self, corpus, save_path=None):
        if type(corpus) == list:
            processed_corpus = []
            for my_dict in tqdm.tqdm(corpus):
                len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                if len_doc < 400:
                    doc_info = True
                else:
                    doc_info = False
                processed_corpus.extend(get_datapoint(self.datapoint, my_dict, doc_info))
            if save_path != None:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_corpus, f, indent=6)
        else:
            processed_corpus = defaultdict(list)
            for key, topic in corpus.items():
                for my_dict in tqdm.tqdm(topic):
                    len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                    if len_doc < 400:
                        doc_info = True
                    else:
                        doc_info = False
                    processed_corpus[key].extend(get_datapoint(self.datapoint, my_dict, doc_info))
            if save_path != None:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_corpus, f, indent=6)

        return processed_corpus