from collections import defaultdict
import datetime
import re
from typing import Dict, List, Tuple
import numpy as np
np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")


# Padding function
def padding(sent, max_sent_len = 194, pad_tok=0):
    one_list = [pad_tok] * max_sent_len # none id 
    one_list[0:len(sent)] = sent
    return one_list


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token)
                pointer = end
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span


def sent_id_lookup(my_dict, start_char, end_char = None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']


def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index


def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC


def id_lookup(span_SENT, start_char, end_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= end_char:
            return token_id
    raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {}".format(span_SENT, start_char))


def find_common_lowest_ancestor(tree, nodes):
    ancestor = nx.lowest_common_ancestor(tree, nodes[0], nodes[1])
    for node in nodes[2:]:
        ancestor = nx.lowest_common_ancestor(tree, ancestor, node)
    return ancestor


def get_dep_path(tree, nodes):
    # print(tree.edges)
    try:
        ancestor = nx.lowest_common_ancestor(tree, nodes[0], nodes[1])
        for node in nodes[2:]:
            ancestor = nx.lowest_common_ancestor(tree, ancestor, node)

        paths = []
        for node in nodes:
            paths.append(nx.shortest_path(tree, ancestor, node))
        return paths
    except:
        print(tree.edges)
        print(nx.find_cycle(tree, orientation="original"))
        return None


def mapping_subtok_id(subtoks: List[str], tokens: List[str]):
    text = ' '.join(tokens)
    token_spans = tokenized_to_origin_span(text, tokens)
    subtok_spans = tokenized_to_origin_span(text, subtoks)

    mapping_dict = defaultdict(list)
    for i, subtok_span in enumerate(subtok_spans, start=1):
        tok_id = token_id_lookup(token_spans, start_char=subtok_span[0], end_char=subtok_span[1])
        mapping_dict[tok_id].append(i)
    
    # mapping <unk> token:
    for key in range(len(tokens)):
        if mapping_dict.get(key) == None:
            print(f"haven't_mapping_tok: {tokens[key]}")
            mapping_dict[key] = random.randint(0, len(tokens)-1)
    
    # print(f"tokens: {tokens} \nsub_tokens: {subtoks} \nmapping_dict: {mapping_dict}")
    
    return dict(mapping_dict)
    

def get_new_poss(poss_in_sent: int, new_sid: int, sent_span: Dict[int, Tuple[int, int, int]]):
    new_poss = poss_in_sent
    for _new_sid, _, _, sent_len in sent_span.values():
        if _new_sid < new_sid:
            new_poss += sent_len
    return new_poss

