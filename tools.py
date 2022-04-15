from typing import Dict, List, Tuple
from data_modules.input_example import Relation


def compute_feature_for_augmented_seq(relation: Relation, 
                                    doc_sentences: List[str],
                                    selected_idx: Dict[int, List[int]]):
    augmented_ids = [relation.head.sid, relation.tail.sid] + selected_idx[relation.head.sid] + selected_idx[relation.tail.sid]
    augmented_ids = sorted(list(set(augmented_ids)))
    augmented_sequence = []
    sent_span = {}
    start = 0
    for i, sidx in enumerate(augmented_ids):
        sent = doc_sentences[sidx]
        augmented_sequence.append(sent)
        end = start + len(sent)
        sent_span[sidx] = (i, start, end, len(sent))
    
    augmented_sequence = ' '.join(augmented_sequence)
    head_span_in_augm = get_new_pos((relation.head.start_in_sent, relation.head.end_in_sent), relation.head.sid, sent_span)
    tail_span_in_augm = get_new_pos((relation.tail.start_in_sent, relation.tail.end_in_sent), relation.tail.sid, sent_span)

    return augmented_sequence, head_span_in_augm, tail_span_in_augm


def get_new_pos(span: Tuple[int, int], sid: int, sent_span: Dict[int, Tuple[int, int, int, int]]):
    new_start, new_end = span
    for id in sent_span.keys():
        if sid > id:
            new_start += sent_span[id][3]
            new_end += sent_span[id][3]
    return (new_start, new_end)


def token_id_lookup(token_span_SENT, start_char, end_char, start_id=0):
    idx = []
    for index, token_span in enumerate(token_span_SENT, start=start_id):
        if start_char in range(token_span[0], token_span[1]) or end_char in range(token_span[0], token_span[1]):
            idx.append(index)
    return idx
    

def compute_f1(predicts: List[str], golds: List[str]):
    pass



