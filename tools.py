from typing import Dict, List, Tuple

from sklearn.metrics import classification_report, confusion_matrix
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


def get_rel(seq: str, pair: Tuple[str, str]):
    template = seq.split(".\n\n")[-1]
    trg1, trg2 = pair
    rel = template.split("{trg1} is ".format(trg1=trg1))[1]
    rel = rel.split(" {trg2} ".format(trg2))[0]
    return rel

def compute_f1(dataset: str, predicts: List[str], golds: List[str], pairs: List[Tuple[str, str]], report=False):
    predict_rels = []
    gold_rels = []
    if dataset == "HiEve":
        rel_idx = {
            'super-event of': 0, 
            'sub-event of': 1, 
            'coreference with': 2, 
            'non-relation with': 3
        }
    for pred, gold, pair in zip(predicts, golds, pairs):
        pred_rel = get_rel(pred, pair)
        pred_rel_idx = rel_idx.get(pred_rel) if rel_idx.get(pred_rel) != None else 3
        gold_rel = get_rel(gold, pair)
        gold_rel_idx = rel_idx[gold_rel]
        predict_rels.append(pred_rel_idx)
        gold_rels.append(gold_rel_idx)
    
    CM = confusion_matrix(gold_rels, predict_rels)
    if dataset == "HiEve":
        true = sum([CM[i, i] for i in range(2)])
        sum_pred = sum([CM[i, 0:2].sum() for i in range(4)])
        sum_gold = sum([CM[i].sum() for i in range(2)])
        P = true / sum_pred
        R = true / sum_gold
        F1 = 2 * P * R / (P + R)
    if report:
        print(f"CM: \n{CM}")
        print("Classification report: \n{}".format(classification_report(gold_rels, predict_rels)))     
        print("  P: {0:.3f}".format(P))
        print("  R: {0:.3f}".format(R))
        print("  F1: {0:.3f}".format(F1))     
    return P, R, F1




