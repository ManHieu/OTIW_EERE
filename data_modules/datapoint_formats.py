from itertools import combinations
import networkx as nx


DATAPOINT = {}


def register_datapoint(func):
    DATAPOINT[str(func.__name__)] = func
    return func


def get_datapoint(type, mydict, doc_info=True):
    return DATAPOINT[type](mydict, doc_info)


@register_datapoint
def hieve_datapoint(my_dict):
    """
    Format data for HiEve dataset: each event pair is a datapoint
    """
    data_points = []
    doc_sentences = []
    for sent in my_dict['sentences']:
        doc_sentences.append(' '.join(sent['tokens']))

    for (eid1, eid2), rel in my_dict['relation_dict'].items():
        e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
        s1, s2 = e1['sent_id'], e2['sent_id']
        
        e1_poss, e1_mention = e1['token_id'], e1['mention']
        e2_poss, e2_mention = e2['token_id'], e2['mention']
        triggers = [{'tok_position': e1_poss, 'mention': e1_mention, 'sid': s1},
                    {'tok_position': e2_poss, 'mention': e2_mention, 'sid': s2}]
        
        data_point = {
            'doc_sentences': doc_sentences,
            'triggers': triggers,
            'relations': [0, 1, rel]
        }
        data_points.append(data_point)
    return data_points


@register_datapoint
def hieve_datapoint_v2(my_dict):
    """
    Format data for HiEve dataset which choose the most similar context sentences 
    (two host sentences is a dataopint that mean it can have more than one event pair labeled in a datapoint)
    """
    sentence_pairs = combinations(range(len(my_dict['sentences'])), r=2)
    sentence_pairs = set(sentence_pairs)
    data_points = []
    doc_sentences = []
    for sent in my_dict['sentences']:
        doc_sentences.append(' '.join(sent['tokens']))

    for s1, s2 in sentence_pairs:
        triggers = []
        labels = []
        for (eid1, eid2), rel in my_dict['relation_dict'].items():
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            _s1, _s2 = e1['sent_id'], e2['sent_id']
            if (_s1 == s1 and _s2 == s2) or (_s1 == s2 and _s2 == s1):
                e1_point = {'position': e1['token_id'], 'mention': e1['mention'], 'sid': _s1}
                e2_point = {'position': e2['token_id'], 'mention': e2['mention'], 'sid': _s2}
                if e1_point not in triggers:
                    triggers.append(e1_point)
                if e2_point not in triggers:
                    triggers.append(e2_point)
                labels.append((triggers.index(e1_point), triggers.index(e2_point), rel))
        if len(labels) > 0:
            data_point = {
                'doc_sentences': doc_sentences,
                'triggers': triggers,
                'relations': labels
            }
            data_points.append(data_point)
    
    return data_points

    

