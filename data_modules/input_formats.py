from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

from transformers import PreTrainedTokenizer

from data_modules.tools import tokenized_to_origin_span
from .input_example import InputExample


class BaseInputFormat(ABC):
    name = None
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @abstractmethod
    def format_input(self):
        raise NotImplementedError


INPUT_FORMATS : Dict[str, BaseInputFormat] = {}


def register_input_format(format_class: BaseInputFormat):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


@register_input_format
class EEREInputFormat(BaseInputFormat):
    """
    The input format for EERE task
    """
    def format_input(self, 
                    example: InputExample, 
                    selected_sentences: List[int], 
                    important_words: List[Tuple[int, str]], 
                    trigger1: List[Tuple[int, str]], 
                    trigger2: List[Tuple[int, str]], 
                    rel: str):
        template = "{context}.\n\n{trg1} is {rel} {trg2} because {impotant_words}"
        
        context = ""
        for id, sentence in enumerate(example.doc_sentences):
            if id in selected_sentences:
                context += sentence

        important_words.extend(trigger1)
        important_words.extend(trigger2)
        important_words = sorted(important_words, key=lambda x: x[0])
        _important_words = ' '.join([word for idx, word in important_words])

        label = template.format(context=context, trg1=trigger1[1], rel=rel, trg2=trigger2[1], important_words=_important_words)
        
        label_ids = self.tokenizer(label).input_ids
        subwords = []
        for idx in label_ids:
            token = self.tokenizer.decode([idx])
            subwords.append(token)
        subwords_span = tokenized_to_origin_span(label, subwords[1:-1]) # w/o <s> and </s> with RoBERTa

        masked_char = []
        rel_start = len("{context}.\n\n{trg1} is ".format(context=context, trg1=trigger1[1]))
        rel_end = rel_start + len(rel)
        masked_char.extend(list(range(rel_start, rel_end)))

        important_words_start = len("{context}.\n\n{trg1} is {rel} {trg2} because ".format(context=context, trg1=trigger1[1], rel=rel, trg2=trigger2[1]))
        important_words_span = tokenized_to_origin_span(_important_words, [word for idx, word in important_words])
        trigger_idx = [idx for idx, tok in trigger1] + [idx for idx, tok in trigger2]
        for span, (idx, tok) in zip(important_words_span, important_words):
            if idx not in trigger_idx:
                masked_char.extend(list(range(span[0] + important_words_start, span[1] + important_words_start)))
        
        input_ids = [self.tokenizer.cls_token_id]
        for ids, span in zip(label_ids[1:-1], subwords_span):
            if span[0] in masked_char or span[1] in masked_char:
                input_ids.append(self.tokenizer.mask_token_id)
            else:
                input_ids.append(ids)
        
        return input_ids, label_ids

 

