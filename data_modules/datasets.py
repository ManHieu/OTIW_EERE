import json
import os
import random
from typing import Dict, List
from data_modules.base_dataset import BaseDataset
from data_modules.input_example import Entity, InputExample, Relation, RelationType


DATASETS: Dict[str, BaseDataset] = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(name:str,
                tokenizer: str,
                data_dir: str,
                max_input_length: int,
                seed: int = None,
                split = 'train',):
    '''
    Load a registered dataset
    '''
    return DATASETS[name](
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_input_length=max_input_length,
        seed=seed,
        split=split,
    )


class EEREDataset(BaseDataset):
    relation_types = None
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use
    sample = 1

    def load_schema(self):
        self.relation_types = {natural: RelationType(short=short, natural=natural)
                            for short, natural in self.natural_relation_types.items()}
    
    def load_data(self, split: str) -> List[InputExample]:
        examples = []
        self.load_schema()
        file_path = os.path.join(self.data_path, f'{split}.json')
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} for split {split} of {self.name} with the sample rate is {self.sample}")
            for i, datapoint in enumerate(data):
                # print(f"datapoint: {datapoint}")
                triggers = [Entity(mention=trigger['mention'], position=trigger['position'],
                            sid=trigger['sid'], start_in_sent=trigger['sent_span'][0], end_in_sent=trigger['sent_span'][1])
                            for trigger in datapoint['triggers']]
                
                relations = []
                for relation in datapoint['labels']:
                    relation_type = self.relation_types[relation[2]]
                    if relation_type.short == 'NoRel':
                        if random.uniform(0, 1) < self.sample:
                            relations.append(Relation(head=triggers[relation[0]], tail=triggers[relation[1]], type=relation_type))
                    else:
                        relations.append(Relation(head=triggers[relation[0]], tail=triggers[relation[1]], type=relation_type))
                if len(relations) >= 1:
                    example = InputExample(
                                        id=i,
                                        triggers=triggers,
                                        relations=relations,
                                        doc_sentences=datapoint['doc_sentences'],
                                        host_ids=datapoint['host_ids']
                    )
                    examples.append(example)
                
        return examples


@register_dataset
class HiEveDataset(EEREDataset):
    name = 'HiEve'
    sample = 0.4

    natural_relation_types = {
                            "SuperSub": 'super-event of', 
                            "SubSuper": 'sub-event of', 
                            "Coref": 'coreference with', 
                            "NoRel": 'non-relation with'
                            }
