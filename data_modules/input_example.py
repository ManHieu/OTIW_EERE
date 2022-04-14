from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple, Union
from torch import Tensor
from torch.utils.data.dataset import Dataset


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class RelationType:
    """
    A relation type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    mention: str = None                     # mention of entity
    type: Optional[EntityType] = None       # entity type
    sid: Optional[int] = None               # sentence id which containt entity
    position: Optional[List[int]] = None    # possition of entity in sentence

    def to_tuple(self):
        return self.type.natural, self.mention, self.sid, self.position

    def __hash__(self):
        return hash((self.sid, self.mention, self.position))


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: RelationType  # relation type
    head: Entity        # head of the relation
    tail: Entity        # tail of the relation

    def to_tuple(self):
        return self.type.natural, self.head.to_tuple(), self.tail.to_tuple()


@dataclass
class InputExample:
    """
    A single training/ testing example
    """
    dataset: Optional[Dataset] = None
    id: Optional[str] = None
    triggers: List[Entity] = None
    relations: List[Relation] = None
    doc_sentences: List[str] = None

