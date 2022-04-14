from collections import defaultdict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.nn.utils.rnn import pad_sequence
from model.sinkhorn import SinkhornDistance


class WOT(nn.Module):
    def __init__(self,
                encoder: AutoModel,
                OT_eps: float = 0.1,
                OT_max_iter: int = 100,
                OT_reduction: str = 'mean',
                null_prob: float = 0.5,
                ) -> None:

        self.encoder = encoder
        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)
        self.null_prob = null_prob

    def forward(self, 
                input_ids: torch.Tensor,        # (bs, max_sq_len)
                input_attn_mask: torch.Tensor,  # (bs, max_sq_len)
                trigger_pos: List[List[int]],
                ls: List[int]
                ):
        
        bs = len(trigger_pos)
        input_emb = self.encoder(input_ids, input_attn_mask)[0]

        trigger_emb = []
        context_emb = []
        trigger_maginal = []
        context_maginal = []
        for i in range(bs):
            _trigger_pos = trigger_pos[i]
            _context_pos = list(set(range(ls[i])) - set(_trigger_pos))
            _trigger_emb = input_emb[i, _trigger_pos]
            _context_emb = input_emb[i, _context_pos]
            _null_emb = torch.mean(_context_emb, dim=0).unsqueeze(0)
            _trigger_emb = torch.cat([_null_emb, _trigger_emb], dim=0)
            trigger_emb.append(_trigger_emb)
            context_emb.append(_context_emb)

            _trigger_maginal = torch.tensor([1.0/len(_trigger_pos)] * len(_trigger_pos), dtype=torch.float)
            _trigger_maginal = torch.cat([torch.Tensor([self.null_prob]), (1 - self.null_prob) * _trigger_maginal], dim=0)
            _context_pos = torch.tensor(_context_pos, dtype=torch.float)
            _context_maginal = torch.stack([_context_pos - i for i in _trigger_pos], dim=0)
            _context_maginal = torch.min(torch.abs(_context_maginal), dim=0)[0]
            _context_maginal = nn.Softmax(_context_maginal, dim=0)
            trigger_maginal.append(_trigger_maginal)
            context_maginal.append(_context_maginal)

        trigger_emb = pad_sequence(trigger_emb, batch_first=True).cuda()
        context_emb = pad_sequence(context_emb, batch_first=True).cuda()
        trigger_maginal = pad_sequence(trigger_maginal, batch_first=True).cuda()
        context_maginal = pad_sequence(context_maginal, batch_first=True).cuda()
        cost, pi, C = self.sinkhorn(context_emb, trigger_emb, context_maginal, trigger_maginal, cuda=True)
        algins = torch.max(pi, dim=2)[1]
        important_words = []
        for i in range(bs):
            ns = ls[i]
            _important_words = []
            _important_words.extend(trigger_pos[i])
            context_tok_id = list(set(range(ns)) - set(trigger_pos[i]))
            algin = algins[i] - 1
            algin = list(algin[0: len(context_tok_id)].cpu().numpy())
            for _trigger_id, context_id in zip(algin, context_tok_id):
                if _trigger_id >= 0:
                    _important_words.append(context_id)
            important_words.append(_important_words)
        
        return cost, important_words


        
