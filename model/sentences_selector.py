from collections import defaultdict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from model.sinkhorn import SinkhornDistance


class SOT(nn.Module):
    def __init__(self,
                encoder: AutoModel,
                encoder_hidden_size: int,
                hidden_size: int,
                rnn_num_layers: int,
                is_finetune: bool = False,
                OT_eps: float = 0.1,
                OT_max_iter: int = 100,
                OT_reduction: str = 'mean',
                dropout: float = 0.5,
                null_prob: float = 0.5,
                ) -> None:
        super().__init__()

        self.encoder = encoder
        self.is_finetune = is_finetune

        if rnn_num_layers > 1:
            self.lstm = nn.LSTM(encoder_hidden_size, hidden_size, rnn_num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.lstm = nn.LSTM(encoder_hidden_size, hidden_size, rnn_num_layers,
                                batch_first=True, bidirectional=True)

        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)
        self.null_prob = null_prob
    
    def encode(self, input_ids, attn_masks):
        print(f"sentence_vector:")
        hidden_states = self.encoder(input_ids, attn_masks, output_hidden_states=True).hidden_states
        return hidden_states[-1][:, 0] # (ns, encoder_hidden_size)
    
    def encode_with_rnn(self, inp: torch.Tensor(), ls: List[int]) -> torch.Tensor(): # (batch_size, max_ns, hidden_dim*2)
        packed = pack_padded_sequence(inp, ls, batch_first=True, enforce_sorted=False)
        rnn_encode, _ = self.lstm(packed)
        outp, _ = pad_packed_sequence(rnn_encode, batch_first=True)
        return outp
    
    def forward(self, 
                doc_ids: List[torch.Tensor],        # (n_sent, max_len_sent)
                doc_attn_mask: List[torch.Tensor],  # (n_sent, max_len_sent)
                host_sent_id: List[List[int]]):
        
        bs = len(host_sent_id)

        if self.is_finetune:
            sentences_emb = [self.encode(ids, attn_mask) for ids, attn_mask in zip(doc_ids, doc_attn_mask)]
        else:
            with torch.no_grad():
                sentences_emb = [self.encode(ids, attn_mask) for ids, attn_mask in zip(doc_ids, doc_attn_mask)]
        
        ls = [emb.size(0) for emb in sentences_emb]
        sentences_emb = self.encode_with_rnn(sentences_emb, ls)
        
        host_sentence_emb = []
        context_sentence_emb = []
        host_maginal = []
        context_maginal = []
        context_sent_id = []
        for i in range(bs):
            ns = ls[i]
            host_id = host_sent_id[i]
            context_id = list(set(range(ns)) - set(host_id))
            context_sent_id.append(context_id)
            _host_sentence_emb = sentences_emb[i, host_id]
            _context_sentence_emb = sentences_emb[i, context_id]
            _null_presenation = torch.mean(_context_sentence_emb, dim=0).unsqueeze(0)
            _host_sentence_emb = torch.cat([_null_presenation, _host_sentence_emb], dim=0)
            host_sentence_emb.append(_host_sentence_emb)
            context_sentence_emb.append(_context_sentence_emb)

            _host_maginal = torch.tensor([1.0/len(host_id)] * len(host_id), dtype=torch.float)
            _host_maginal = torch.cat([torch.Tensor([self.null_prob]), (1 - self.null_prob) * _host_maginal], dim=0)
            _context_maginal = torch.tensor(context_id, dtype=torch.float)
            _context_maginal = torch.stack([_context_maginal - host_id[0], _context_maginal - host_id[-1]], dim=0)
            _context_maginal = torch.min(torch.abs(_context_maginal), dim=0)[0]
            _context_maginal = torch.softmax(_context_maginal, dim=0)
            host_maginal.append(_host_maginal)
            context_maginal.append(_context_maginal)
        
        host_sentence_emb = pad_sequence(host_sentence_emb, batch_first=True).cuda()
        context_sentence_emb = pad_sequence(context_sentence_emb, batch_first=True).cuda()
        host_maginal = pad_sequence(host_maginal, batch_first=True).cuda()
        context_maginal = pad_sequence(context_maginal, batch_first=True).cuda()

        cost, pi, C = self.sinkhorn(context_sentence_emb, host_sentence_emb, context_maginal, host_maginal, cuda=True)
        algins = torch.max(pi, dim=2)[1]
        selected_sentences = []
        for i in range(bs):
            context_id = context_sent_id[i]
            host_id = host_sent_id[i]
            algin = algins[i] - 1
            algin = list(algin[0: len(context_id)].cpu().numpy())
            _selected_sentences = defaultdict(list)
            for _host, context_id in zip(algin, context_id):
                if _host >= 0:
                    _selected_sentences[host_id[_host]].append(context_id)
            selected_sentences.append(dict(_selected_sentences))
        
        return cost, selected_sentences




