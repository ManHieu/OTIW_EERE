import json
from platform import release
from typing import List
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_modules.input_example import InputExample
from data_modules.input_formats import INPUT_FORMATS
from data_modules.tools import padding, tokenized_to_origin_span
from sentences_selector import SOT
from important_word_selector import WOT
from tools import compute_f1, compute_feature_for_augmented_seq, token_id_lookup


class HOTIW_EERE(pl.LightningModule):
    def __init__(self,
                dataset: str,
                tokenizer_name: str,
                pretrain_model_name: str,
                rnn_hidden_size: int,
                rnn_num_layers: int,
                lr: float,
                warmup: float,
                adam_epsilon: float,
                weight_decay: float,
                w_sent_cost: float,
                w_word_cost: float,
                input_format: str = 'EERE_MLM',
                is_finetune: bool = False,
                OT_eps: float = 0.1,
                OT_max_iter: int = 100,
                OT_reduction: str = 'mean',
                dropout: float = 0.5,
                sentence_null_prob: float = 0.5,
                word_null_prob: float = 0.5) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pretrained_model: AutoModel = AutoModel.from_pretrained(pretrain_model_name)
        self.pretrained_hidden_size = 1024 if 'large' in pretrain_model_name else 768

        self.sentence_selector = SOT(encoder=self.pretrained_model,
                                    encoder_hidden_size=self.pretrained_hidden_size,
                                    hidden_size=rnn_hidden_size,
                                    rnn_num_layers=rnn_num_layers,
                                    is_finetune=is_finetune,
                                    OT_eps=OT_eps,
                                    OT_max_iter=OT_max_iter,
                                    OT_reduction=OT_reduction,
                                    dropout=dropout,
                                    null_prob=sentence_null_prob,)
        
        self.important_word_selector = WOT(encoder=self.pretrained_model,
                                        OT_eps=OT_eps,
                                        OT_max_iter=OT_max_iter,
                                        OT_reduction=OT_reduction,
                                        null_prob=word_null_prob,)

        self.input_format = INPUT_FORMATS[input_format](self.tokenizer)

        self.model_results = []

    def forward(self, batch: List[InputExample]):
        doc_ids = []
        doc_attn_mask = []
        host_ids = []
        for example in batch:
            doc_sentences = example.doc_sentences
            doc_tokenized = self.tokenizer(doc_sentences, return_tensors="pt")
            _doc_ids = doc_tokenized.input_ids
            _doc_attn_mask = doc_tokenized.attention_mask
            _host_ids = example.host_ids
            doc_ids.append(_doc_ids)
            doc_attn_mask.append(_doc_attn_mask)
            host_ids.append(_host_ids)
        sent_transport_cost, selected_senteces = self.sentence_selector(doc_ids, doc_attn_mask, host_ids)

        input_seqs = []
        input_ids = []
        input_attn_mask = []
        ls = []
        head_ids = []
        tail_ids = []
        rels = []
        batch_subwords = []
        batch_subwords_len = []
        for selected_idx, example in zip(selected_senteces, batch):
            relations = example.relations
            doc_sentences = example.doc_sentences
            for relation in relations:
                augmented_sequence, head_span_in_augm, tail_span_in_augm = compute_feature_for_augmented_seq(relation, doc_sentences, selected_idx)
                augmented_sequence_tokenized = self.tokenizer(augmented_sequence)
                augmented_sequence_ids = augmented_sequence_tokenized['input_ids']
                augmented_sequence_attn_mask = augmented_sequence_tokenized['attention_mask']
                subwords = []
                for idx in augmented_sequence_ids:
                    subwords.append(self.tokenizer.decode([idx]))
                batch_subwords.append(subwords)
                subwords_span = tokenized_to_origin_span(augmented_sequence, subwords[1:-1])
                batch_subwords_len.append([abs(span[1] - span[0]) for span in subwords_span])
                head_token_id = token_id_lookup(subwords_span, head_span_in_augm[0], head_span_in_augm[1] - 1, start_id=1)
                tail_token_id = token_id_lookup(subwords_span, tail_span_in_augm[0], tail_span_in_augm[1] - 1, start_id=1)
                input_seqs.append(augmented_sequence)
                input_ids.append(augmented_sequence_ids)
                input_attn_mask.append(augmented_sequence_attn_mask)
                ls.append(len(augmented_sequence_ids))
                head_ids.append(head_token_id)
                tail_ids.append(tail_token_id)
                rels.append(relation)
        max_ns = max(ls)
        input_ids = [padding(ids, max_sent_len=max_ns, pad_tok=self.tokenizer.pad_token_id) for ids in input_ids]
        input_attn_mask = [padding(mask, max_sent_len=max_ns, pad_tok=0) for mask in input_attn_mask]
        input_ids = torch.tensor(input_ids)
        input_attn_mask = torch.tensor(input_attn_mask)

        input_emb = self.pretrained_model(input_ids, input_attn_mask, output_hidden_states=True).hidden_states[-1]
        trigger_pos = [head_id + tail_id for head_id, tail_id in zip(head_ids, tail_ids)]
        word_transport_cost, important_word_ids = self.important_word_selector(input_emb, trigger_pos, ls)
        
        bs = len(input_seqs)
        input_ids = []
        label_ids = []
        labels = []
        ls = []
        pairs = []
        for i in range(bs):
            seq = input_seqs[i]
            _important_words_ids = important_word_ids[i]
            _head_ids = head_ids[i]
            _tail_ids = tail_ids[i]
            _subwords = batch_subwords[i]
            _subwords_len = batch_subwords_len[i]
            _head = [(idx, _subwords[idx]) for idx in _head_ids]
            _tail = [(idx, _subwords[idx]) for idx in _tail_ids]
            _important_words = [(idx, _subwords[idx]) for idx in _important_words_ids]
            _input_ids, _label_ids, _label = self.input_format.format_input(context=seq,
                                                                            important_words=_important_words, 
                                                                            trigger1=_head, 
                                                                            trigger2=_tail, 
                                                                            rel=rels[i].type.natural)
            ls.append(len(_input_ids))
            input_ids.append(_input_ids)
            label_ids.append(_label_ids)
            labels.append(_label)
            pairs.append((''.join([sw for idx, sw in _head]), ''.join([sw for idx, sw in _tail])))
        max_ns = max(ls)
        input_ids = [padding(ids, max_sent_len=max_ns, pad_tok=self.tokenizer.pad_token_id) for ids in input_ids]
        label_ids = [padding(ids, max_sent_len=max_ns, pad_tok=self.tokenizer.pad_token_id) for ids in label_ids]
        input_attn_mask = [padding([1]*l, max_sent_len=max_ns, pad_tok=0) for l in ls]
        input_ids = torch.tensor(input_ids)
        label_ids = torch.tensor(label_ids)
        label_ids = torch.where(input_ids == self.tokenizer.mask_token_id, label_ids, -100)
        input_attn_mask = torch.tensor(input_attn_mask)
        outputs = self.pretrained_model(input_ids, input_attn_mask, labels=label_ids)
        predicted_token_id = outputs.logits.argmax(axis=-1)
        predicted_seq = [self.tokenizer.decode(predicted_token_id[i]) for i in range(predicted_token_id.size(0))]
        mlm_loss = outputs.loss
        return sent_transport_cost, word_transport_cost, mlm_loss, predicted_seq, labels, pairs
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        sent_transport_cost, word_transport_cost, mlm_loss, predicted_seq, labels, pairs = self.forward(batch)
        loss = self.hparams.w_sent_cost * sent_transport_cost \
                + self.hparams.w_word_cost * word_transport_cost \
                + (1.0 - self.hparams.w_sent_cost - self.hparams.w_word_cost) * mlm_loss
        self.log_dict({'s_OT': sent_transport_cost, 'w_OT': word_transport_cost, 'mlm': mlm_loss}, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        sent_transport_cost, word_transport_cost, mlm_loss, predicted_seq, labels, pairs = self.forward(batch)
        return predicted_seq, labels, pairs
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        golds = []
        predicts = []
        preds = []
        pairs = []
        for output in outputs:
            for sample in zip(*output):
                predicts.append(sample[0])
                golds.append(sample[1])
                pairs.append(sample[2])
                preds.append({
                    'pairs': sample[2],
                    'predicted': sample[0],
                    'gold': sample[1]
                })
        p, r, f1 = compute_f1(dataset=self.hparams.dataset,
                            golds=golds,
                            predicts=predicts,
                            pairs=pairs,
                            report=True)
        self.log_dict({'f1_dev': f1, 'p_dev': p, 'r_dev': r}, prog_bar=True)
        with open('./dev.json','w') as writer:
            writer.write(json.dumps(preds, indent=6)+'\n')
        return f1
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        sent_transport_cost, word_transport_cost, mlm_loss, predicted_seq, labels, pairs = self.forward(batch)
        return predicted_seq, labels, pairs
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        golds = []
        predicts = []
        preds = []
        pairs = []
        for output in outputs:
            for sample in zip(*output):
                predicts.append(sample[0])
                golds.append(sample[1])
                pairs.append(sample[2])
                preds.append({
                    'pairs': sample[2],
                    'predicted': sample[0],
                    'gold': sample[1]
                })
        p, r, f1 = compute_f1(dataset=self.hparams.dataset,
                            golds=golds,
                            predicts=predicts,
                            pairs=pairs,
                            report=True)
        with open('./test.json','w') as writer:
            writer.write(json.dumps(preds, indent=6)+'\n')
        
        self.model_results = (p, r, f1)
    
    def configure_optimizers(self):
        """
        Prepare optimizer and schedule (linear warmup and decay)
        """
        num_batches = self.trainer.max_epochs * len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_pretrain_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },]
        optimizer = AdamW(optimizer_grouped_pretrain_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        num_warmup_steps = self.hparams.warmup * num_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }
