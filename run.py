import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import random
from typing import Dict
import optuna
from pytorch_lightning.trainer.trainer import Trainer
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from pytorch_lightning.utilities.seed import seed_everything
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_modules.data_modules import load_data_module
from data_modules.preprocess import Preprocessor
from model.model import HOTIW_EERE
import shutil


def run(defaults: Dict, random_state):
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    print("Hyperparams: {}".format(defaults))
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            defaults[key] = True if defaults[key]=='True' else False
        if defaults[key] == 'None':
            defaults[key] = None
    
    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    # print(second_parser.parse_args_into_dataclasses(remaining_args))
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)
    data_args.datasets = job

    if args.tuning:
        training_args.output_dir = './tuning_experiments'
    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    f1s = []
    ps = []
    rs = []
    val_f1s = []
    val_ps = []
    val_rs = []
    for i in range(data_args.n_fold):
        print(f"TRAINING AND TESTING IN FOLD {i}: ")
        fold_dir = f'{data_args.data_dir}/{i}' if data_args.n_fold != 1 else data_args.data_dir
        dm = load_data_module(module_name = 'EERE',
                            data_args=data_args,
                            fold_dir=fold_dir)
        
        # number_step_in_epoch = len(dm.train_dataloader())/training_args.gradient_accumulation_steps
        # construct name for the output directory
        output_dir = os.path.join(
            training_args.output_dir,
            f'{args.job}'
            f'-random_state{random_state}'
            f'-{model_args.residual_type}'
            f'-lr{training_args.lr}'
            f'-e_lr{training_args.encoder_lr}'
            f'-eps{training_args.num_epoches}'
            f'-regu_weight{training_args.regular_loss_weight}'
            f'-OT_weight{training_args.OT_loss_weight}'
            f'-gcn_num_layers{model_args.gcn_num_layers}'
            f'-fn_actv{model_args.fn_actv}'
            f'-rnn_hidden{model_args.hidden_size}')
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        if data_args.n_fold != 1:
            output_dir = os.path.join(output_dir,f'fold{i}')
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
        
        checkpoint_callback = ModelCheckpoint(
                                    dirpath=output_dir,
                                    save_top_k=1,
                                    monitor='f1_dev',
                                    mode='max',
                                    save_weights_only=True,
                                    filename='{epoch}-{f1_dev:.2f}', # this cannot contain slashes 
                                    )
        lr_logger = LearningRateMonitor(logging_interval='step')
        model = HOTIW_EERE(dataset=job,
                        tokenizer_name=data_args.tokenizer,
                        pretrain_model_name=model_args.pretrained_model_name_or_path,
                        rnn_hidden_size=model_args.rnn_hidden_size,
                        rnn_num_layers=model_args.rnn_num_layers,
                        lr=training_args.lr,
                        warmup=training_args.warmup_ratio,
                        adam_epsilon=training_args.adam_epsilon,
                        weight_decay=training_args.weight_decay,
                        w_sent_cost=training_args.sent_transport_cost,
                        w_word_cost=training_args.word_transport_cost,
                        input_format='EERE_MLM',
                        is_finetune=training_args.fine_tune_sentence_level,
                        OT_eps=model_args.OT_eps,
                        OT_max_iter=model_args.OT_max_iter,
                        OT_reduction=model_args.OT_reduction,
                        dropout=model_args.dropout,
                        sentence_null_prob=model_args.sentence_null_prob,
                        word_null_prob=model_args.word_null_prob
                        )
        
        trainer = Trainer(
            # logger=tb_logger,
            deterministic=True,
            min_epochs=training_args.num_epoches,
            max_epochs=training_args.num_epoches, 
            gpus=[args.gpu], 
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            num_sanity_val_steps=0, 
            val_check_interval=1.0, # use float to check every n epochs 
            callbacks = [lr_logger, checkpoint_callback],
        )

        print("Training....")
        dm.setup('fit')
        trainer.fit(model, dm)

        best_model = HOTIW_EERE.load_from_checkpoint(checkpoint_callback.best_model_path)
        print("Testing .....")
        dm.setup('test')
        trainer.test(best_model, dm)
        # print(best_model.model_results)
        p, r, f1 = best_model.model_results
        val_p, val_r, val_f1 = best_model.best_vals
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        val_f1s.append(val_f1)
        val_ps.append(val_p)
        val_rs.append(val_r)
        print(f"RESULT IN FOLD {i}: ")
        print(f"F1: {f1}")
        print(f"P: {p}")
        print(f"R: {r}")

        shutil.rmtree(f'{output_dir}')
    
        # with open(output_dir+f'-{f1}', 'w', encoding='utf-8') as f:
        #     f.write(f"F1: {f1} \n")
        #     f.write(f"P: {p} \n")
        #     f.write(f"R: {r} \n")

    f1 = sum(f1s)/len(f1s)
    p = sum(ps)/len(ps)
    r = sum(rs)/len(rs)
    val_f1 = sum(val_f1s)/len(val_f1s)
    val_p = sum(val_ps)/len(val_ps)
    val_r = sum(val_rs)/len(val_rs)
    print(f"F1: {f1} - P: {p} - R: {r}")
    
    return p, f1, r, val_p, val_r, val_f1


def objective(trial: optuna.Trial):
    defaults = {
        'lr': trial.suggest_categorical('lr', [8e-7, 1e-6, 2e-6]),
        'OT_max_iter': trial.suggest_categorical('OT_max_iter', [100]),
        'batch_size': trial.suggest_categorical('batch_size', [8]),
        'warmup_ratio': 0.1,
        'num_epoches': trial.suggest_categorical('num_epoches', [15]), # 
        'seed': 1741, # trial.suggest_int('seed', 1, 10000, log=True),
        'rnn_hidden_size': trial.suggest_categorical('rnn_hidden_size', [512, 768]),
        'rnn_num_layers': trial.suggest_categorical('rnn_num_layers', [1, 2]),
        'sent_transport_cost': trial.suggest_categorical('sent_transport_cost', [0.05, 0.1, 0.2]),
        'word_transport_cost': trial.suggest_categorical('word_transport_cost', [0.05, 0.1, 0.2]),
        'fine_tune_sentence_level': False,
        'dropout': trial.suggest_categorical('dropout', [0.5]),
        'sentence_null_prob': trial.suggest_categorical('sentence_null_prob', [0.5]),
        'word_null_prob': trial.suggest_categorical('word_null_prob', [0.5])
    }

    random_state = defaults['seed']
    print(f"Random_state: {random_state}")

    seed_everything(random_state, workers=True)

    dataset = args.job
    if dataset == 'HiEve':
        datapoint = 'hieve_datapoint_v3'
        corpus_dir = 'datasets/hievents_v2/processed/'
        processor = Preprocessor(dataset, datapoint)
        corpus = processor.load_dataset(corpus_dir)
        corpus = list(sorted(corpus, key=lambda x: x['doc_id']))
        train, test = train_test_split(corpus, train_size=100.0/120, test_size=20.0/120, random_state=random_state)
        train, validate = train_test_split(train, train_size=80.0/100., test_size=20.0/100, random_state=random_state)

        processed_path = 'datasets/hievents_v2/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/hievents_v2/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/hievents_v2/test.json'
        test = processor.process_and_save(test, processed_path)
    
    elif dataset == 'ESL':
        datapoint = 'ESL_datapoint'
        kfold = KFold(n_splits=5)
        processor = Preprocessor(dataset, datapoint, intra=True, inter=False)
        corpus_dir = './datasets/EventStoryLine/annotated_data/v0.9/'
        corpus = processor.load_dataset(corpus_dir)

        _train, test = [], []
        data = defaultdict(list)
        for my_dict in corpus:
            topic = my_dict['doc_id'].split('/')[0]
            data[topic].append(my_dict)

            if '37/' in my_dict['doc_id'] or '41/' in my_dict['doc_id']:
                test.append(my_dict)
            else:
                _train.append(my_dict)

        # print()
        # processed_path = f"./datasets/EventStoryLine/intra_data.json"
        # processed_data = processor.process_and_save(processed_path, data)

        random.shuffle(_train)
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(_train)):
            try:
                os.mkdir(f"./datasets/EventStoryLine/{fold}")
            except FileExistsError:
                pass

            train = [_train[id] for id in train_ids]
            # print(train[0])
            validate = [_train[id] for id in valid_ids]
        
            processed_path = f"./datasets/EventStoryLine/{fold}/train.json"
            train = processor.process_and_save(train, processed_path)

            processed_path = f"./datasets/EventStoryLine/{fold}/val.json"
            validate = processor.process_and_save(validate, processed_path)
            
            processed_path = f"./datasets/EventStoryLine/{fold}/test.json"
            test = processor.process_and_save(test, processed_path)
    
    p, f1, r, val_p, val_r, val_f1 = run(defaults=defaults, random_state=random_state)

    record_file_name = 'result.txt'
    if args.tuning:
        record_file_name = 'tuning_result.txt'

    with open(record_file_name, 'a', encoding='utf-8') as f:
        f.write(f"{'--'*10} \n")
        f.write(f"Random_state: {random_state}\n")
        f.write(f"Hyperparams: \n {defaults}\n")
        f.write(f"F1: {f1} - {val_f1} \n")
        f.write(f"P: {p} - {val_p} \n")
        f.write(f"R: {r} - {val_r} \n")

    return f1


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='tune hyperparameters')

    args, remaining_args = parser.parse_known_args()
    
    if args.tuning:
        print("tuning ......")
        # sampler = optuna.samplers.TPESampler(seed=1741)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25)
        trial = study.best_trial
        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))


