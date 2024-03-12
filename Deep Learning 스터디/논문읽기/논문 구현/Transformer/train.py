import sys
sys.path.append("..")
import os, argparse, datetime, time, re, collections, random
from tqdm import tqdm, trange
import numpy as np
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import sentencepiece as spm

import config as cfg
import model as transformer
import data
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_vocab(file):
    vocab = spm.SentencePieceProcessor()
    vocab.load(file)
    return vocab

""" random seed """
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


""" init_process_group """ 
def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


""" destroy_process_group """
def destroy_process_group():
    dist.destroy_process_group()


""" 모델 epoch 평가 """
def eval_epoch(config, rank, model, data_loader):
    matchs = []
    model.eval()

    n_word_total = 0
    n_correct_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid({rank})") as pbar:
        for i, value in enumerate(data_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]
            _, indices = logits.max(1)

            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0


""" 모델 epoch 학습 """
def train_epoch(config, rank, epoch, model, criterion, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]

            loss = criterion(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


vocab_model_path = r"C:\Users\cbigo\Desktop\temp\kowiki.model"
config_path = r"D:\workspace\difficult\git\my_repository\Deep Learning 스터디\논문읽기\논문 구현\Transformer\config.json"
train_json_path = r"C:\Users\cbigo\Desktop\temp\ratings_train.json"
test_json_path = r"C:\Users\cbigo\Desktop\temp\ratings_test.json"
"""
https://drive.google.com/drive/folders/15XGr-L-W6DSoR5TbniPMJASPsA0IDTiN
"""


""" 모델 학습 """
def train_model(rank, world_size, args):

    vocab = load_vocab(vocab_model_path)
    config = cfg.Config.load(config_path)
    config.n_enc_vocab, config.n_dec_vocab = len(vocab), len(vocab)
    config.device = torch.device("cuda")
    print(config)

    best_epoch, best_loss, best_score = 0, 0, 0
    model = transformer.MovieClassification(config)
    if os.path.isfile(args.save):
        best_epoch, best_loss, best_score = model.load(args.save)
        print(f"rank: {rank} load state dict from: {args.save}")
    
    model.to(config.device)

    criterion = torch.nn.CrossEntropyLoss()

    train_loader, train_sampler = data.build_data_loader(vocab, train_json_path, args, shuffle=True)
    test_loader, _ = data.build_data_loader(vocab, test_json_path, args, shuffle=False)

    t_total = len(train_loader) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    offset = best_epoch
    for step in trange(args.epoch, desc="Epoch"):
        if train_sampler:
            train_sampler.set_epoch(step)
        epoch = step + offset

        loss = train_epoch(config, rank, epoch, model, criterion, optimizer, scheduler, train_loader)
        score = eval_epoch(config, rank, model, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="save_best.pth", type=str, required=False,
                        help="save file")
    parser.add_argument("--epoch", default=20, type=int, required=False,
                        help="epoch")
    parser.add_argument("--batch", default=32, type=int, required=False,
                        help="batch")
    parser.add_argument("--gpu", default=None, type=int, required=False,
                        help="GPU id to use.")
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help="random seed for initialization")
    parser.add_argument('--weight_decay', type=float, default=0, required=False,
                        help="weight decay")
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False,
                        help="learning rate")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, required=False,
                        help="adam epsilon")
    parser.add_argument('--warmup_steps', type=float, default=0, required=False,
                        help="warmup steps")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()
    else:
        args.n_gpu = 0
    set_seed(args)


    train_model(0, args.n_gpu, args)
