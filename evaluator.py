import torch
import torch.nn as nn
import os
from datasets import load_metric
import numpy as np
from utils import data_loader
import time
from models import CommentGenerator_fusion
import pandas as pd
import json
import glob
import argparse


class Evaluator:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Let's use {torch.cuda.device_count()} GPUs!\n\n")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        args.name = ''
        if 'music4all' in args.corpus:
            args.name = args.corpus[10:]

        if 'val' in args.phase:
            print('#########Val set##########')
            val_loader = data_loader('val', args.batch_size, args.csv_path, args.audio_path, args.name,
                                   args.audio_len, args.comment_len, args.sample_rate, args.n_workers)
            print('Done\n')
        else:
            val_loader = None

        if 'test' in args.phase:
            print('#########Test set##########')
            test_loader = data_loader('test', args.batch_size, args.csv_path, args.audio_path, args.name,
                                   args.audio_len, args.comment_len, args.sample_rate, args.n_workers)
            print('Done\n')
        else:
            test_loader = None

        print('Initializing model....')
        model = CommentGenerator_fusion(args.comment_len, args.num_beams)

        prefix = f'results/{args.corpus}_w_contrast={args.w_contrast}/{args.idx}/gen_n_beams={args.num_beams}'
        os.makedirs(prefix, exist_ok=True)
        print('Resuming from the saved checkpoint....')
        ckp = torch.load(f'results/{args.corpus}_w_contrast={args.w_contrast}/{args.idx}/model_{args.epoch}.pt')
        model_state_dict = ckp['model_state_dict']
        for each in model_state_dict:
            model_state_dict[each] = model_state_dict[each].to(device)
        model.load_state_dict(model_state_dict)
        model = nn.DataParallel(model)
        model.to(device)
        print('Done\n')

        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.args = args
        self.prefix = prefix

    def eval(self, phase='val'):
        t1 = time.time()
        print(f'Evaluating on {phase} set....')
        loader = self.val_loader if phase == 'val' else self.test_loader
        lyrics = []
        comments = []
        output_ids = []
        interval = max(len(loader) // 20, 1)
        self.model.eval()
        t = time.time()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                wav_embs = batch['wav_emb'].to(self.device)
                input_ids_lyrics = batch['input_ids_lyrics'].to(self.device)
                lyrics_ = batch['lyrics']
                comments_ = batch['comment']
                output_ids_ = self.model(wav_embs, input_ids_lyrics)
                output_ids_ = output_ids_.detach().to('cpu').numpy().tolist()
                lyrics.extend(lyrics_)
                comments.extend(comments_)
                output_ids.extend(output_ids_)
                batch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t))
                if i % interval == 0 or i == len(loader) - 1:
                    print(f"Batch: {i+1}/{len(loader)}\tbatch_time:{batch_time}")
                    t = time.time()

        print('Decoding....')
        from transformers import BartTokenizer

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        comments_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print('Done\n')

        df = pd.DataFrame({'lyrics': lyrics, 'comment': comments, 'comment_pred': comments_preds})
        df.to_csv(f'{self.prefix}/df_gen_{phase}_{self.args.epoch}.csv', index=False)
        print(df.head())
        del df

        print('Loading metrics....')
        metric_rouge = load_metric('metrics/rouge.py')
        metric_meteor = load_metric('metrics/meteor.py')
        print('Done\n')

        print('Computing metrics....')

        print('Rouge....')
        metric_rouge.add_batch(predictions=comments_preds, references=comments)
        rouges = metric_rouge.compute()
        rouge1 = rouges['rouge1'].mid.recall
        rouge2 = rouges['rouge2'].mid.recall
        rougeL = rouges['rougeL'].mid.recall
        rouge1 = round(rouge1, 4)
        rouge2 = round(rouge2, 4)
        rougeL = round(rougeL, 4)
        del metric_rouge
        print('Done\n')

        print('Meteor....')
        metric_meteor.add_batch(predictions=comments_preds, references=comments)
        meteor = metric_meteor.compute()['meteor']
        meteor = round(meteor, 4)
        del metric_meteor
        print('Done\n')

        path_metrics = f'{self.prefix}/df_metrics_{phase}.csv'
        if not os.path.exists(path_metrics):
            df_metrics = pd.DataFrame([[phase, self.args.epoch, rouge1, rouge2, rougeL, meteor]],
                              columns=['phase', 'epoch',
                                       'ROUGE1', 'ROUGE2', 'ROUGEL', 'METEOR'])
        else:
            df_metrics = pd.read_csv(path_metrics)
            df_metrics2 = pd.DataFrame([[phase, self.args.epoch, rouge1, rouge2, rougeL, meteor]],
                               columns=['phase', 'epoch',
                                        'ROUGE1', 'ROUGE2', 'ROUGEL', 'METEOR'])
            df_metrics = df_metrics.append(df_metrics2)

        df_metrics.to_csv(path_metrics, index=False)

        print(f'ROUGE1: {rouge1:.3f}\tROUGE2: {rouge2:.3f}\tROUGEL: {rougeL:.3f}\t'
              f'Meteor: {meteor:.3f}\t\n\n\n')

        print(f"*****Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='music4all_full',
                        choices=('music4all_full', 'music4all_positive', 'music4all_not_negative'))
    parser.add_argument('--idx', type=int, default=1)

    # data prep
    parser.add_argument('--phase', type=str, choices=('val', 'test', 'val_test'), default='val_test')
    parser.add_argument('--csv_path', type=str, default='../bart-fusion/code/datasets',
                        help='where to put the csv files')
    parser.add_argument('--audio_path', type=str, default='../data/music4all/audios',
                        help='where to put the audio numpy files')
    parser.add_argument('--audio_len', type=int, default=15)
    parser.add_argument('--comment_len', type=int, default=200)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--w_contrast', type=float, default=0.)

    # training config
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--num_beams', type=int, default=1)

    args = parser.parse_args()

    prefix = f'results/{args.corpus}_w_contrast={args.w_contrast}/{args.idx}'
    with open(f'{prefix}/config.json', 'r') as f:
        config = json.load(f)
    for name in ['audio_len', 'comment_len', 'sample_rate', 'seed']:
        args.__setattr__(name, config[name])
    print(json.dumps(args.__dict__, indent=2))

    model_names = glob.glob(f'{prefix}/model_*.pt')

    engine = Evaluator(args)
    if 'test' in args.phase:
        engine.eval('test')
    print('\n\n\n\n')
    if 'val' in args.phase:
        engine.eval('val')