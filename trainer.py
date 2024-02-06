import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_DATASETS_CACHE'] = 'hf_datasets_cache'
import time
import torch
import torch.nn as nn
import numpy as np
from utils import precompute_audio_embeddings, data_loader
from models import CommentGenerator_fusion
from models import InfoNCE, PositivePairLoss
import itertools
import argparse
import json


class Trainer:
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

        # precompute the audio embeddings
        precompute_audio_embeddings(args.csv_path, args.audio_path, args.audio_len, args.sample_rate,
                                    device, args.n_workers)

        print('Preparing dataset....')
        print('#########Training set##########')
        train_loader = data_loader('train', args.batch_size, args.csv_path, args.audio_path, args.name,
                                   args.audio_len, args.comment_len, args.sample_rate, args.n_workers)
        print('Done\n')

        prefix = f'results/{args.corpus}_w_contrast={args.w_contrast}/{args.idx}'

        model = CommentGenerator_fusion(args.comment_len)
        model = nn.DataParallel(model)
        model.to(device)

        if args.w_contrast > 0:
            if args.contrast_loss == 'infonce':
                criterion_contrast = InfoNCE(temperature=0.3, dim1=256, dim2=768)
            else:
                criterion_contrast = PositivePairLoss(dim1=256, dim2=768)
            criterion_contrast = nn.DataParallel(criterion_contrast)
            criterion_contrast.to(device)
        else:
            criterion_contrast = None

        import transformers
        params = model.parameters() if args.w_contrast == 0 else itertools.chain(model.parameters(),
                                                                                 criterion_contrast.parameters())
        optimizer = transformers.Adafactor(params, lr=args.lr, warmup_init=False, relative_step=False,
                                           weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.criterion_contrast = criterion_contrast
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.prefix = prefix

    def train(self):
        if self.args.load_ckp_path:
            print('Resuming from the specified checkpoint....')
            ckp = torch.load(self.args.load_ckp_path)
            model_state_dict = ckp['model_state_dict']
            for each in model_state_dict:
                model_state_dict[each] = model_state_dict[each].to(self.device)
            self.model.module.load_state_dict(model_state_dict)
            print('Done\n')

        if self.args.resume_epoch > 0:
            print('Resuming from the saved checkpoint....')
            ckp = torch.load(f'{self.prefix}/model_{self.args.resume_epoch}.pt')
            model_state_dict = ckp['model_state_dict']
            for each in model_state_dict:
                model_state_dict[each] = model_state_dict[each].to(self.device)
            self.model.module.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(ckp['optim_state_state'])
            print('Done\n')

        t = time.time()
        start_epoch = self.args.resume_epoch
        for epoch in range(start_epoch, start_epoch+self.args.epochs):
            print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss, loss_caption, loss_contrast = self.train_epoch(epoch)

            if (epoch + 1) % self.args.eval_interval == 0:
                print('Saving the checkpoint....')
                model_state_dict = self.model.module.state_dict()
                for each in model_state_dict:
                    model_state_dict[each] = model_state_dict[each].to('cpu')
                state_dict = {
                    'model_state_dict': model_state_dict,
                    'optim_state_state': self.optimizer.state_dict(),
                }
                torch.save(state_dict, f'{self.prefix}/model_{epoch+1}.pt')
                print('Done\n')

            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t))
            print(f'Epoch {epoch + 1}\tElapsed Time:{elapsed_time}\n'
                  f'Train Loss: {loss:.3f}\tTrain Loss_caption: {loss_caption:.3f}\t'
                  f'Train Loss_contrast: {loss_contrast:.3f}\n\n\n')

        print('Training finished.\n')

    def train_epoch(self, epoch):
        epoch_loss = 0
        epoch_loss_caption = 0
        epoch_loss_contrast = 0
        interval = max(len(self.train_loader) // 20, 1)
        self.model.train()
        t_epoch = time.time()
        t = time.time()
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            music_ids = np.unique(batch['music_id'], return_inverse=True)[1]
            music_ids = torch.tensor(music_ids).to(self.device)
            wav_embs = batch['wav_emb'].to(self.device)
            input_ids_lyrics = batch['input_ids_lyrics'].to(self.device)
            attention_mask_lyrics = batch['attention_mask_lyrics'].to(self.device)
            input_ids_comments = batch['input_ids_comments'].to(self.device)
            outputs = self.model(wav_embs, input_ids_lyrics, attention_mask_lyrics, input_ids_comments)
            loss_caption = outputs['loss'].mean()
            if self.args.w_contrast > 0 and epoch >= self.args.start_epoch_contrast:
                wav_embs = wav_embs.mean(dim=1) if self.args.agg == 'mean' else wav_embs[:, 0]
                lyrics_embs = outputs['encoder_hidden_states'][self.args.contrast_layer].mean(dim=1) \
                    if self.args.agg == 'mean' else outputs['encoder_hidden_states'][self.args.contrast_layer][:, 0]
                loss_contrast = self.criterion_contrast(wav_embs, lyrics_embs, music_ids).mean()
            else:
                loss_contrast = torch.tensor(0)

            loss = loss_caption + self.args.w_contrast * loss_contrast
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_caption += loss_caption.item()
            epoch_loss_contrast += loss_contrast.item()

            batch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t))
            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f"Batch: {i + 1}/{len(self.train_loader)}\tloss: {loss.item():.3f}\t"
                      f"loss_caption: {loss_caption.item():.3f}\tloss_contrast: {loss_contrast.item():.3f}\t"
                      f"batch_time:{batch_time}")
                t = time.time()

        epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_epoch))
        print(f"Epoch training time: {epoch_time}\n")

        return epoch_loss / len(self.train_loader), epoch_loss_caption / len(self.train_loader), \
               epoch_loss_contrast / len(self.train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='music4all_full',
                        choices=('music4all_full', 'music4all_positive', 'music4all_not_negative'))
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--load_ckp_path', type=str, default='')

    # data prep
    parser.add_argument('--csv_path', type=str, default='../bart-fusion/code/datasets',
                        help='where to put the csv files')
    parser.add_argument('--audio_path', type=str, default='../../zihaohe-cn/data/music4all/audios',
                        help='where to put the audio numpy files')
    parser.add_argument('--audio_len', type=int, default=15)
    parser.add_argument('--comment_len', type=int, default=200)
    parser.add_argument('--sample_rate', type=int, default=16000)

    # model
    parser.add_argument('--agg', type=str, choices=('cls', 'mean'), default='mean')

    # training config
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=160)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--w_contrast', type=float, default=0.)
    parser.add_argument('--start_epoch_contrast', type=int, default=0)
    parser.add_argument('--contrast_loss', type=str, choices=('infonce', 'positive_pair'), default='infonce')
    parser.add_argument('--contrast_layer', type=int, default=4, help='at which layer of encoder to use contrastive learning')
    parser.add_argument('--n_workers', type=int, default=8, help='# of workers in dataloader.')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    print(args)

    import socket
    hostname = socket.gethostname()
    node = hostname.split(',')[0]
    print()
    print(node)

    prefix = f'results/{args.corpus}_w_contrast={args.w_contrast}/{args.idx}'

    args_dict = args.__dict__
    with open(f'{prefix}/config.json', 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(json.dumps(args_dict, indent=2))

    solver = Trainer(args)
    solver.train()
