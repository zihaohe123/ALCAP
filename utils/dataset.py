from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pickle
import os


class SongCommentDataset(Dataset):
    def __init__(self, phase, csv_path, audio_path, name='', audio_len=15, comment_len=200, sample_rate=16000,
                 n_workers=8):
        if phase != 'test':
            df = pd.read_csv(f'{csv_path}/df_{name}_{phase}.csv')
        else:
            df = pd.read_csv(f'{csv_path}/df_{phase}.csv')

        music_ids = df['music4all_id'].tolist()
        lyrics = df['lyrics'].tolist()
        comments = df['comment'].tolist()

        music_ids_unique = df['music4all_id'].drop_duplicates().tolist()
        print(f'{len(music_ids)} instances')

        # tokenize lyrics and comments
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        os.makedirs('cache', exist_ok=True)
        if phase != 'test':
            encodings_path = f'cache/music4all_encodings_{name}_{phase}_{comment_len}.pkl'
        else:
            encodings_path = f'cache/music4all_encodings_test_{comment_len}.pkl'

        print('Tokenizing lyrics and comments....')
        if not os.path.exists(encodings_path):
            encodings_lyrics = tokenizer(lyrics, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encodings_comments = tokenizer(comments, padding=True, truncation=True, max_length=comment_len,
                                           return_tensors='pt')
            pickle.dump((encodings_lyrics, encodings_comments), open(encodings_path, 'wb'))
        else:
            encodings_lyrics, encodings_comments = pickle.load(open(encodings_path, 'rb'))
        print('Done\n')
        input_ids_lyrics = encodings_lyrics['input_ids']
        attention_mask_lyrics = encodings_lyrics['attention_mask']
        input_ids_comments = encodings_comments['input_ids']
        attention_mask_comments = encodings_comments['attention_mask']

        print('Loading the audio embeddings into memory....')
        audio_embs_path = f'{audio_path}/audio_embs_{audio_len}_{sample_rate}'
        series_music_ids_unique = pd.Series(music_ids_unique)
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=False, nb_workers=n_workers)
        series_audio_embs = series_music_ids_unique.parallel_apply(lambda x : np.load(f'{audio_embs_path}/{x}.npz')['a'])
        music_id2wav_emb = dict(zip(series_music_ids_unique.tolist(), series_audio_embs.tolist()))
        print('Done\n')

        self.audio_path = audio_path
        self.audio_len = audio_len
        self.sample_rate = sample_rate
        self.music_ids = music_ids
        self.lyrics = lyrics
        self.input_ids_lyrics = input_ids_lyrics
        self.attention_mask_lyrcis = attention_mask_lyrics
        self.comments = comments
        self.input_ids_comments = input_ids_comments
        self.attention_mask_comments = attention_mask_comments
        self.music_id2wav_emb = music_id2wav_emb

    def __len__(self):
        return len(self.music_ids)

    def __getitem__(self, index):
        music_id = self.music_ids[index]
        wav_emb = self.music_id2wav_emb[music_id]
        lyrics = self.lyrics[index]
        input_ids_lyrics = self.input_ids_lyrics[index]
        attention_mask_lyrics = self.attention_mask_lyrcis[index]
        comment = self.comments[index]
        input_ids_comments = self.input_ids_comments[index]
        attention_mask_comments = self.attention_mask_comments[index]

        item = {
            'music_id': music_id,
            'wav_emb': wav_emb,
            'lyrics': lyrics,
            'input_ids_lyrics': input_ids_lyrics,
            'attention_mask_lyrics': attention_mask_lyrics,
            'comment': comment,
            'input_ids_comments': input_ids_comments,
            'attention_mask_comments': attention_mask_comments
        }

        return item


def data_loader(phase, batch_size, csv_path, audio_path, name='', audio_len=15, comment_len=200,
                sample_rate=16000, n_workers=8):
    shuffle = True if phase == 'train' else False
    drop_last = False if phase == 'train' else True
    dataset = SongCommentDataset(phase, csv_path, audio_path, name, audio_len, comment_len, sample_rate, n_workers)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, pin_memory=True,
                        drop_last=drop_last)
    return loader


def data_loader_train_static(phase, batch_size, csv_path, audio_path, name='', audio_len=15, comment_len=200,
                sample_rate=16000, n_workers=8):
    dataset = SongCommentDataset(phase, csv_path, audio_path, name, audio_len, comment_len, sample_rate, n_workers)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True,
                        drop_last=False)
    return loader
