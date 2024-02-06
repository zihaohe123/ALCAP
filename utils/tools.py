import pandas as pd
import torch
import torch.nn as nn
import os
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models.music_encoder import CNNSA


def precompute_audio_embeddings(csv_path, audio_path, audio_len, sample_rate, device, n_workers=8):
    import pandarallel
    pandarallel.pandarallel.initialize(progress_bar=True, nb_workers=n_workers)

    df_full = pd.read_csv(f'{csv_path}/df_full.csv')
    df_test = pd.read_csv(f'{csv_path}/df_test.csv')
    df = pd.concat([df_full, df_test])
    series_music_ids = df['music4all_id']

    # compute the embeddings of the audios
    audio_arrs_path = f'{audio_path}/audio_arrs_{audio_len}_{sample_rate}'
    audio_embs_path = f'{audio_path}/audio_embs_{audio_len}_{sample_rate}'
    os.makedirs(audio_embs_path, exist_ok=True)
    files = glob.glob(f'{audio_embs_path}/*.npz')
    if len(files) == 0:

        class AudioDataset(Dataset):
            def __init__(self, music_ids):
                self.music_ids = music_ids

            def __len__(self):
                return len(self.music_ids)

            def __getitem__(self, index):
                music_id = self.music_ids[index]
                wav = np.load(f'{audio_arrs_path}/{music_id}.npz')['a']
                wav = torch.tensor(wav, dtype=torch.float)
                item = {'music_id': music_id,
                        'wav': wav}
                return item

        music_ids = series_music_ids.tolist()
        audio_dataset = AudioDataset(music_ids)
        loader = DataLoader(audio_dataset, batch_size=64, num_workers=8)
        model = CNNSA()
        model.load_state_dict(torch.load('ckp/model_cnnsa.pth'))
        model = nn.DataParallel(model)
        model.to(device)

        print('Computing audio embeddings....')
        def save_emb_func(row):
            music_id = row['music_id']
            wav_emb = row['wav_emb']
            np.savez(f'{audio_embs_path}/{music_id}.npz', a=wav_emb)

        wav_embs = []
        music_ids_ = []
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                batch_wavs = batch['wav']
                batch_music_ids = batch['music_id']
                batch_wavs = batch_wavs.to(device)
                wav_embs_ = model(batch_wavs)
                wav_embs_ = wav_embs_.detach().to('cpu').numpy()
                wav_embs.append(wav_embs_)
                music_ids_.append(batch_music_ids)

                if idx % 20 == 0:
                    print(f'{idx}/{len(loader)}')

                if len(wav_embs) * loader.batch_size >= 5000:
                    print('Saving wav embs....')
                    wav_embs = np.concatenate(wav_embs, axis=0)
                    music_ids_ = np.concatenate(music_ids_, axis=0)
                    wav_embs = list(wav_embs)
                    music_ids_ = list(music_ids_)
                    df_wav_embs = pd.DataFrame({
                        'music_id': music_ids_,
                        'wav_emb': wav_embs
                    })
                    df_wav_embs.parallel_apply(save_emb_func, axis=1)

                    wav_embs = []
                    music_ids_ = []

            print('Saving wav embs....')
            wav_embs = np.concatenate(wav_embs, axis=0)
            music_ids_ = np.concatenate(music_ids_, axis=0)
            wav_embs = list(wav_embs)
            music_ids_ = list(music_ids_)
            df_wav_embs = pd.DataFrame({
                'music_id': music_ids_,
                'wav_emb': wav_embs
            })
            df_wav_embs.parallel_apply(save_emb_func, axis=1)

        print('Done\n')