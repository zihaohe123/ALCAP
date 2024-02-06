import os
import glob
import re

if __name__ == '__main__':
    corpus = ['music4all_positive', 'music4all_not_negative', 'music4all_full'][0]

    csv_path = '../data/music4all/metadata'
    audio_path = '../data/music4all/audios'

    idx = 0     # an index for the current setting
    seed = [2022, 2021, 2020][0]
    resume_epoch = 0
    load_ckp_path = ''  # set the path if resuming from the checkpoint

    sample_rate = 16000
    audio_len = 15
    comment_len = 512
    agg = ['cls', 'mean'][1]

    batch_size = 26  # comment_len=512
    epochs = 20
    lr = 5e-5
    weight_decay = 0
    w_contrast = 2e-2
    start_epoch_contrast = 0
    eval_interval = 5
    contrast_loss = ['infonce', 'positive_pair'][0]
    contrast_layer = 4  # options: [0,1,2,3,4,5,6]. 0 -- directly after the embedding layer
    n_workers = 4

    prefix = f'results/{corpus}_w_contrast={w_contrast}'
    os.makedirs(prefix, exist_ok=True)

    gpu = '0'

    # batch time - v100
    # positive: 42s, full: 4min5s, not_negative:3min47s

    append = '>' if resume_epoch == 0 else '>>'

    if resume_epoch == 0 and load_ckp_path == '':
        folders = glob.glob(f'{prefix}/*')
        indices = []
        for folder in folders:
            idx = int(re.findall(r'\d+', folder)[-1])
            indices.append(idx)
        idx = max(indices) + 1 if indices else 1
        os.makedirs(f'{prefix}/{idx}', exist_ok=True)

    command = f'python -u trainer.py ' \
              f'--corpus={corpus} ' \
              f'--csv_path={csv_path} ' \
              f'--audio_path={audio_path} ' \
              f'--idx={idx} ' \
              f'--resume_epoch={resume_epoch} ' \
              f'--load_ckp_path={load_ckp_path} ' \
              f'--eval_interval={eval_interval} ' \
              f'--sample_rate={sample_rate} ' \
              f'--audio_len={audio_len} ' \
              f'--comment_len={comment_len} ' \
              f'--batch_size={batch_size} ' \
              f'--epochs={epochs} ' \
              f'--lr={lr} ' \
              f'--weight_decay={weight_decay} ' \
              f'--w_contrast={w_contrast} ' \
              f'--start_epoch_contrast={start_epoch_contrast} ' \
              f'--contrast_loss={contrast_loss} ' \
              f'--contrast_layer={contrast_layer} ' \
              f'--agg={agg} ' \
              f'--gpu={gpu} ' \
              f'--n_workers={n_workers} ' \
              f'--seed={seed} ' \
              f'{append} {prefix}/{idx}/output.txt'

    print(command)
    os.system(command)