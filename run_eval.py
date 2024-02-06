import os

if __name__ == '__main__':
    corpus = ['music4all_positive', 'music4all_not_negative', 'music4all_full'][0]
    csv_path = '../data/music4all/metadata'
    audio_path = '../data/music4all/audios'
    w_contrast = 2e-2
    idx = 1

    num_beams = 4
    batch_size = 12
    n_workers = 4
    seed = [2022, 2021, 2020][0]

    phases = [
        # 'val'
        'test',
    ]
    epochs = [5,
              # 10, 15, 20
              ]
    gpu = '0'

    path = f'results/{corpus}_w_contrast={w_contrast}/{idx}/gen_n_beams={num_beams}'
    if seed != 2022 or True:
        path += f'_seed={seed}'

    os.makedirs(path, exist_ok=True)

    for phase in phases:
        for epoch in epochs:
            command = f'python3 -u evaluator.py ' \
                      f'--corpus={corpus} ' \
                      f'--csv_path={csv_path} ' \
                      f'--audio_path={audio_path} ' \
                      f'--idx={idx} ' \
                      f'--phase={phase} ' \
                      f'--epoch={epoch} ' \
                      f'--num_beams={num_beams} ' \
                      f'--batch_size={batch_size} ' \
                      f'--w_contrast={w_contrast} ' \
                      f'--gpu={gpu} ' \
                      f'--n_workers={n_workers} ' \
                      f'--seed={seed} ' \
                      f'> {path}/output_{phase}_{epoch}.txt'

            print(command)
            os.system(command)