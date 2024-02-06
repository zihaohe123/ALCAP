import numpy as np
import torch.nn as nn
import torch
from pytorch_metric_learning import losses


class InfoNCE(nn.Module):
    def __init__(self, temperature, dim1=256, dim2=768):
        super(InfoNCE, self).__init__()
        self.criterion = losses.NTXentLoss(temperature=temperature)
        self.temperature = temperature
        self.fc = nn.Linear(max(dim1, dim2), min(dim1, dim2))
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, embs1, embs2, music_ids=None):
        if music_ids is not None:
            music_ids = music_ids.to('cpu').numpy()
            _, indices = np.unique(music_ids, return_index=True)
            embs1 = embs1[indices]
            embs2 = embs2[indices]

        if self.dim1 > self.dim2:
            embs1 = self.fc(embs1)
        elif self.dim1 < self.dim2:
            embs2 = self.fc(embs2)
        assert embs1.shape == embs2.shape

        labels = torch.arange(embs1.shape[0], device=embs1.device)
        embs = torch.cat([embs1, embs2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        n = embs1.shape[0]
        anchors_p = torch.arange(n, device=embs1.device)
        positives = torch.arange(n, device=embs1.device) + n
        anchors_n = []
        negatives = []
        for i in range(n):
            anchors_n_ = [i] * (n-1)
            negatives_ = list(range(n))
            negatives_.remove(i)
            negatives_ = (np.array(negatives_) + n).tolist()
            anchors_n.extend(anchors_n_)
            negatives.extend(negatives_)
        anchors_n = torch.tensor(anchors_n, device=embs1.device)
        negatives = torch.tensor(negatives, device=embs1.device)

        loss = self.criterion(embs, labels, (anchors_p, positives, anchors_n, negatives))
        return loss


class PositivePairLoss(nn.Module):
    def __init__(self, dim1=256, dim2=768):
        super(PositivePairLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.fc = nn.Linear(max(dim1, dim2), min(dim1, dim2))
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, embs1, embs2, music_ids=None):
        if music_ids is not None:
            music_ids = music_ids.to('cpu').numpy()
            _, indices = np.unique(music_ids, return_index=True)
            embs1 = embs1[indices]
            embs2 = embs2[indices]

        if self.dim1 > self.dim2:
            embs1 = self.fc(embs1)
        elif self.dim1 < self.dim2:
            embs2 = self.fc(embs2)
        assert embs1.shape == embs2.shape

        sims = self.cos_sim(embs1, embs2)
        dists = 1 - sims
        return dists.mean()