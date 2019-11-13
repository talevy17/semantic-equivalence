from torch.utils.data import Dataset
import torch
import numpy as np
from bert_serving.client import BertClient

embedded_dim = 300
print("Connecting to pretrained BERT model server...\n")
bc = BertClient(ip='127.0.0.1')
print("Connected\n")


def encoder(sentence):
    # return [w2v.wv.vocab[word].index for word in sentence if word in w2v.wv.vocab]
    # return torch.tensor(encoded, dtype=torch.long)
    return bc.encode([sentence]).tolist()


class SentencesDataset(Dataset):

    def __init__(self, sentences, labels):
        self.texts = []
        self.labels = []
        for text, label in zip(sentences, labels):
            sats, lab = self.endoce(text, label)
            if sats is None or label is None:
                continue
            self.texts.append(sats)
            self.labels.append(lab)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    # text = self.texts[idx]
    # dim = len(text[0][0])
    # # print(encoder(self.w2v, text[0]).shape)
    # # print(encoder(self.w2v, text[1]).shape)
    # # print(np.zeros(dim).reshape(1, dim))
    # data = []
    #
    # sat1 = encoder(self.w2v, text[0])
    # sat2 = encoder(self.w2v, text[1])
    #
    # x = len(sat1)
    # data.append(x)
    # data.extend(sat1.copy())
    # data.extend(sat2.copy())
    #
    # #data = np.concatenate(data) # np.concatenate((sat1 ,np.zeros(dim).reshape(1, dim), sat2), axis=0)
    # label = self.labels[idx]
    # return torch.tensor(data), torch.tensor(label, dtype=torch.float)

    def endoce(self, text, label):
        # print(encoder(self.w2v, text[0]).shape)
        # print(encoder(self.w2v, text[1]).shape)
        # print(np.zeros(dim).reshape(1, dim))
        data = []

        sat1 = encoder(text[0])[0]
        sat2 = encoder(text[1])[0]
        if len(sat1) is 0 or len(sat2) is 0:
            return None, None
        x = len(sat1)
        data.append(x)
        data.extend(sat1.copy())
        data.extend(sat2.copy())
        return torch.tensor(data), torch.tensor(label, dtype=torch.float)
