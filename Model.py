import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,
                 freeze_embeddings=False):
        super(Siamese, self).__init__()
        # remove the cnn

                # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 64, 10),  # 64@96*96
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),  # 64@48*48
        #     nn.Conv2d(64, 128, 7),
        #     nn.ReLU(),    # 128@42*42
        #     nn.MaxPool2d(2),   # 128@21*21
        #     nn.Conv2d(128, 128, 4),
        #     nn.ReLU(), # 128@18*18
        #     nn.MaxPool2d(2), # 128@9*9
        #     nn.Conv2d(128, 256, 4),
        #     nn.ReLU(),   # 256@6*6
        # )
        # TODO: add lstm or a simple rnn or whatever
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=freeze_embeddings)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 5)

    def lstm_forword(self, input_sentence, batch_size=None):
        pred = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        pred = pred.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = torch.zeros(1, self.batch_size, self.hidden_size)  # Initial hidden state of the LSTM
            c_0 = torch.zeros(1, self.batch_size, self.hidden_size)  # Initial cell state of the LSTM
        else:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
            c_0 = torch.zeros(1, batch_size, self.hidden_size)
        output, (final_hidden_state, final_cell_state) = self.lstm(pred, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])
        return final_output

    def forward_one(self, x):
        x = self.lstm_forword(x)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return F.softmax(out)
        # return out

