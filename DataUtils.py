import pandas as pd
from SentencesDataset import SentencesDataset


class DataUtils():
    @staticmethod
    def load_data(filename):
        return pd.read_csv(filename, sep="\t", header=None, names=["score", "sant1", "sant2"])

    @staticmethod
    def load_pickle(filename):
        return pd.read_pickle(filename)

    @staticmethod
    def data2pickle(filename, path_to_save):
        pd.read_csv(filename, sep="\t", header=None, names=["score", "sant1", "sant2"]).to_pickle(path_to_save)

    @staticmethod
    def process_data(data_frame):
        scores = []
        sants = []
        for i in range(len(data_frame.sant1)):
            sant1 = data_frame.sant1[i]
            sant2 = data_frame.sant2[i]
            score = data_frame.score[i]
            if type(sant1) is not float and type(sant2) is not float:
                sants1_word = str(sant1)
                sants2_word = str(sant2)
                sants.append((sants1_word, sants2_word))
                scores.append(score)
        return sants, scores

    @staticmethod
    def load_process_data(filename):
        return DataUtils.process_data(DataUtils.load_data(filename))

    @staticmethod
    def load_process_pickle(filename):
        return DataUtils.process_data(DataUtils.load_pickle(filename))

    @staticmethod
    def load_dataset(filename, pickle=False):
        if pickle:
            sants, scores = DataUtils.load_process_data(filename)
        else:
            sants, scores = DataUtils.load_process_data(filename)
        return SentencesDataset(sants, scores)