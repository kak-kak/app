import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDatasetFromCsv(Dataset):
    def __init__(self, file_name, sequence_length):
        self.data = pd.read_csv(file_name).values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.sequence_length, :-1],
                self.data[idx+self.sequence_length, -1])
