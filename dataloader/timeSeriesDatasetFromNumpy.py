from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDatasetFromNumpy(Dataset):
    def __init__(self, data, label, sequence_length, transform=None):
        self.data = data
        self.label = label
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        outDate = self.data[idx:idx+self.sequence_length] 
        if self.transform:
            outDate = self.transform(outDate)
        label = np.mean(self.label[idx:idx+self.sequence_length])
        return (outDate, label)
