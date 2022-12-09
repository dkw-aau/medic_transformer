class LOSDataset:
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels
        self.n = data.shape[0]

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.data[idx], self.labels[idx], idx]