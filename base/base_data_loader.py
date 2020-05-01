from torch.utils.data import DataLoader

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle = shuffle
        super().__init__(dataset=dataset,batch_size=batch_size,shuffle=shuffle)

    def _split_sampler(self, split):
        ## TODO:
        pass

    def split_validation(self):
        ## TODO:
        pass