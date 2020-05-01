from base import BaseDataLoader

class FGNetDataLoader(BaseDataLoader):
    def __init__(dataset, batch_size, shuffle=True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
