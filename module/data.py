import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']
        return src, trg


class Collator(object):
    def __init__(self, config):
        self.pad_id = config.pad_id

    def __call__(self, batch):
        src_batch, trg_batch = [], []
        
        for src, trg in batch:
            src_batch.append(torch.LongTensor(src))
            trg_batch.append(torch.LongTensor(trg))
        
        src_batch = pad_sequence(src_batch,
                                 batch_first=True,
                                 padding_value=pad_id)
        
        trg_batch = pad_sequence(trg_batch, 
                                 batch_first=True, 
                                 padding_value=pad_id)
        
        return {'src': src_batch, 
                'trg': trg_batch}


def load_dataloader(config, split):

    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False,
                      collate_fn=Collator(config),
                      num_workers=2)