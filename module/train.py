import time, math, json, torch
import torch.nn as nn
import torch.amp as amp
from optim import AdamW
from optim.lr_scheduler import ReduceLROnPlateau



class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__()
        
        self.model = model
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.vocab_size = config.vocab_size

        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate        

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=2)
        self.criterion = nn.CrossEntropyLoss()
        
        self.early_stop = config.early_stop
        self.patience = config.patience        

        self.ckpt = config.ckpt
        self.record_path = f"ckpt/{config.task}/{config.model_type}.json"
        self.record_keys = ['epoch', 'train_loss', 'train_ppl', 'valid_loss', 
                            'valid_ppl', 'learning_rate', 'train_time']


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              Train PPL: {record_dict['train_ppl']:.2f}""".replace(' ' * 14, ''))

        print(f"""  >> Valid Loss: {record_dict['valid_loss']:.3f} | \
              Valid PPL: {record_dict['valid_ppl']:.2f}\n""".replace(' ' * 14, ''))


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def train(self):
        records = []
        prev_loss, best_loss = float('inf'), float('inf')
        patience = self.patience

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.lr_scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
            
            #Early Stopping Process
            if self.early_stop:
                if prev_loss > val_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = val_loss

            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def get_loss(self, logit, label):
        return self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )


    def train_epoch(self):
        self.model.train()
        tot_len = len(self.train_dataloader)
        epoch_loss = 0


        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            label = batch['label'].to(self.device)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                logit = self.model(x, y)
                loss = self.get_loss(logit, label)                
                loss = loss / self.iters_to_accumulate
            
            #Backward Loss
            self.scaler.scale(loss).backward()        
            
            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3) 

        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in self.valid_dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device) 
                label = batch['label'].to(self.device)
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    logit = self.model(x, y)
                    loss = self.get_loss(logit, label)

                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        
        return epoch_loss, epoch_ppl