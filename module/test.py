import torch, math, time, evaluate
import torch.nn as nn
import torch.nn.functional as F
from module.search import Search



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.batch_size = config.batch_size        
        self.vocab_size = config.vocab_size
        self.search = Search(config, self.model, tokenizer)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, label_smoothing=0.1).to(self.device)

        self.metric_name = 'BLEU'
        self.metric_module = load_metric('bleu')


    def test(self):
        greedy_metric_score, beam_metric_score = 0, 0
        with torch.no_grad():
            print(f'Test Results on {self.task.upper()}')
            for idx, batch in enumerate(self.dataloader):
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)

                greedy_pred = self.search.greedy_search(src)
                beam_pred = self.search.beam_search(src)

                greedy_metric_score += self.metric_score(greedy_pred, trg)
                beam_metric_score += self.metric_score(beam_pred, trg)

        print(f'Total Greedy Test Metric Score: {greedy_metric_score}')
        print(f'Total  Beam  Test Metric Score: {beam_metric_score}')


    
    def metric_score(self, pred, label):        
        score = self.metric_module.compute(predictions=[pred.split()], 
                                           references=[[label.split()]])['bleu']            
        return score * 100