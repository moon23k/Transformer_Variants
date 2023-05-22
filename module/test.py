import torch, math, time, evaluate
from tqdm import tqdm
from module.search import Search
from transformers import BertModel, BertTokenizerFast



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.search = Search(config, self.model)
        
        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = evaluate.load('bleu')

        elif self.task == 'dialog':
            self.metric_name = 'BERT'
            self.metric_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.metric_model = BertModel.from_pretrained('bert-base-uncased')
            self.metric_model.eval()

        elif self.task == 'sum':
            self.metric_name = 'ROUGE'
            self.metric_module = evaluate.load('rouge')



    def test(self):
        self.model.eval()
        tot_len, greedy_score, beam_score = 0, 0, 0

        with torch.no_grad():

            print(f'Test Results on {self.task.upper()}')
            for batch in tqdm(self.dataloader):
            
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)
                tot_len += src.size(0)
        
                greedy_pred = self.search.greedy_search(src)
                beam_pred = self.search.beam_search(src)
                
                greedy_score += self.metric_score(greedy_pred, trg)
                beam_score += self.metric_score(beam_pred, trg)
        
        greedy_score = round(greedy_score/tot_len, 2)
        beam_score = round(beam_score/tot_len, 2)
        
        return greedy_score, beam_score
        


    def metric_score(self, pred, label):

        pred = self.tokenizer.decode(pred)
        label = self.tokenizer.decode(label.tolist())

        #For Translation and Summarization Tasks
        if self.task != 'dialog':
            self.metric_module.add_batch(predictions=pred, references=[[l] for l in label])
            if self.task == 'nmt':
                score = self.metric_module.compute()['bleu']
            elif self.task == 'sum':        
                score = self.metric_module.compute()['rouge2']

        #For Dialogue Generation Task
        elif self.task == 'dialog':
            encoding = self.metric_tokenizer(pred, label, padding=True, truncation=True, return_tensors='pt')
            bert_out = self.metric_model(**encoding)[0]

            normalized = torch.nn.functional.normalize(bert_out[:, 0, :], p=2, dim=-1)
            dist = normalized.matmul(normalized.T)
            sim_matrix = dist.new_ones(dist.shape) - dist
            score = sim_matrix[0, 1].item()

        return (score * 100)
