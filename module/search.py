import torch, operator
import torch.nn.functional as F
from queue import PriorityQueue
from collections import namedtuple



class Search:
    def __init__(self, config, model, tokenizer):
        super(Search, self).__init__()
        
        self.beam_size = 4
        self.model = model
        self.task = config.task

        self.tokenizer = tokenizer
        self.device = config.device

        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id
        
        self.max_len = 500
        self.max_repeat = 5

        self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'length'])


    def get_score(self, node, min_length=5, alpha=1.2):            
        repeat_penalty = 1
        pred = node.pred.tolist()
        
        for idx in range(len(pred) - self.max_repeat):
            exceed = len(set(pred[idx: idx + self.max_repeat])) == 1
            if exceed and pred[idx] != self.pad_id:
                repeat_penalty = -1
                break
        
        len_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        score = node.log_prob / len_penalty
        score = score * repeat_penalty
        return score


    def get_nodes(self):
        Node = self.Node
        nodes = PriorityQueue()
        start_tensor = torch.LongTensor([[self.bos_id]]).to(self.device)

        start_node = Node(prev_node = None,
                          pred = start_tensor,
                          log_prob = 0.0,
                          length = 0)

        for _ in range(self.beam_size):
            nodes.put((0, start_node))
                    
        return Node, nodes, [], []    


    def process_input(self, input_seq):
        if self.task == 'sum':
            max_len = 0
            seq_list, input_tensor = [], []

            for seq in input_seq:
                if seq:
                    tok_seq = self.tokenizer.encode(seq)
                    seq_list.append(tok_seq)
                    if max_len < len(tok_seq):
                        max_len = len(tok_seq)

            for seq in seq_list:
                pad_len = max_len - len(seq)
                input_tensor.append(seq + ([self.pad_id] * pad_len))
            
            input_tensor = torch.LongTensor(input_tensor)

        else:
            input_tokens = self.tokenizer.encode(input_seq)
            input_tensor = torch.LongTensor(input_tokens)

        return input_tensor.unsqueeze(0).to(self.device)


    def beam_search(self, input_seq):
        Node, nodes, end_nodes, top_nodes = self.get_nodes()
        input_tensor = self.process_input(input_seq)

        if self.task == 'sum':
            seq_mask, e_mask = self.model.enc_mask(input_tensor)
            memory = self.model.encoder(input_tensor, seq_mask, e_mask)
        else:
            e_mask = self.model.enc_mask(input_tensor)
            memory = self.model.encoder(input_tensor, e_mask)        

        for t in range(self.max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]
            
            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[:, -1].item() == self.eos_id and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue

                d_input = curr_node.pred 
                d_mask = self.model.dec_mask(d_input)
                d_out = self.model.decoder(d_input, memory, e_mask, d_mask)
                out = self.model.fc_out(d_out)[:, -1]
                
                logits, preds = torch.topk(out, self.beam_size)
                logits, preds = logits, preds
                log_probs = -F.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[:, k].unsqueeze(0)
                    log_prob = log_probs[:, k].item()
                    pred = torch.cat([curr_node.pred, pred], dim=-1)           
                    
                    next_node = Node(prev_node = curr_node,
                                     pred = pred,
                                     log_prob = curr_node.log_prob + log_prob,
                                     length = curr_node.length + 1)
                    next_score = self.get_score(next_node)                
                    nodes.put((next_score, next_node))
                
                if (not t) or (len(end_nodes) == self.beam_size):
                    break

        if len(end_nodes) == 0:
            _, top_node = nodes.get()
        else:
            _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
        
        beam_out = top_node.pred.squeeze(0).tolist()
        return self.tokenizer.decode(beam_out)      
    

    def greedy_search(self, input_seq):
        input_tensor = self.process_input(input_seq)

        output_seq = [[self.pad_id  if i else self.bos_id for i in range(self.max_len)]]
        output_tensor = torch.LongTensor(output_seq).to(self.device)

        if self.task == 'sum':
            seq_mask, e_mask = self.model.enc_mask(input_tensor)
            memory = self.model.encoder(input_tensor, seq_mask, e_mask)
        else:
            e_mask = self.model.enc_mask(input_tensor)
            memory = self.model.encoder(input_tensor, e_mask)        

        for i in range(1, self.max_len):
            d_mask = self.model.dec_mask(output_tensor)
            out = self.model.decoder(output_tensor, memory, e_mask, d_mask)
            out = self.model.fc_out(out)
            
            pred = out[:, i].argmax(-1)
            output_tensor[:, i] = pred

            if pred.item() == self.eos_id:
                break

        return self.tokenizer.decode(output_tensor.squeeze(0).tolist())