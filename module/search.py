import torch, operator
import torch.nn.functional as F
from itertools import groupby
from queue import PriorityQueue
from collections import namedtuple



class Search:
    def __init__(self, config, model):
        super(Search, self).__init__()
        
        self.model = model
        self.device = model.device

        self.max_len = 512
        self.beam_size = 4

        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id
        
        self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'length'])


    def get_score(self, node, max_repeat=5, min_length=5, alpha=1.2): 
        if not node.log_prob:
            return node.log_prob

        #find max number of consecutively repeated tokens
        repeat = max([sum(1 for token in group if token != self.pad_id) for _, group in groupby(node.pred.tolist())])

        repeat_penalty = 0.5 if repeat > max_repeat else 1
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



    def beam_search(self, input_tensor):
        Node, nodes, end_nodes, top_nodes = self.get_nodes()

        src_pad_mask = torch.zeros_like(input_tensor, dtype=torch.bool).to(self.device)
        memory = self.model.encoder(input_tensor, src_key_padding_mask=src_pad_mask)

        for t in range(self.max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]
            
            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[:, -1].item() == self.eos_id and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue

                d_input = curr_node.pred 
                d_pad_mask = torch.zeros_like(d_input, dtype=torch.bool).to(self.device)
                d_mask = self.generate_square_subsequent_mask(d_input.size(1))
                d_out = self.model.decoder(d_input, memory, tgt_mask=d_mask, 
                                           tgt_key_padding_mask=d_pad_mask,
                                           memory_key_padding_mask=src_pad_mask)
                out = self.model.generator(d_out)[:, -1]
                
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
        
        return top_node.pred.squeeze(0)
    

    def greedy_search(self, input_tensor):
        output_tensor = torch.LongTensor([[self.bos_id]]).to(self.device)

        src_pad_mask = torch.zeros_like(input_tensor, dtype=torch.bool).to(self.device)
        memory = self.model.encoder(input_tensor, src_key_padding_mask=src_pad_mask)        

        
        for i in range(1, self.max_len):
            #Masking
            trg_pad_mask = torch.zeros_like(output_tensor, dtype=torch.bool).to(self.device)
            trg_mask = self.generate_square_subsequent_mask(output_tensor.size(1))

            #Decoding and Generating
            dec_out = self.model.decoder(output_tensor, memory, tgt_mask=trg_mask, 
                                         tgt_key_padding_mask=trg_pad_mask, 
                                         memory_key_padding_mask=src_pad_mask)

            logit = self.model.generator(dec_out)
            
            next_token = logit[:, -1].argmax(-1).unsqueeze(0)
            output_tensor = torch.cat([output_tensor, next_token], dim=1)

            if next_token == self.eos_id:
                break

        return output_tensor.squeeze(0)


    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)