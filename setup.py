import os, re, json, yaml, argparse
from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def load_data(task):
    if task == 'nmt':
        data = load_dataset(
            'wmt14', 'de-en', split='train'
        )['translation']

    elif task == 'dialog':
        loaded_data = load_dataset('daily_dialog')
        data = loaded_data['train']['dialog'] + \
               loaded_data['validation']['dialog'] + \
               loaded_data['test']['dialog']

    elif task == 'sum':
        loaded_data = load_dataset('cnn_dailymail', '3.0.0')

        data = []
        for split in ['train', 'validation', 'test']:
            for elem in loaded_data[split]:
                data.append({'article': elem['article'], 
                             'highlights': elem['highlights']})
                
    return data



#NMT
def process_nmt(orig_data, volumn=101100):
    min_len = 10 
    max_len = 300
    max_diff = 50

    volumn_cnt = 0
    corpus, processed = [], []
    
    for elem in orig_data:
        temp_dict = dict()
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict['src'] = src
            temp_dict['trg'] = trg
            processed.append(temp_dict)
            corpus.append(src)
            corpus.append(trg)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    with open('data/nmt/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    return processed



#Dialog
def process_dialog(orig_data):
    corpus, processed = [], []
    src_list, trg_list = [], []

    for dial in orig_data:
        dial_list = []
        dial_turns = len(dial)

        if max([len(d) for d in dial]) > 300:
            continue
        
        for uttr in dial:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr).strip().lower()
            if len(_uttr) > 300:
                break
            dial_list.append(_uttr)
        
        if dial_turns < 2:
            continue

        elif dial_turns == 2:
            src_list.append(dial_list[0])
            trg_list.append(dial_list[1])
            continue  #To avoid duplicate on below condition

        #Incase of dial_turns is even
        elif dial_turns % 2 == 0:
            src_list.extend(dial_list[0::2])
            trg_list.extend(dial_list[1::2])

            src_list.extend(dial_list[1:-1:2])
            trg_list.extend(dial_list[2::2])
        
        #Incase of dial_turns is odds
        elif dial_turns % 2 == 1:
            src_list.extend(dial_list[0:-1:2])
            trg_list.extend(dial_list[1::2])
            
            src_list.extend(dial_list[1::2])
            trg_list.extend(dial_list[2::2])   


    assert len(src_list) == len(trg_list)
    for src, trg in zip(src_list, trg_list):
        temp_dict = dict()
        temp_dict['src'] = src
        temp_dict['trg'] = trg
        
        corpus.append(src)
        corpus.append(trg)
        processed.append(temp_dict)

        
    with open('data/dialog/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))
    
    return processed    



#Sum
def process_sum(orig_data, volumn=101100):    
    volumn_cnt = 0
    corpus, processed = [], []
    min_len, max_len = 500, 3000


    for elem in orig_data:
        src, trg = elem['article'], elem['highlights']

        if min_len < len(src) < max_len:
            if len(trg) < min_len:
                
                #Lowercase
                src, trg = src.lower(), trg.lower()

                #Remove unnecessary characters in trg sequence
                trg = re.sub(r'\n', ' ', trg)                 #remove \n
                trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

                processed.append({'src': src, 'trg': trg})
                corpus.append(src)
                corpus.append(trg)

                #End Condition
                volumn_cnt += 1
                if volumn_cnt == volumn:
                    break

    with open('data/sum/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))
    
    return processed           



def train_tokenizer(task):
    corpus_path = f'data/{task}/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(WordPiece(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"data/{task}/tokenizer.json")



def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{task}/{key}.json')




def main(task):
    #Prerequisite
    os.makedirs(f'data/{task}', exist_ok=True)

    #Load Original Data
    orig = load_data(task)

    #PreProcess Data
    if task == 'nmt':
        processed = process_nmt(orig)
    elif task == 'dialog':
        processed = process_dialog(orig)
    elif task == 'sum':
        processed = process_sum(orig)        

    #Train Tokenizer
    train_tokenizer(task)

    #Save Data
    save_data(task, processed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    
    args = parser.parse_args()
    assert args.task in ['all', 'nmt', 'dialog', 'sum']
    
    if args.task == 'all':
        for task in ['nmt', 'dialog', 'sum']:
            main(task)
    else: 
        main(args.task)    