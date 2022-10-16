import os, yaml, json
import sentencepiece as spm
from run import load_tokenizer



def read_text(f_name):
    with open(f"{f_name}", 'r') as f:
        data = f.readlines()
    return data


def save_json(data_obj, split):
    with open(f"data/{split}.json", 'w') as f:
        json.dump(data_obj, f)


def process_data(f_name):
    assert os.path.exists(f'data/{f_name}')
    with open(f"data/{f_name}", 'r') as f:
        data = f.readlines()
    #sort by lenth and do lower_case
    lowered = [seq.lower() for seq in sorted(data)]
    with open(f"data/{f_name}", 'w') as f:
        f.write(''.join(lowered))


def build_vocab(lang):
    os.system(f'cat data/*.{lang} >> data/concat.{lang}')
    assert os.path.exists(f'data/concat.{lang}')

    with open('configs/vocab.yaml', 'r') as f:
        vocab_dict = yaml.load(f, Loader=yaml.FullLoader)

    opt = f"--input=data/concat.{lang}\
            --model_prefix=data/{lang}_spm\
            --vocab_size={vocab_dict['vocab_size']}\
            --character_coverage={vocab_dict['coverage']}\
            --model_type={vocab_dict['type']}\
            --unk_id={vocab_dict['unk_id']} --unk_piece={vocab_dict['unk_piece']}\
            --pad_id={vocab_dict['pad_id']} --pad_piece={vocab_dict['pad_piece']}\
            --bos_id={vocab_dict['bos_id']} --bos_piece={vocab_dict['bos_piece']}\
            --eos_id={vocab_dict['eos_id']} --eos_piece={vocab_dict['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove(f'data/concat.{lang}')


def tokenize_data(split, src_tokenizer, trg_tokenizer):
    tokenized = list()
    src_data = read_text(f"data/{split}.src") 
    trg_data = read_text(f"data/{split}.trg")

    for src_sent, trg_sent in zip(src_data, trg_data):
        temp_dict = dict()
        temp_dict['src'] = src_tokenizer.EncodeAsIds(src_sent.strip())
        temp_dict['trg'] = src_tokenizer.EncodeAsIds(trg_sent.strip())
        tokenized.append(temp_dict)
    
    return tokenized


def main():
    name_dict = {'train.en': 'train.src',
                 'train.de': 'train.trg', 
                 'val.en': 'valid.src',
                 'val.de': 'valid.trg', 
                 'test_2016_flickr.en': 'test.src',
                 'test_2016_flickr.de': 'test.trg'}
    
    #download data
    os.system('bash download_data.sh')
    
    #process data
    for k, v in name_dict.items():
        assert os.path.exists(f'data/{k}')
        os.system(f'mv data/{k} data/{v}')
        process_data(v)
        assert os.path.exists(f'data/{v}')

    #build vocab
    assert os.path.exists('configs/vocab.yaml')
    build_vocab('src')
    build_vocab('trg')
    
    #load_tokenizers
    src_tokenizer = load_tokenizer('src')
    trg_tokenizer = load_tokenizer('trg')
    
    #tokenize datasets and save'em
    for split in ['train', 'valid', 'test']:
        tokenized_data = tokenize_data(split, src_tokenizer, trg_tokenizer)
        save_json(tokenized_data, split)
        os.remove(f"data/{split}.src")
        os.remove(f"data/{split}.trg")
        assert os.path.exists(f'data/{split}.json')


if __name__ == '__main__':
    main()
    