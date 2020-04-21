import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import sys

if len(sys.argv) != 4:
   print("We need 4 more arguments")
   exit()

print("hey bert, be nice, don't give me an error")
print(torch.cuda.is_available())


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def create_tokenized_data(file_raw, file_tok):
    # with open("data/valid.en","r",encoding="utf-8")  as f1:
    with open(file_raw,"r",encoding="utf-8")  as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()
            marked_sent ="[CLS] " + sent + " [SEP]"
            tokenized_sent =tokenizer.tokenize(marked_sent)
            tokenized_sents.append(tokenized_sent)

    with open(file_tok, "w", encoding="utf-8") as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')

file_path = sys.argv[1]
file_raw = sys.argv[2]
file_tok = sys.argv[3]

create_tokenized_data(file_path+'/'+file_raw,file_path+'/'+file_tok)
print(file_path+'/'+file_tok)


