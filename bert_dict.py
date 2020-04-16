import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

src_ids = list(tokenizer.vocab.values())
src_values = list(tokenizer.vocab.keys())
src_dict_size = len(src_ids)


def main():
    file_vocab = "/project/student_projects2/dhe/Bert/saves/bert.word2ids.en"
    with open(file_vocab, "w", encoding="utf-8")  as f_dict:
        for i in range(src_dict_size):
            word2id = str(src_values[i]) + " " + str(src_ids[i])
            f_dict.write(word2id)
            f_dict.write('\n')
if __name__ == "__main__":
    main()









