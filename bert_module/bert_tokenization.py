import sys
sys.path.append('/home/dhe/hiwi/Exercises/bert/')

from pytorch_pretrained_bert import BertTokenizer
import onmt.Markdown
import argparse

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")
parser.add_argument('-test_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-test_tgt', required=True,
                    help="Path to the validation target data")
opt = parser.parse_args()


def tokenize_data(raw_data, tokenizer, lang):
    with open(raw_data, "r", encoding="utf-8") as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()
            if lang == "en":
                marked_sent = "[CLS] " + sent + " [SEP]"
            elif lang == "zh":
                marked_sent = sent
            tokenized_sent = tokenizer.tokenize(marked_sent)
            tokenized_sents.append(tokenized_sent)

    new_data = raw_data + ".bert.tok"
    with open(new_data, "w", encoding="utf-8") as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')


def main():
    tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')

    for data in [opt.train_src, opt.valid_src, opt.test_src]:
        lang = "en"
        tokenize_data(data, tokenizer_en, lang)

    for data in [opt.train_tgt, opt.valid_tgt, opt.test_tgt]:
        lang = "zh"
        tokenize_data(data, tokenizer_zh, lang)


if __name__ == "__main__":
    main()
