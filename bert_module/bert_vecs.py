import torch
import torch.nn as nn
from pytorch_pretrained_bert import  BertTokenizer 
from torch.nn import LayerNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm
import argparse

import onmt
from .modeling import BertModel  
#from apex import amp
from .scalar_mix import ScalarMix


#def add_args(parser):
#    parser.add_argument('--bert-model-name', default='bert-base-uncased', type=str)
#    parser.add_argument('--warmup-from-nmt', action='store_true', )
#    parser.add_argument('--warmup-nmt-file', default='checkpoint_nmt.pt', )
#    parser.add_argument('--encoder-bert-dropout', action='store_true',)
#    parser.add_argument('--encoder-bert-dropout-ratio', default=0.25, type=float)
#    return parser
#
#parser = argparse.ArgumentParser(description='bert.py')
#parser = add_args(parser)
#args = parser.parse_args()
#
#DEFAULT_MAX_SOURCE_POSITIONS = 1024
#DEFAULT_MAX_TARGET_POSITIONS = 1024
#
#args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
#args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS


def replace_layer_norm(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == FusedLayerNorm:
            layer_norm = LayerNorm(target_attr.normalized_shape,
                                   eps=target_attr.eps,
                                   elementwise_affine=target_attr.elementwise_affine)

            layer_norm.load_state_dict(target_attr.state_dict())

            setattr(m, attr_str, layer_norm)

    for n, ch in m.named_children():
        replace_layer_norm(ch, n)

print("build bert_encoder")
model_dir = "/project/student_projects2/dhe/BERT/experiments/pytorch_bert_model"
bert_model = BertModel.from_pretrained(cache_dir=model_dir)
#replace_layer_norm(bert_model, "Transformer")

if torch.cuda.is_available():
    bert_model = bert_model.cuda()


def make_bert_vec(batch):
    # already batch first: [batch_size, sent_length ]
    tokens_tensor = batch
    segments_tensor = tokens_tensor.ne(onmt.Constants.PAD).long()
    input_mask = tokens_tensor.ne(0).long()


    bert_model.eval()
    with torch.no_grad():
    # encoded_layers is a list, 12 layers in total, for every element of the list :
    # 【batch_size, sent_len, hidden_size】
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensor, input_mask)

    return encoded_layers
