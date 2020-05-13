import torch
from torch.nn import LayerNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm

import onmt
from .modeling import BertModel

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

pretrained_Bert_dir = "/project/student_projects2/dhe/BERT/experiments/pytorch_bert_model"
#bert_encoder = BertModel.from_pretrained(cache_dir=pretrained_Bert_dir)
#replace_layer_norm(bert_encoder, "Transformer")

#if torch.cuda.is_available():
#    bert_model = bert_encoder.cuda()


def make_bert_vec(batch):
    # already batch first: [batch_size, sent_length ]
    tokens_tensor = batch
    segments_tensor = tokens_tensor.ne(onmt.Constants.PAD).long()
    input_mask = tokens_tensor.ne(0).long()


    bert_encoder.eval()
    with torch.no_grad():
    # encoded_layers is a list, 12 layers in total, for every element of the list :
    # 【batch_size, sent_len, hidden_size】
        encoded_layers, _ = bert_encoder(tokens_tensor, segments_tensor, input_mask)

    return encoded_layers
