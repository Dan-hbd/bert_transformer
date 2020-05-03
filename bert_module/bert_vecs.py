import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForMaskedLM
from torch.nn import LayerNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm
# from typing import Any
import onmt
#from apex import amp
from .scalar_mix import ScalarMix


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

class BertVec(nn.Module):
    def __init__(self,do_layer_norm=True, scalar_mix_parameters=None):
        super(BertVec,self).__init__()

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        replace_layer_norm(self.bert_model, "Transformer")
        self.bert_model = self.bert_model.cuda()
        self.num_layers = onmt.Constants.BERT_LAYERS
        # self._scalar_mixes: Any = []
        self.scalar_mix = ScalarMix(
            self.num_layers,
            do_layer_norm=do_layer_norm,
            initial_scalar_parameters=scalar_mix_parameters,
            trainable=scalar_mix_parameters is None,
        )
        # self._scalar_mixes.append(scalar_mix)


    def forward(self, batch):
        # make batch first: [batch_size, sent_length ]
        tokens_tensor = batch.t()
        tokens_tensor = tokens_tensor.cuda()
        # print("tokens_tensor", tokens_tensor[0])
        segments_tensor = tokens_tensor.ne(onmt.Constants.PAD)
        segments_tensor = segments_tensor.long()
        input_mask = tokens_tensor.ne(0).long()

        representations = []

        # Predict hidden states features for each layer, no backward, so no gradient
        self.bert_model.eval()

        with torch.no_grad():
            # encoded_layers is a list, 12 layers in total, for every element of the list :
            # 【batch_size, sent_len, hidden_size】
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensor, input_mask)
            # combine 12 layers to make this one whole big Tensor

            # for layer in range(len(list1)):
            #     print(torch.nn.functional.cosine_similarity(list1,list2[0],dim=2))

            # token_embeddings = torch.stack(encoded_layers, dim=0)
            # 高维是0， 最低维度是-1, 用最后四层
            # bert_vecs = torch.cat(encoded_layers[-4:], dim=-1)   # 【batch_size, sent_len, hidden_size*4】

            # as in the typical case
            # tensors: (batch_size, seq_len, dim)    mask : (batch_size, seq_len)

            bert_vecs = self.scalar_mix(encoded_layers, input_mask)
            # 只用最后一层，用了CLS
            # bert_vecs = encoded_layers[-1]


            bert_vecs = bert_vecs.cuda()

            # 【batch_size, sent_len-1, hidden_size】
            # print(bert_vecs.size(),bert_vecs_noClsSep.size())

        return bert_vecs
