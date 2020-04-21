import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForMaskedLM
import onmt
from torch.nn import LayerNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm


def replace_layer_norm(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == FusedLayerNorm:
            layer_norm = LayerNorm(target_attr.normalized_shape,
                                                eps=target_attr.eps,
                                                elementwise_affine=target_attr.elementwise_affine)
            # the first
            # layer_norm.weight = target_attr.weight
            # layer_norm.bias = target_attr.bias

            # the second
            # layer_norm.weight.data.copy_(target_attr.weight.data)
            # layer_norm.bias.data.copy_(target_attr.bias.data)

            # the third
            layer_norm.load_state_dict(target_attr.state_dict())

            setattr(m, attr_str, layer_norm)

    for n, ch in m.named_children():
        replace_layer_norm(ch, n)


bert_model = BertModel.from_pretrained('bert-base-uncased')
replace_layer_norm(bert_model, "Transformer")
bert_model = bert_model.cuda()


def bert_make_vecs(batch):
    # batch_size = batch.size(1)  # batch_size
    # tokens_start = torch.ones([1, batch_size], dtype=torch.int64)*onmt.Constants.BERT_CLS
    # tokens_start =tokens_start.cuda()
    # tokens_end = torch.ones([1, batch_size], dtype=torch.int64)*onmt.Constants.BERT_SEP
    # tokens_end = tokens_end.cuda()
    #
    # tokens_tensor = torch.cat((tokens_start,batch,tokens_end ),0)

    # 【sent_length, batch_size】=> [batch_size, sent_length ]

    tokens_tensor = batch.t()
    batch_size = tokens_tensor.size(0)
    segments_tensor = tokens_tensor.ne(onmt.Constants.PAD)

    # 如果你打印出来你会发现其中有一行是全部为True的
    # for i in range(batch_size):
    #     print(segments_tensor[i]==True)
    segments_tensor = segments_tensor.long()

    bert_model.eval()
    # Predict hidden states features for each layer, no backward, so no gradient

    with torch.no_grad():
        # encoded_layers is a list, 12 layers in total, for every element of the list :
        # 【batch_size, sent_len, hidden_size】
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensor)
        # combine 12 layers to make this one whole big Tensor

        # for layer in range(len(list1)):
        #     print(torch.nn.functional.cosine_similarity(list1,list2[0],dim=2))

        # token_embeddings = torch.stack(encoded_layers, dim=0)
        # 高维是0， 最低维度是-1
        bert_vecs = torch.cat(encoded_layers[-4:], dim=-1)   # 【batch_size, sent_len, hidden_size*4】

        bert_vecs = bert_vecs.cuda()
    return bert_vecs