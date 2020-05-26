import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import onmt


class VariationalDropout(torch.nn.Module):
    def __init__(self, p=0.5, batch_first=False):
        super().__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or not self.p:
            return x

        if self.batch_first:
            m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)

        mask = m / (1 - self.p)
        # mask = mask.expand_as(x)

        return mask * x



def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    # X = embed._backend.Embedding.apply(words, masked_embed_weight,
        # padding_idx, embed.max_norm, embed.norm_type,
        # embed.scale_grad_by_freq, embed.sparse
    # )
    x = F.embedding(
            words, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

    return x




def switchout(words, vocab_size, tau=1.0, transpose=False, offset=0):
    """
    :param offset: number of initial tokens to be left "untouched"
    :param transpose: if the tensor has initial size of l x b
    :param words: torch.Tensor(b x l)
    :param vocab_size: vocabulary size
    :param tau: temperature control
    :return:
    sampled_words torch.LongTensor(b x l)
    """
    if transpose:
        words = words.t()

    if offset > 0:
        offset_words = words[:, :offset]
        words = words[:, offset:]

    mask = torch.eq(words, onmt.Constants.BOS) | \
        torch.eq(words, onmt.Constants.EOS) | torch.eq(words, onmt.Constants.PAD)
    # 每句话中（除去BOS等）可以用来被操作的单词个数
    lengths = (1 - mask.byte()).float().sum(dim=1)
    batch_size, n_steps = words.size()

    # first, sample the number of words to corrupt for each sent in batch
    # 第一步选定个数，一个batch 中每句话要corrupt的单词个数
    logits = torch.arange(n_steps).type_as(words).float() # size l

    logits = logits.mul_(-1).unsqueeze(0).expand_as(words).contiguous().masked_fill_(mask, -float("inf"))

    probs = torch.nn.functional.log_softmax(logits.mul_(tau), dim=1)
    # 每个位置被选中的概率，BOS等位置都是0
    probs = torch.exp(probs)
    # 每句话中要替换的单词个数
    num_words = torch.distributions.Categorical(probs).sample().float()


    # second, sample the corrupted positions
    # 第二步选定位置
    # corrupt_pos 显示一个常数矩阵，每句话中每个位置被选中的概率相同，1/（这句话中会被替换的单词个数）
    corrupt_pos = num_words.div(lengths)
    corrupt_pos = corrupt_pos.unsqueeze(1).expand_as(words).contiguous()

    # mask 中为True的位置置为0 ，把不能被选中的位置的选中概率置为0
    corrupt_pos.masked_fill_(mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()
    # 总共要操作的单词个数
    total_words = int(corrupt_pos.sum())

    # sample the corrupted values, which will be added to sents
    # 选定要替换为新的单词的value（虽然并不是最终替换的value）
    # 先创造一个长度是 total_words 的LongTensor 的随机矩阵
    corrupt_val = torch.LongTensor(total_words).type_as(words)
    # 从vocab_size 中随机挑选一个词给每一个要替换的位置
    corrupt_val = corrupt_val.random_(1, vocab_size)
    corrupts = words.clone().zero_()

    # corrupt_pos.type_as(mask): 是一个[batch, len]的tensor, 一共选中的total_words 个位置的值为True, 其余为False,
    # corrupt_val 是一个长度[total_words] 的tensor, 里面的值是选中来替换的值
    # corrupts要替换的位置的值为对应的corrupt_val 里的值，其余为0
    corrupts = corrupts.masked_scatter_(corrupt_pos.type_as(mask), corrupt_val)

    # to add the corruption and then take the remainder w.r.t the vocab size
    # 把corrupts 和 words的原来的值相加，再除以vocab_size 取余数
    # 例如原来的值是1012， 替换的值为26750， 因为1012+25738=26750 没有超过vocab_size， 则会替换为26750，否则会替换为取余数后的值
    sampled_words = words.add(corrupts).remainder_(vocab_size)

    if offset > 0:
        sampled_words = torch.cat([offset_words, sampled_words], dim=1)

    if transpose:
        sampled_words = sampled_words.t()

    return sampled_words



 
def bertmask(words, vocab_size, tau=1.0, transpose=False):
    """
    :param offset: number of initial tokens to be left "untouched"
    :param transpose: if the tensor has initial size of l x b
    :param words: torch.Tensor(b x l)
    :param vocab_size: vocabulary size
    :param tau: temperature control
    :return:
    sampled_words torch.LongTensor(b x l)
    """
    if transpose:
        words = words.t()

    mask = torch.eq(words, onmt.Constants.BOS) | \
        torch.eq(words, onmt.Constants.EOS) | torch.eq(words, onmt.Constants.PAD)
    # 每句话中（除去BOS等）可以用来被操作的单词个数
    lengths = (1 - mask.byte()).float().sum(dim=1)
    batch_size, n_steps = words.size()

    # first, sample the number of words to corrupt for each sent in batch
    # 第一步选定个数，一个batch 中每句话要corrupt的单词个数
    logits = torch.arange(n_steps).type_as(words).float() # size l
    logits = logits.mul_(-1).unsqueeze(0).expand_as(words).contiguous().masked_fill_(mask, -float("inf"))
    
    # 暂时还没有理解tau的影响，但是越大的话后面被选中的个数可能越多
    probs = torch.nn.functional.log_softmax(logits.mul_(tau), dim=1)
    # 每个位置被选中的概率，BOS等位置都是0
    probs = torch.exp(probs)
    # 每句话中要替换的单词个数
    num_words = torch.distributions.Categorical(probs).sample().float()


    # second, sample the corrupted positions
    # 第二步选定位置
    # corrupt_pos 显示一个常数矩阵，每句话中每个位置被选中的概率相同，1/（这句话中会被替换的单词个数）
    corrupt_pos = num_words.div(lengths)
    corrupt_pos = corrupt_pos.unsqueeze(1).expand_as(words).contiguous()

    # mask 中为True的位置置为0 :把不能被选中的位置的选中概率置为0
    corrupt_pos.masked_fill_(mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()

    corrupt_val = corrupt_pos * onmt.Constants.BERT_MASK
    
    corrupt_words = (1- corrupt_pos) * words
    
    mask_words = corrupt_words.add(corrupt_val)

    if transpose:
        mask_words = mask_words.t()

    return mask_words
 
