import torch
import torch.nn as nn
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer, MixedEncoder
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.RelativeTransformer.Layers import SinusoidalPositionalEmbedding
from bert_module.modeling import BertModel
from bert_module.bert_vecs import replace_layer_norm

init = torch.nn.init

MAX_LEN = onmt.Constants.max_position_length  # This should be the longest sentence from the dataset


def build_model(opt, dicts):

    model = None
    
    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'
        
    if not hasattr(opt, 'layer_norm'):
        opt.layer_norm = 'slow'
        
    if not hasattr(opt, 'attention_out'):
        opt.attention_out = 'default'
    
    if not hasattr(opt, 'residual_type'):
        opt.residual_type = 'regular'

    if not hasattr(opt, 'input_size'):
        opt.input_size = 40

    if not hasattr(opt, 'init_embedding'):
        opt.init_embedding = 'xavier'

    if not hasattr(opt, 'ctc_loss'):
        opt.ctc_loss = 0

    if not hasattr(opt, 'encoder_layers'):
        opt.encoder_layers = -1

    if not hasattr(opt, 'fusion'):
        opt.fusion = False

    if not hasattr(opt, 'cnn_downsampling'):
        opt.cnn_downsampling = False

    if not hasattr(opt, 'switchout'):
        opt.switchout = 0.0

    if not hasattr(opt, 'variational_dropout'):
        opt.variational_dropout = False

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    if not opt.fusion:
        model = build_tm_model(opt, dicts)
    else:
        model = build_fusion(opt, dicts)

    return model


def build_tm_model(opt, dicts):

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        # by me
        # len_max 是否要修改
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    # BUILD GENERATOR
    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    # BUILD EMBEDDING
    if 'src' in dicts:
        # embedding_src = nn.Embedding(dicts['src'].size(),
        #                              opt.model_size,
        #                              padding_idx=onmt.Constants.PAD)

        # by me 我们用bert的词向量作为embedding, 如果bert的词向量维度和transformer的词向量维度不一致，我们做线性转换
        if onmt.Constants.BERT_HIDDEN != opt.model_size:
            bert_linear = nn.Linear(onmt.Constants.BERT_HIDDEN, opt.model_size)
        else:
            bert_linear = None

    else:
        embedding_src = None

    if opt.join_embedding and embedding_src is not None:
        embedding_tgt = embedding_src
        print("* Joining the weights of encoder and decoder word embeddings")
    else:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)

    if opt.ctc_loss != 0:
        generators.append(onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size() + 1))

    if opt.model == 'transformer':
        onmt.Constants.init_value = opt.param_init

        if opt.encoder_type == "text":
            encoder = TransformerEncoder(opt, bert_linear, positional_encoder, opt.encoder_type)
            # 这里 bert_model_dir 可以是pytorch提供的预训练模型，也可以是经过自己fine_tune的bert
            bert_model = BertModel.from_pretrained(cache_dir=opt.bert_model_dir, pretrained_model =opt.pretrained_model_name)
            replace_layer_norm(bert_model, "Transformer")

        else:
            print ("Unknown encoder type:", opt.encoder_type)
            exit(-1)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, attribute_embeddings=None)

        model = Transformer(bert_model, encoder, decoder, nn.ModuleList(generators))


    elif opt.model == 'relative_transformer':
        from onmt.modules.RelativeTransformer.Models import RelativeTransformer
        positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        # if opt.encoder_type == "text":
        # encoder = TransformerEncoder(opt, embedding_src, positional_encoder, opt.encoder_ty   pe)
        # encoder = RelativeTransformerEncoder(opt, embedding_src, relative_positional_encoder, opt.encoder_type)
        if opt.encoder_type == "audio":
            raise NotImplementedError
            # encoder = TransformerEncoder(opt, None, positional_encoder, opt.encoder_type)
        generator = nn.ModuleList(generators)
        model = RelativeTransformer(opt, [embedding_src, embedding_tgt], positional_encoder, generator=generator)

    else:
        raise NotImplementedError

    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    if opt.encoder_type == "audio":

        if opt.init_embedding == 'xavier':
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    else:
        if opt.init_embedding == 'xavier':
            if model.encoder.word_lut:
                init.xavier_uniform_(model.encoder.word_lut.weight)
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            if model.encoder.word_lut:
                init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def init_model_parameters(model, opt):

    # currently this function does not do anything
    # because the parameters are locally initialized
    pass


def build_language_model(opt, dicts):

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    from onmt.modules.LSTMLM.Models import LSTMLMDecoder, LSTMLM

    decoder = LSTMLMDecoder(opt, dicts['tgt'])

    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    model = LSTMLM(None, decoder, nn.ModuleList(generators))

    if opt.tie_weights:
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def build_fusion(opt, dicts):

    # the fusion model requires a pretrained language model
    print("Loading pre-trained language model from %s" % opt.lm_checkpoint)
    lm_checkpoint = torch.load(opt.lm_checkpoint, map_location=lambda storage, loc: storage)

    # first we build the lm model and lm checkpoint
    lm_opt = lm_checkpoint['opt']

    lm_model = build_language_model(lm_opt, dicts)

    # load parameter for pretrained model
    lm_model.load_state_dict(lm_checkpoint['model'])

    # main model for seq2seq (translation, asr)
    tm_model = build_tm_model(opt, dicts)

    from onmt.modules.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model
