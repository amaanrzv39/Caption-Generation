import math
import torch
import torchvision
from torch import nn
from torchvision.models import convnext_small
from transformers import GPT2Tokenizer
import yaml


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"bos_token": "<|startoftext|>", "unk_token": "<|unk|>", "pad_token": "[PAD]"})

def get_cnn_model():
    model = convnext_small(
        weights=torchvision.models.convnext.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-2])
    return model

# sinusoidal PositionalEncoding class

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, emb_size, nhead, num_decoder_layers, tgt_vocab_size, dim_feedforward, dropout, activation):
        super().__init__()
        # Embedding Layer
        self.emb_size = emb_size
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)
        # Tranformer Decoder Blocks
        self.text_decoder = nn.TransformerDecoder(
                                  nn.TransformerDecoderLayer(
                                      d_model=emb_size,
                                      nhead=nhead,
                                      dim_feedforward=dim_feedforward,
                                      dropout=dropout,
                                      activation=activation
                                    ),
                                  num_layers=num_decoder_layers
                                  )
        # Dense Layer
        self.dense = nn.Linear(emb_size, tgt_vocab_size)
        # add positional encoding
        self.pos_encoder = PositionalEncoding(emb_size, dropout=dropout)
        # initialize weights
        self.init_weights()

    def init_weights(self):
        range = 0.1
        self.embedding.weight.data.uniform_(-range, range)
        self.dense.bias.data.zero_()
        self.dense.weight.data.uniform_(-range, range)

    def forward(self, src_emb, tgt_tokens, tgt_mask, tgt_padding_mask):
        B, D, H, W = src_emb.shape
        src_emb = src_emb.reshape(B, D, -1).permute(2, 0, 1)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.emb_size)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        tgt_emb = self.pos_encoder(tgt_emb)

        outs = self.text_decoder(
            tgt_emb, src_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask
        )

        return self.dense(outs)

    # For inference
    def predict(self, img_features, tgt_tokens):
        src_emb = self.pos_encoder(img_features)
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.emb_size)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        tgt_emb = self.pos_encoder(tgt_emb)

        outs = self.text_decoder(tgt_emb, src_emb)

        return self.dense(outs)

class CaptionModel(nn.Module):
    def __init__(self, emb_size, nhead, num_decoder_layers, tgt_vocab_size, dim_feedforward, dropout, activation):
        super().__init__()

        self.image_encoder = get_cnn_model()
        self.text_decoder = TransformerDecoder(
                                emb_size,
                                nhead,
                                num_decoder_layers,
                                tgt_vocab_size,
                                dim_feedforward,
                                dropout,
                                activation,
                            )

    def forward(self, img_inp, tgt_tokens, tgt_mask, tgt_padding_mask):
        src_emb = self.image_encoder(img_inp)
        text_out = self.text_decoder(src_emb, tgt_tokens, tgt_mask, tgt_padding_mask)

        return text_out


with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


NUM_LAYERS = config['model']['num_layers']
EMB_DIM = config['model']['emb_dim']
NHEAD = config['model']['nhead']
DIM_FEEDFORWARD = config['model']['dim_feedforward']
DROPOUT = config['model']['dropout']
ACTIVATION = config['model']['activation']


model = CaptionModel(
    emb_size=EMB_DIM,
    nhead=NHEAD,
    num_decoder_layers=NUM_LAYERS,
    tgt_vocab_size=len(tokenizer),
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    activation=ACTIVATION,
)

model = model.to("mps")