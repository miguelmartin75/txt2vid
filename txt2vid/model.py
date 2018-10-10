import torch
import torch.nn as nn
import torch.nn.init as init

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size=None, embed_size=256, hidden_size=256, encoding_size=128, num_layers=5):
        super().__init__()

        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.ltsm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.to_vec = nn.Linear(hidden_size, encoding_size)

        self.apply(weights_init)

    def forward(self, x, lengths, states=None):
        max_len = lengths[0]
        embeddings = self.embed(x)

        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        packed_out, _ = self.ltsm(packed, states)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)

        hn = out[:, -1, :]

        return self.to_vec(hn)


class Discrim(nn.Module):

    def __init__(self, txt_encode_size=256, num_filters=64, num_channels=3):
        super().__init__()

        self.vid = nn.Sequential(
            nn.Conv3d(3, 64, 4, 2, 1, bias=False), # 64
            nn.BatchNorm3d(num_filters),
            nn.ReLU(True),

            nn.Conv3d(64, 128, 4, 2, 1, bias=False), # 128
            nn.BatchNorm3d(num_filters * 2),
            nn.ReLU(True),

            nn.Conv3d(128, 256, 4, 2, 1, bias=False), # 256
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.Conv3d(256, 512, 4, 2, 1, bias=False), # 512
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.Conv3d(512, 256, 4, 2, 1, bias=False), # 512
            nn.BatchNorm3d(256),
            nn.ReLU(True),
        )

        self.predictor = nn.Sequential(
            nn.Linear(txt_encode_size + 1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.ReLU(True),
            # TODO: don't use sigmoid?
            # might be better to use, idk
            nn.Sigmoid() 
        )

        self.apply(weights_init)

    def forward(self, vids=None, sent=None):
        vids = self.vid(vids)

        # flatten
        vids = vids.view(vids.size(0), -1)
        sent = sent.view(sent.size(0), -1)

        # concat img + sentence
        vids_plus_sent = torch.cat((vids, sent), dim=1)

        # predict
        return self.predictor(vids_plus_sent)

class Generator(nn.Module):
    def __init__(self, latent_size=356):
        super().__init__()

        self.latent_size = latent_size
        
        self.seq = nn.Sequential(
            # input is Z, going into a de-convolution
            nn.ConvTranspose3d(latent_size, 512, kernel_size=(2, 4, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(512, 256, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(256, 128, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(128, 64, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(64, 3, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(0.2, True),

            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.seq(input)

def weights_init(layer):
    name = layer.__class__.__name__
    #if 'Conv' in name or 'Linear' in name:
    #    init.xavier_normal_(layer.weight.data)
    #    if layer.bias is not None:
    #        layer.bias.data.fill_(0.0)
    #elif 'BatchNorm' in name:
    if 'Conv' in name or 'Linear' in name:
        if hasattr(layer, 'weight') and layer.weight is not None:
            layer.weight.data.normal_(1.0, 0.02)

        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.fill_(0.0)

