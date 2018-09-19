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

    def __init__(self, txt_encode_size=128, num_filters=64, num_channels=3):
        super().__init__()

        self.img = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_channels, num_filters, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (num_filters) x 32 x 32
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            # state size. (num_filters*2) x 16 x 16
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            # state size. (num_filters*4) x 8 x 8
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (num_filters*8) x 4 x 4
            #nn.Conv2d(num_filters * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

        self.predictor = nn.Sequential(
            nn.Linear(txt_encode_size + num_filters*8*4*4, 1),
            # TODO: don't use sigmoid?
            # might be better to use, idk
            nn.Sigmoid() 
        )

        self.apply(weights_init)

    def forward(self, img=None, sent=None):
        img = self.img(img)

        # flatten
        img = img.view(img.size(0), -1)
        sent = sent.view(sent.size(0), -1)

        # concat img + sentence
        img_plus_sent = torch.cat((img, sent), dim=1)

        # predict
        return self.predictor(img_plus_sent)

class Generator(nn.Module):
    def __init__(self, latent_size=200, num_filters=128, num_channels=3):
        super().__init__()

        self.latent_size = latent_size
        
        self.seq = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (num_filters*8) x 4 x 4
            nn.ConvTranspose2d(num_filters* 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (num_filters*4) x 8 x 8
            nn.ConvTranspose2d(num_filters* 4, num_filters* 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (num_filters*2) x 16 x 16
            nn.ConvTranspose2d(num_filters* 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_filters, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.seq(input)

def weights_init(layer):
    name = layer.__class__.__name__
    if 'Conv' in name or 'Linear' in name:
        init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
    elif 'BatchNorm' in name:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0.0)

