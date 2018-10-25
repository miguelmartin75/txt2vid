import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size=None, embed_size=128, hidden_size=256, encoding_size=256, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        #self.to_vec = nn.Linear(hidden_size, encoding_size)
        self.to_vocab = nn.Linear(hidden_size, vocab_size)

        self.apply(weights_init)

    def forward(self, x, lengths=None, initial_state=None, raw_output=True):
        max_len = lengths[0]
        embeddings = self.embed(x)

        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        out, hidden = self.lstm(packed, initial_state)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)

        hn = out[:, -1, :]
        if not raw_output:
            out = self.to_vocab(out.squeeze(1))

        return out, hidden, hn

    # basically decode
    def sample(self, true_inputs=None, initial_hidden=None, max_seq_len=60, teacher_force=False):
        # bless this
        # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py#L35
        raw_outputs = []
        symbols = []

        inputs = true_inputs[:, 0].unsqueeze(1)
        hidden = initial_hidden
        assert inputs is not None

        for i in range(max_seq_len):
            inputs = self.embed(inputs)

            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.to_vocab(outputs.squeeze(1))
            _, predicted = outputs.max(1)

            raw_outputs.append(outputs)
            symbols.append(predicted)

            if teacher_force:
                inputs = true_inputs[:, i].unsqueeze(1)
            else:
                inputs = predicted.unsqueeze(1)

        raw_outputs = torch.stack(raw_outputs, 1)
        symbols = torch.stack(symbols, 1)
        return raw_outputs, symbols

    def create_initial_state(self):
        # TODO
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden

class Discrim(nn.Module):

    def __init__(self, txt_encode_size=256, num_filters=64, num_channels=1):
        super().__init__()

        self.vid = nn.Sequential(
            nn.Conv3d(num_channels, 64, 4, 2, 1, bias=False), # 64
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(64, 128, 4, 2, 1, bias=False), # 128
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(128, 256, 4, 2, 1, bias=False), # 128
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(256, 512, 4, 2, 1, bias=False), # 256
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),


            #nn.BatchNorm3d(1024),
            #nn.LeakyReLU(0.2, True),

            #nn.Conv3d(1024, txt_encode_size, (2, 4, 4), (1, 1, 1), 0, bias=False), # 512
            #nn.BatchNorm3d(txt_encode_size),
            #nn.LeakyReLU(0.2, True),
        )

        self.sent_map = nn.Sequential(
            nn.Linear(txt_encode_size, txt_encode_size),
            nn.BatchNorm1d(txt_encode_size),
            nn.LeakyReLU(0.2, True)
        )

        self.predictor = nn.Sequential(
            nn.Conv3d(512 + txt_encode_size, 512, (1,1,1), 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 1, (2, 4, 4), 1, 0, bias=False),
            nn.Sigmoid()
        #    nn.Linear(txt_encode_size*2, txt_encode_size),
        #    nn.BatchNorm1d(txt_encode_size),
        #    nn.LeakyReLU(0.2, True),
        #    nn.Linear(txt_encode_size, 1),

        #    nn.Sigmoid() 
        )

        self.apply(weights_init)

    def forward(self, vids=None, sent=None, device=None):
        sent = self.sent_map(sent)
        vids = self.vid(vids)

        sent_temp = torch.zeros((vids.size(0), sent.size(1), vids.size(2), vids.size(3), vids.size(4)), device=device)
        for i in range(vids.size(2)):
            for j in range(vids.size(3)):
                for k in range(vids.size(4)):
                    sent_temp[:, :, i, j, k] = sent

        sent = sent_temp

        vids_plus_sent = torch.cat((vids, sent), dim=1)

        pred = self.predictor(vids_plus_sent)
        return pred.view(-1, 1).squeeze(1)

class MotionDiscrim(nn.Module):
    def __init__(self, txt_encode_size=256):
        super().__init__()

        self.motion_map = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(512 + txt_encode_size, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        self.sent_map = nn.Sequential(
            nn.Linear(txt_encode_size, txt_encode_size),
            nn.BatchNorm1d(txt_encode_size),
            nn.LeakyReLU(0.2, True)
        )

        self.apply(weights_init)

    def forward(self, frames, sent=None, device=None):
        sent = self.sent_map(sent)
        motions = frames[1:] - frames[0:-1]

        outputs = []
        # loop through each motion 'frame'
        for i in range(motions.size(0)): 
            motion = motions[i, :, :, :, :]
            motion = self.motion_map(motion)

            sent_dupe = torch.zeros((sent.size(0), sent.size(1), motion.size(2), motion.size(3)), device=device)

            for i in range(motion.size(2)):
                for j in range(motion.size(3)):
                    sent_dupe[:, :, i, j] = sent

            motion_and_sent = torch.cat((motion, sent_dupe), dim=1)
            output = self.predictor(motion_and_sent)
            output = output.view(output.size(0), -1).squeeze(1)
            outputs.append(output)

        return torch.stack(outputs, 0).to(device)


class FrameMap(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        self.frame_map = nn.Sequential(
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.apply(weights_init)


    def forward(self, videos):
        frames = videos.permute(0, 2, 1, 3, 4)

        frames_mapped = []
        for i in range(frames.size(1)):
            frame = frames[:, i, :, :, :]
            frame = self.frame_map(frame)
            frames_mapped.append(frame)

        frames = torch.stack(frames_mapped)
        return frames


class FrameDiscrim(nn.Module):
    def __init__(self, txt_encode_size=256):
        super().__init__()

        self.frame_map = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(512 + txt_encode_size, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        self.sent_map = nn.Sequential(
            nn.Linear(txt_encode_size, txt_encode_size),
            nn.BatchNorm1d(txt_encode_size),
            nn.LeakyReLU(0.2, True)
        )

        self.apply(weights_init)

    def forward(self, frames, sent=None, device=None):
        sent = self.sent_map(sent)

        outputs = []
        # input is frame x batch x ch x h x w
        # loop through each frame
        for i in range(frames.size(0)): 
            frame = frames[i, :, :, :, :]
            frame = self.frame_map(frame)

            sent_dupe = torch.zeros(sent.size(0), sent.size(1), frame.size(2), frame.size(3), device=device)

            for i in range(frame.size(2)):
                for j in range(frame.size(3)):
                    sent_dupe[:, :, i, j] = sent

            #print('frame=', frame.size())
            #print('sent=', sent_dupe.size())
            frame_and_sent = torch.cat((frame, sent_dupe), dim=1)
            output = self.predictor(frame_and_sent)
            output = output.view(output.size(0), -1).squeeze(1)
            outputs.append(output)

        return torch.stack(outputs, 0).to(device)

class Generator(nn.Module):
    def __init__(self, latent_size=256, num_channels=1):
        super().__init__()

        self.latent_size = latent_size
        
        self.seq = nn.Sequential(
            # input is Z, going into a de-convolution
            nn.ConvTranspose3d(latent_size, 512, kernel_size=(2, 4, 4), padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(64, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm3d(num_channels),
            #nn.LeakyReLU(0.2, True),

            #nn.ConvTranspose3d(latent_size, 1024, kernel_size=(2, 4, 4), padding=0, bias=False),
            #nn.BatchNorm3d(1024),
            #nn.LeakyReLU(0.2, True),

            #nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm3d(512),
            #nn.LeakyReLU(0.2, True),

            #nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm3d(256),
            #nn.LeakyReLU(0.2, True),

            #nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm3d(128),
            #nn.LeakyReLU(0.2, True),

            #nn.ConvTranspose3d(128, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm3d(num_channels),
            #nn.LeakyReLU(0.2, True),

            nn.Tanh()
        )

        self.input_map = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2, True)
        )

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1))
        x = self.input_map(x)
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        vids = self.seq(x)
        return vids


USE_NORMAL_INIT=True

def weights_init(layer):
    name = layer.__class__.__name__
    if 'Conv' in name or 'Linear' in name:
        #global USE_NORMAL_INIT
        if USE_NORMAL_INIT:
            print("normal init")
            layer.weight.data.normal_(0.0, 0.02)
        else:
            print("xavier init")
            init.xavier_normal_(layer.weight.data)

        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
    elif 'BatchNorm' in name:
        if hasattr(layer, 'weight') and layer.weight is not None:
            layer.weight.data.normal_(1.0, 0.02)

        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.fill_(0.0)
