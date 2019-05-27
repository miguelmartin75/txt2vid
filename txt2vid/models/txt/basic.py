import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, separate_decoder=False, vocab_size=None):
        super().__init__()

        self.separate_decoder = separate_decoder
        self.encoder = RecurrentModel(vocab_size=vocab_size, is_decoder=not separate_decoder)
        if separate_decoder:
            self.decoder = RecurrentModel(vocab_size=vocab_size, is_decoder=True, bi=False)
        else:
            self.decoder = self.encoder

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder.sample(*args, **kwargs)

class RecurrentModel(nn.Module):
    def __init__(self, vocab_size=None, embed_size=256, hidden_size=256, encoding_size=256, num_layers=4, bi=True, is_decoder=False):
        super().__init__()

        self.bi = bi
        self.num_layers = num_layers
        if self.bi:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers, batch_first=True, bidirectional=self.bi)

        # decoder
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.to_vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths=None, initial_state=None, raw_output=True):
        max_len = lengths[0]
        embeddings = self.embed(x)

        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        out, hidden = self.lstm(packed, initial_state)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)

        if self.bi:
            hn = hidden[0].view(self.num_layers, 2, -1, self.hidden_size)
            hn_forward = hn[-1, 0]
            hn_back = hn[-1, 1]
            hn = torch.cat((hn_forward, hn_back), dim=1)
        else:
            hn = hidden[0].view(self.num_layers, 1, -1, self.hidden_size)[-1]

        if not raw_output:
            assert self.is_decoder
            out = self.to_vocab(out.squeeze(1))

        return out, hidden, hn

    # decode
    def sample(self, true_inputs=None, initial_hidden=None, max_seq_len=60, teacher_force=False):
        # bless this
        # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py#L35
        raw_outputs = []
        symbols = []

        # note: assumes start of true input is start char
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
        return torch.zeros(self.num_layers, 1, self.hidden_size)

if __name__ == '__main__':
    # TODO
    print("TODO")
