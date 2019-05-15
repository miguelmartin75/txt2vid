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

if __name__ == '__main__':
    # TODO
    print("TODO")
