from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        print(self.lstm.parameters())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # print("output: ", lstm_out.shape)
        # print("h: ", h.shape)
        # print("c: ", c.shape)
        return lstm_out


class LinearOne(nn.Module):
    def __init__(self, input, output):
        super(LinearOne, self).__init__()
        self.linear = nn.Linear(input, output)

    def forward(self, x):
        return self.linear(x).squeeze(1)
