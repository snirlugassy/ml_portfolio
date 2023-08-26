import torch


class RRNet(torch.nn.Module):
    def __init__(self, dim, depth, activation=torch.nn.ReLU, dropout=0.3, batch_norm=True, skip_connection=True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = batch_norm
        self.use_skip_connection = skip_connection

        _mlps = []
        for _ in range(depth):
            _mlps.append(torch.nn.Linear(dim, dim))
            _mlps.append(activation())
            if self.use_batch_norm:
                _mlps.append(torch.nn.BatchNorm1d(self.dim))
            _mlps.append(torch.nn.Dropout(dropout))
        _mlps.append(torch.nn.Linear(dim, dim)) # last layer without activation
        self.mlps = torch.nn.Sequential(*_mlps)

    def forward(self, x):
        # forward pass with skip connection
        # return x + self.mlps(x)
        y = self.mlps(x)

        if self.use_skip_connection:
            y = x + y

        y = y / y.sum(dim=1).unsqueeze(1)

        return y


class RRNet2(torch.nn.Module):
    def __init__(self, dim, depth, activation=torch.nn.ReLU, dropout=0.3, batch_norm=True, skip_connection=True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = batch_norm
        self.use_skip_connection = skip_connection

        _mlps = []
        for _ in range(depth):
            _mlps.append(torch.nn.Linear(dim, dim))
            _mlps.append(activation())
            if self.use_batch_norm:
                _mlps.append(torch.nn.BatchNorm1d(self.dim))
            _mlps.append(torch.nn.Dropout(dropout))
        _mlps.append(torch.nn.Linear(dim, dim)) # last layer without activation
        self.mlps = torch.nn.Sequential(*_mlps)

    def forward(self, x):
        # forward pass with skip connection
        # return x + self.mlps(x)
        y = self.mlps(x)

        if self.use_skip_connection:
            y = x + y

        y = y / y.sum(dim=1).unsqueeze(1)

        return y
