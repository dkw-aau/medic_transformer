from torch import nn
import torch as th


class SingleClassificationModel(nn.Module):
    def __init__(
            self,
            i_dim,
            h_dim,
            o_dim,
            n_layers,
            dropout=0,
            out_type='binary'):
        super().__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_type = out_type
        self.batch_norms = nn.ModuleList([])

        self.layers = nn.ModuleList([])

        # I - H
        self.layers.append(
            nn.Linear(self.i_dim, self.h_dim)
        )
        self.batch_norms.append(nn.BatchNorm1d(self.h_dim))

        # H - H
        for _ in range(self.n_layers):
            self.layers.append(
                nn.Linear(self.h_dim + self.i_dim, self.h_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(self.h_dim))

        # H - O
        self.out_layer = nn.Linear(self.h_dim + i_dim, self.o_dim)

        self.activation = nn.ReLU()
        self.out_act = nn.Sigmoid()

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, h):
        feats = h
        for idx, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            if idx == 0:
                h = layer(h)
            else:
                h = layer(th.cat([feats, h], dim=1))
            h = self.activation(h)
            h = self.dropout(h)
            if h.shape[0] > 1:
                h = batch_norm(h)

        output = self.out_layer(th.cat([feats, h], dim=1))

        if self.out_type == 'binary':
            output = self.out_act(output)

        return output
