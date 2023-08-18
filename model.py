import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15, sparsemax


class HemorrhageDetector(nn.Module):
    def __init__(
        self,
        encoder,
        n_dim,
        hidden_size,
        embedding_dropout=0.50,
        attention_dropout=0.25,
        attention_activation="softmax",
    ):
        super(HemorrhageDetector, self).__init__()
        self.encoder = encoder
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=embedding_dropout)
        self.encoder = self.__encoder__()
        self.attention_mechanism = self.__attention_mechanism__(
            dropout=attention_dropout
        )
        self.attention_activation = None
        if attention_activation == "softmax":
            self.attention_activation = F.softmax
        if attention_activation == "sparsemax":
            self.attention_activation = sparsemax
        if attention_activation == "entmax15":
            self.attention_activation = entmax15
        self.classifier = self.__classifier__()

    def __encoder__(self):
        if "resnet" in self.encoder:
            encoder = torch.hub.load(
                "pytorch/vision:v0.9.0", self.encoder, pretrained=True
            )

            # freeze first conv layer
            for p in encoder.conv1.parameters():
                p.requires_grad = False
            # freeze first two layers
            for p in encoder.layer1.parameters():
                p.requires_grad = False
            for p in encoder.layer2.parameters():
                p.requires_grad = False

            # classification head
            num_features = encoder.fc.in_features
            encoder.fc = nn.Linear(num_features, self.n_dim)
            # randomly initialize FC layer weights, remove bias
            nn.init.kaiming_normal_(encoder.fc.weight)
            nn.init.constant_(encoder.fc.bias, 0)
        return encoder

    def __attention_mechanism__(self, dropout):
        class Attention(nn.Module):
            def __init__(self, n_dim, hidden_size, dropout):
                super(Attention, self).__init__()
                self.n_dim = n_dim
                self.hidden_size = hidden_size
                self.V = nn.Linear(self.n_dim, self.hidden_size)
                self.U = nn.Linear(self.n_dim, self.hidden_size)
                self.W = nn.Linear(self.hidden_size, 1)
                self.tanh = nn.Tanh()
                self.sigmoid = nn.Sigmoid()
                self.dropout = nn.Dropout(p=dropout)

            def forward(self, x):
                _V = self.V(x)
                _V = self.dropout(_V)

                x = self.tanh(_V)
                x = self.W(x)
                return x

        return Attention(
            n_dim=self.n_dim,
            hidden_size=self.hidden_size,
            dropout=dropout,
        )

    def __classifier__(self):
        return nn.Sequential(nn.Linear(self.n_dim, 1), nn.Sigmoid())

    def forward(self, x, attention=True, return_aux=False):
        H = self.encoder(x)
        H = self.dropout(H)
        if attention:
            A = self.attention_activation(self.attention_mechanism(H).t(), dim=1)
            z = torch.mm(A, H)
            x = self.classifier(z)
        else:
            x = self.classifier(H)
        if return_aux:
            return {"logit": x, "attention": A, "embeddings": H}
        else:
            return x
