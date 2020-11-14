import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    seq2seq的encoder，主要就是使用Embedding和GRU对输入进行编码
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, dec_units: int, dropout: float):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=enc_units, bidirectional=True)
        self.fc = nn.Linear(enc_units * 2, dec_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        inputs = self.embedding(inputs)
        dropout = self.dropout(inputs)
        outputs, state = self.gru(dropout)
        # 这里使用了双向GRU，所以这里将两个方向的特征成合并起来，维度将会是units * 2
        state = torch.tanh(
            self.fc(torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)))
        return outputs, state


class BahdanauAttention(nn.Module):
    def __init__(self, units: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=2 * units, out_features=units)
        self.W2 = nn.Linear(in_features=units, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor]:
        values = values.permute(1, 0, 2)
        hidden_with_time_axis = torch.unsqueeze(query, 1)
        score = self.V(torch.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))
        attention_weights = F.softmax(score, 1)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    """
    seq2seq的decoder，将初始化的inputs、隐藏层和encoder的输出作为
    输入，encoder的输入用来和隐藏层进行attention，得到的上下文
    向量和x进行整合然后丢到gru里去，最后Linear输出一下
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int,
                 dec_units: int, dropout: float, attention: nn.Module):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=(enc_units * 2) +
                          embedding_dim, hidden_size=dec_units)
        self.fc = nn.Linear(in_features=(enc_units * 3) +
                            embedding_dim, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, enc_output: torch.Tensor) -> Tuple[torch.Tensor]:
        inputs = inputs.unsqueeze(0)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        embedding = self.dropout(self.embedding(inputs))
        gru_inputs = torch.cat(
            (embedding, torch.unsqueeze(context_vector, dim=0)), dim=-1)
        output, dec_state = self.gru(gru_inputs, hidden.unsqueeze(0))
        embedding = embedding.squeeze(0)
        output = output.squeeze(0)
        context_vector = context_vector
        output = self.fc(
            torch.cat((embedding, context_vector, output), dim=-1))

        return output, dec_state.squeeze(0)
