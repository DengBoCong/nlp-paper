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
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=enc_units, bidirectional=True)
        self.fc = nn.Linear(enc_units * 2, dec_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        inputs = self.embedding(inputs)
        dropout = self.dropout(inputs)
        outputs, state = self.gru(dropout)
        # 这里使用了双向GRU，所以这里将两个方向的特征成合并起来，维度将会是units * 2
        state = torch.tanh(self.fc(torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)))
        return outputs, state # [40, 2, 2048] [2, 1024]


class BahdanauAttention(nn.Module):
    def __init__(self, units: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=2 * units, out_features=units)
        self.W2 = nn.Linear(in_features=units, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor]:
        #[2, 1024] [40, 2, 2048]
        values = values.permute(1, 0, 2) # [2, 40, 2048]
        hidden_with_time_axis = torch.unsqueeze(query, 1) # [2, 1, 1024]
        score = self.V(torch.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        )) #[2, 40, 1]
        attention_weights = F.softmax(score, 1) # [2, 40, 1]
        context_vector = attention_weights * values # [2, 40, 2048]
        context_vector = torch.sum(context_vector, dim=1) # [2, 2048]

        return context_vector, attention_weights # [2, 2048]  [2, 40, 1]

# class Attention(nn.Module):
#     """
#     普通的注意力机制
#     """
#
#     def __init__(self, enc_units: int, dec_units: int, dim: int):
#         super(Attention, self).__init__()
#         self.attn_in = (enc_units * 2) + dec_units
#         self.attn = nn.Linear(self.attn_in, dim)
#
#     def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor) -> torch.Tensor:
#         length = enc_outputs.shape[0]
#         repeated_decoder_hidden = dec_hidden.unsqueeze(1).repeat(1, length, 1)
#         enc_outputs = enc_outputs.permute(1, 0, 2)
#
#         energy = torch.tanh(self.attn(torch.cat(repeated_decoder_hidden, enc_outputs, dim=2)))
#         attention = torch.sum(energy, dim=2)
#         return F.softmax(attention, dim=1)


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
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=(enc_units * 2) + embedding_dim, hidden_size=dec_units)
        self.fc = nn.Linear(in_features=(enc_units * 3) + embedding_dim, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

    # def _weighted_encoder_rep(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor) -> torch.Tensor:
    #     """
    #     通过使用attention计算输入的encoder——outputs的权重
    #     Args:
    #         dec_hidden: decoder的隐藏层
    #         enc_outputs: encoder的输出
    #     Returns:
    #         weighted_encoder_rep: 计算输入的encoder——outputs的权重
    #     """
    #     attention = self.attention(dec_hidden, enc_outputs)
    #     attention = attention.unsqueeze(1)
    #
    #     enc_outputs = enc_outputs.permute(1, 0, 2)
    #     weighted_encoder_rep = torch.bmm(attention, enc_outputs)
    #     weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
    #
    #     return weighted_encoder_rep

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, enc_output: torch.Tensor) -> Tuple[torch.Tensor]:
        # print(inputs.shape)
        # print(hidden.shape)
        # print(enc_output.shape)
        # exit(0)

        #[2] [2, 1024] [40, 2, 2048]
        inputs = inputs.unsqueeze(0) #[1, 2]
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # [2, 2048]  [2, 40, 1]
        embedding = self.dropout(self.embedding(inputs)) #[1, 2, 256]
        gru_inputs = torch.cat((embedding, torch.unsqueeze(context_vector, dim=0)), dim=-1)#[1, 2, 2304]
        output, dec_state = self.gru(gru_inputs, hidden.unsqueeze(0)) # [1, 2, 1024] [1, 2, 1024]
        embedding = embedding.squeeze(0)#[2, 256]
        output = output.squeeze(0)#([2, 1024]
        context_vector = context_vector#[2, 2048]
        output = self.fc(torch.cat((embedding, context_vector, output), dim=-1)) # [2, 300]

        return output, dec_state.squeeze(0)  # [2, 300]  [2, 1024]
        # b,s,d  b,1,u
        # inputs = inputs.unsqueeze(0)
        # embedded = self.dropout(self.embedding(inputs))
        #
        # attention = self.attention(hidden, enc_output)
        # attention = attention.unsqueeze(1)
        # enc_output = enc_output.permute(1, 0, 2)
        # attention_weight = torch.bmm(attention, enc_output)
        # attention_weight = attention_weight.permute(1, 0, 2)
        #
        # gru_input = torch.cat((embedded, attention_weight), dim=2)
        # output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # embedded = embedded.squeeze(0)
        # output = self.fc(torch.cat((output, attention_weight, embedded), dim=1))
        #
        # return output, hidden.squeeze(0)


# class Seq2Seq(nn.Module):
#     def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor, teacher_forcing_ratio: float=0.5) -> torch.Tensor:
#         batch_size = input.shape[1]
#         max_len = target.shape[0]
#         vocab_size = self.decoder.vocab_size
#
#         outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
#         encoder_outputs, hidden = self.encoder(input)
#         output = target[0, :]
#
#         for i in