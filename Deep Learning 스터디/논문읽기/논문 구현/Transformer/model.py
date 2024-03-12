import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_seq, d_hidn):
    """
    sequence 개수 + 1, hidden 차원을 입력으로 함

    1. 각 Position 별로 angle 값을 구한다.
    2. 구해진 angle 중 짝수 index의 값에 대한 sin값을 구한다.
    3. 구해진 angle 중 홀수 index의 값에 대한 cos값을 구한다.

    ※ 추가 조사 필요(https://velog.io/@gibonki77/DLmathPE)
    """
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn//2) / d_hidn)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # odd index cos

    return sinusoid_table

def get_attn_decoder_mask(seq):
    """
    - Masked Multi-Head Attention과 Multi-Head Attention은 Attention mask를 제외하고 모두 동일하다.
    tensor([[False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False,  True,  True]])
    tensor([[0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]])
    tensor([[False,  True,  True,  True,  True,  True,  True,  True],
            [False, False,  True,  True,  True,  True,  True,  True],
            [False, False, False,  True,  True,  True,  True,  True],
            [False, False, False, False,  True,  True,  True,  True],
            [False, False, False, False, False,  True,  True,  True],
            [False, False, False, False, False, False,  True,  True],
            [False, False, False, False, False, False,  True,  True],
            [False, False, False, False, False, False,  True,  True]])
    """

    subsequent_mask = torch.ones_like(seq).unsqueeze(1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)

    return subsequent_mask

""" attention pad mask """
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # hidden dimension => feed forward 지정 dim => hidden dimension
        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)

        # (bs, n_seq, d_ff, d_hidn)
        output = self.conv2(output).transpose(1,2)
        output = self.dropout(output)

        # (bs, n_seq, d_hidn)
        return output


class ScaledDotproductAttention(nn.Module):
    """
    - query, key, value 가 입력된다.
    - attention mask가 있으며 Decoder에서 사용한다.
    - K,V 가 같고 Q가 다른 경우 Attention 이다. (다른 feature와 비교하여 어떤 Feature에 집중해야 하는지 학습)
    - Q, K, V가 모두 동일한 경우 self-attention이라 한다.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head**0.5) # Q와 K의 내적 연산을 차원으로 나누는 값

    def forward(self, Q, K, V, attn_mask):
        
        """
        attnd_mask 인경우 0(pad) 부분이 True이며 pad 부분 score를 0으로 만든다.
        """

        # (bs, n_head, n_q_seq, n_k_seq), 배치 사이즈 X head 수 x query 길이 x key 길이
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9) # mask를 씌운다, 없으면 그냥 Attention

        # (bs, n_head, n_q_seq, n_k_seq), 배치 사이즈 X head 수 x query 길이 x key 길이
        attn_prob = nn.Softmax(dim=-1)(scores) # score의 텐서를 소프트 맥스 함수를 사용하여 어텐션 확률로 변환
        """
        attnd_layer = nn.Softmax(dim=-1)
        attnd_prob = attnd_layer(scores)
        """
        attn_prob = self.dropout(attn_prob)

        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)

        # context: (bs, n_head, n_q_seq, d_v)
        # attn_prob: (bs, n_head, n_q_seq, n_k_seq)
        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 적절한 Q, K, V 를 만들기 위해 Weight를 학습하는 Layer
        # d_hidn: Embedding된 차원수
        # n_head: head의 개수
        # d_head: head의 차원
        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)

        # 1개의 Scaled Dot-Product Attention Layer
        self.scaled_dot_attn = ScaledDotproductAttention(self.config)

        # (head 수 * head 차원) => 다시 Attention 입력을 위해서 d_hidn으로 만든다.
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attnd_mask):
        batch_size = Q.size(0)

        # (bs, n_head, n_k_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        
        # (bs, n_head, n_q_seq, n_k_seq)
        '''
        print(attn_mask.size()) # torch.Size([2, 8, 8])
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        print(attn_mask.size()) # torch.Size([2, 2, 8, 8])
        '''
        attnd_mask = attnd_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # context: (bs, n_head, n_q_seq, d_head)
        # context: (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self. scaled_dot_attn(q_s, k_s, v_s, attnd_mask)

        # (bs, n_q_seq, n_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)

        # (bs, n_q_seq, d_hidn)
        output = self.linear(context)
        output = self.dropout(output)

        # output: (bs, n_q_seq, d_hidn) => 입력 Q와 동일한 차원이 됨
        # attnd_prob: (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):

        # multi-head-self-attention
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs) # residual 

        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs) # residual 

        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token Embedding을 위한 Embedding Layer
        # vocab의 차원 => hidden 차원
        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        
        # Position Embedding
        # Position Embedding을 위한 Table
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq +1, self.config.d_hidn))
        # sinusoid_table로 초기화 하며, 학습 때는 사용하지 않는다.
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        # 실제 Encoder 부분, Multi-Head Attention 연산의 연속이다.
        # 설정 대로 n개의 Layer로 구성한다.
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, inputs):
        """
        - 데이터가 입력 됬을때 forward하는 부분
        - Token이 입력 된다.
        """

        # Position Embedding
        # inputs.size(1) : 입력 과 동일한 크기를 갖는 position을 구한다.
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous()+1
        # input중 0(pad)인 값을 찾는다.
        pos_mask = inputs.eq(0)
        # 입력이 0(pad)인 부분에 position 값을 0으로 만든다.
        positions.masked_fill_(pos_mask, 0)
        
        # 입력 token을 Embedding한 feature와 Position Embedding을 더한다.
        # (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions)

        # (bs, n_enc_seq, n_enc_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)


        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, attn_probs



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        # output에 대한 multi-head-self-attention
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)

        # encoder feature와 attention 연산
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_enc_seq)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)

        # position-wise-feedforward layer
        # (bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)

        return ffn_outputs, self_attn_prob, dec_enc_attn_prob
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidn))
        
        # Position Embedding
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        # N개의 Decoder Layer
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (bs, n_dec_seq, d_hidn)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)

        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        # (bs, n_dec_seq, n_dec_seq)
        # pad과련 mask와 다음 token을 보지 못하게 하는 mask 두개 모두
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        # (bs, n_dec_seq, n_enc_seq)
        # Key, query에 대한 mask 구함 => 다른 경우 mask가 다르게 되야 한다.
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        
        # (bs, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]S
        return dec_outputs, self_attn_probs, dec_enc_attn_probs
    

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)

         # (bs, n_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # (bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    
class MovieClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_dec_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)
        # (bs, d_hidn)
        dec_outputs, _ = torch.max(dec_outputs, dim=1)

        # 긍정/부정 으로 output 채널이 2이다.
        logits = self.projection(dec_outputs)

        # (bs, n_output), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    
    def save(self, epoch, loss, score, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "score": score,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"], save["score"]