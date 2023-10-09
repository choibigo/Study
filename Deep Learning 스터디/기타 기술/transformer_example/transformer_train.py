import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

spacy_en = spacy.load('en_core_web_sm') # 영어 Tokenization
spacy_de = spacy.load('de_core_news_sm') # 독일어 Tokenization

# region spacy_test
# tokenized = spacy_en.tokenizer("I am a graduate student.") # 문장을 토큰으로 바꿔준다.

# for i, token in enumerate(tokenized):
#     print(f"인덱서 {i} : {token.text}")
# endregion

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            batch_first=True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            batch_first=True)
'''
- init_token: 시작 토큰
- eos_token: 종료 토큰
- lower: 모든 문자 소문자 화 여부
- batch_first: 미니 배치 차원을 맨 앞으로 하여 데이터륿 불러올 것인지 여부(False가 기본), Transformer를 학습할 때는 Batch가 맨앞으로 오게 하는것이 기본이다.
'''


train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

'''
!wget -P ~/.torchtext/cache/Multi30k https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz
!wget -P ~/.torchtext/cache/Multi30k https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz
!wget -P ~/.torchtext/cache/Multi30k https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz
위 코드 안되면 위 링크에서 직접 다운로드
'''

print(f"Train Dataset Length: {len(train_dataset.examples)}")
print(f"Validation Dataset Length: {len(valid_dataset.examples)}")
print(f"Test Dataset Length: {len(test_dataset.examples)}")

# print(vars(train_dataset.examples[30])['src'])
# print(vars(train_dataset.examples[30])['trg'])
# print(vars(train_dataset[30]))

SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)
# 최소 2번 이상 나온 단어들로 Vocabulary만들기

print(f"len(SRC): {len(SRC.vocab)}")
print(f"len(TRG): {len(TRG.vocab)}")

print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
print(TRG.vocab.stoi[""]) # : 2
print(TRG.vocab.stoi[""]) # : 3
print(TRG.vocab.stoi["hello"])
print(TRG.vocab.stoi["world"])

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

# 일반적인 Data loader의 Iterator와 유사하게 사용가느하다.
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_dataset, test_dataset, valid_dataset), batch_size=BATCH_SIZE, device=device)

for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg

    print(f"첫 번째 배치 크기: {src.shape}") 
    # torch.Size([128, 24])이다.
    # 해당 배치 데이터중 가장 긴 문장의 길이가 24인 것이다.
    # 따라서 해당 배치에서는 24길이(단어수)를 갖는 문장 128개가 1개 Batch인 것이다.
    # 다른 Batch에서는 가장 긴 문장의 길이가 달라지고 shape 상에선 shpae[1]=col이 달라진다.

    # 현재 배치에 있는 하나의 문장에 포함된 정보 출력
    for i in range(src.shape[1]):
        print(f"인덱스 {i}: {src[0][i].item()}") # 여기에서는 [Seq_num, Seq_len]

    # 첫 번째 배치만 확인
    break


import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0 # 정확히 나눠지도록 설계해야 head의 dimension을 잡을 수 있다.

        self.hidden_dim = hidden_dim # 임베딩 차원
        self.n_heads = n_heads # head개수: 서로 다른 attention의 개수
        self.head_dim = hidden_dim // n_heads # 각 head의 Embedding 차원, head의 차원은 hidden_dim//n_heads이다 이를 통해 단순 concat으로 연결할 수 있다.

        self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query에서 적용할 Weight
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key에서 적용할 Weight
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value에서 적용할 Weight

        self.fc_o = nn.Linear(hidden_dim, hidden_dim) # 최종 Output Layer

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) # attention 연산후 softmax로 입력 전에 나눠줄 Scale값

    def forward(self, query, key, value, mask=None):
        
        batch_size = query.shape[0]
        # ()_len: sequence의 길이를 의미한다.
        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]
    
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q: [batch_size, query_len, hidden_dim]
        # k: [batch_size, key_len, hidden_dim]
        # v: [batch_size, value_len, hidden_dim]

        # hidden_dim => n_head X head_dim 형태로 변경
        # n_heads개의 서로 다른 Attention 학습을 유도한다.
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q: [batch_size, n_head, query_len, head_dim]
        # k: [batch_size, n_head, key_len, head_dim]
        # v: [batch_size, n_head, value_len, head_dim]
        # 해당 구현에서는 query_len, key_len, value_len이 embedding차원과 동일한 값

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # permute는 transpose와 비슷한 것으로 transpose는 2개 차원에 대해서만 연산하지만 permute는 다수의 차원에 대해서도 연산할 수 있다.
        # energy: [batch_size, n_head, query_len, key_len]

        if mask is not None:
            # 마스크 값이 있다면 0인 부분을 -e10으로 채우기 => softmax값이 0이 되도록 만든다.
            energy = energy.masked_fill(mask==0, -1e10)
        
        # Attention Score 계산: 각각의 단어에 대한 유사도를 계산한다.
        attention = torch.softmax(energy, dim=-1)
        # attention: [batch_size, n_heads, query_len, key_len]

        # scaled Dot-product Attention을 계산: value값과 연산해 실제로 Score를 구한다.
        x = torch.matmul(self.dropout(attention), V)
        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0,2,1,3).contiguous()
        # x: [batch_size, query_len, n_head, head_dim]
        # contiguous는 permute연산으로 텐서의 메모리 주소가 연속적이지 않게 될 수 있다, 그러므로 contiguous()을 통해 메모리를 인접하게 만들어 준다.

        x = x.view(batch_size, -1, self.hidden_dim)
        # x: [batch_size, query_len, hidden_dim] # head_dim과 n_head가 합쳐(concat)져서 hidden_dim이 됨

        x = self.fc_o(x) 
        # 최종 output forward
        # x: [batch_size, query_len, hidden_dim]

        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        
        x = self.fc_1(x)
        # x: [batch_size, seq_len, pf_dim]

        x = self.dropout(x)
        # x: [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)
        # x: [batch_size, seq_len, hidden_dim]

        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim) # attention layer에 대한 Normalization
        self.ff_layer_norm = nn.LayerNorm(hidden_dim) # Feedforward Layer에 대한 Normalization
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device) # Self-Attention Layer
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio) # positionwise_feedforward Layer
        self.dropout = nn.Dropout(dropout_ratio)

    # 하나의 임베딩이 복제되어 Query, Key, Value로 입력되는 방식
    def forward(self, src, src_mask):
        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]

        # self attention
        # 필요한 경우 Mask 행렬을 이용하여 Attention할 단어를 조절가능하다, Encoder단 에서는 <pad>에 대해서만 Mask를 씌운다.
        # Self-Attention의 의 key, value, query는 자기 자신이 된다
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # self_attention에 dropout을 통과 시킨 이후 연산전 입력과 더한다 이를 통해 Residual 연산을 구현한다.
        # 이후 결과를 Normalization한다.
        # src: [batch_size, src_len, hidden_dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src: [batch_size, src_len, hidden_dim]

        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()
        
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hidden_dim) # 원핫 인코딩 형태의 입력을 hidden_dim 차원으로 Embedding한다.
        self.pos_embedding = nn.Embedding(max_length, hidden_dim) # 문장 최대 단어수에 대한 위치정보를 hidden_dim 차원으로 Embedding한다.
        # embedding 은 앞에 input demension이 작아도 알아서 max_length로 내부적으로 변경되어서 사용하나봄

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)]) # n_layers만큼의 Encoder Layer를 쌓는다.

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device) # 차원에 대해 square root를 구한다.
    
    def forward(self, src, src_mask):
        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, src_len]

        # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # 모든 인코더 레이어를 차례대로 거치면서 forward 수행
        for encoder_layer in self.layers:
            src = encoder_layer(src, src_mask)
        # src: [batch_size, src_len, hidden_dim]

        return src # 마지막 레이어 출력반환
    

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim) # encoder과 Decoder self-attention의 attention output에 대한 Normalization 
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # self attention
        # 자기 자신에 대한 Attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection, normalization
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg: [batch_size, trg_len, hidden_dim]

        # encoder와 attention(query=trg 데이터, key,value=Encoder의 결과)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection, normalization
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg: [batch_size, trg_len, hidden_dim]
        
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual connection, normalization
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, trg_len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg: [batch_size, trg_len, hidden_dim]

        for layer in self.layers:
            # 소스 마스크와 타겟 마스크 모두 사용
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        output = self.fc_out(trg)
        # output: [batch_size, trg_len, output_dim]

        return output, attention


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 소스 문장의 토큰에 대해서 mask 값을 0으로 변경
    def make_src_mask(self, src):
        # src: [batch_size, src_len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask: [batch_size, 1, 1, src_len]

        return src_mask

     # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    def make_trg_mask(self, trg):
        # trg: [batch, trg_len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # padding에 대한 mask
        # trg_pad_mask: [batch_size, 1, 1, trg_len]

        trg_len = trg.shape[1]
        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
        """

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        # trg_sub_mask: [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask: [batch_size, 1, trg_len, trg_len]

        return trg_mask
    
    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)
        # enc_src: [batch_size, src_len, hidden_dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output: [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return output, attention


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

# 파라미터 초기화
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# model.apply(initialize_weights)

# 학습 및 평가 함수 정의
# Adam optimizer로 학습 최적화
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 뒷 부분의 패딩(padding)에 대해서는 값 무시
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# Train 함수 정의
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스()는 제외
        # 입력을 할 때는 부터 시작하도록 처리
        output, _ = model(src, trg[:,:-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # output: [배치 크기 * trg_len - 1, output_dim]


        # 출력 단어의 인덱스 0()은 제외
        trg = trg[:,1:].contiguous().view(-1)
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        # criterion은 cross entropy로 softmax함수가 내부에 있어 trg와 계산 가능하다.
        loss = criterion(output, trg)
        loss.backward() # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 출력 단어의 마지막 인덱스()는 제외
            # 입력을 할 때는 부터 시작하도록 처리
            output, _ = model(src, trg[:,:-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0()은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


import math
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import time
import math
import random

N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time() # 시작 시간 기록

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time() # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_german_to_english.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')

model.load_state_dict(torch.load('transformer_german_to_english.pt'))

test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')