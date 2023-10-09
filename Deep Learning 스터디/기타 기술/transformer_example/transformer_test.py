import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import math

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

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

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

model_path = r"D:\workspace\Difficult\my_repository\Deep Learning 스터디\기타 기술\transformer_example\transformer_german_to_english.pt"

model.load_state_dict(torch.load(model_path))
test_loss = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')

# 번역 함수
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    
    # 시작, 끝 토큰 붙이기
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    if logging:
        print(f"전체 소스 토큰: {tokens}")

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    if logging:
        print(f"소스 문장 인덱스: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # 소스 문장에 따른 마스크 생성
    src_mask = model.make_src_mask(src_tensor)

    # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 처음에는 토큰 하나만 가지고 있도록 하기(시작 토큰 1개)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # 출력 문장에 따른 마스크 생성
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # 출력 문장에서 가장 마지막 단어만 사용
        pred_token = output.argmax(2)[:,-1].item() # softmax 결과중 가장 큰 값을 갖는 index를 찾고 1차원으로 바꾼후 item값 가져옴 
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # eos_token을 만나면 끝 
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 출력 단어 인덱스를 실제 단어로 반환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째는 제외하고 출력 문장 반환
    return trg_tokens[1:], attention


example_idx = 10

src = vars(test_dataset.examples[example_idx])['src']
trg = vars(test_dataset.examples[example_idx])['trg']

print(f'소스 문장: {src}')
print(f'타겟 문장: {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device, logging=True)

print("모델 출력 결과:", " ".join(translation))


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    # 출력할 그림 크기 조절
    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # 어텐션(Attention) 스코어 확률 값을 이용해 그리기
        # head의 i 번째 attention을 꺼냄
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + [''] + [t.lower() for t in sentence] + [''], rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()
    plt.close()

display_attention(src, translation, attention)


from torchtext.data.metrics import bleu_score

def show_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len, logging=False)

        # 마지막  토큰 제거
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

        index += 1
        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{len(data)}]")
            print(f"예측: {pred_trg}")
            print(f"정답: {trg}")

    bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Total BLEU Score = {bleu*100:.2f}')

    individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
    individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
    individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])

    print(f'Individual BLEU1 score = {individual_bleu1_score*100:.2f}') 
    print(f'Individual BLEU2 score = {individual_bleu2_score*100:.2f}') 
    print(f'Individual BLEU3 score = {individual_bleu3_score*100:.2f}') 
    print(f'Individual BLEU4 score = {individual_bleu4_score*100:.2f}') 

    cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
    cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
    cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])

    print(f'Cumulative BLEU1 score = {cumulative_bleu1_score*100:.2f}') 
    print(f'Cumulative BLEU2 score = {cumulative_bleu2_score*100:.2f}') 
    print(f'Cumulative BLEU3 score = {cumulative_bleu3_score*100:.2f}')
    print(f'Cumulative BLEU4 score = {cumulative_bleu4_score*100:.2f}') 

# show_bleu(test_dataset, SRC, TRG, model, device)