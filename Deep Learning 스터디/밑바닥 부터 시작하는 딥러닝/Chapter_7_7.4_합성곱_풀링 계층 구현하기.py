import os, sys
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    - 다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    - 입력데이터를 연산할 필터의 높이, 필터의 너비, 스트라이드, 패딩을 고려해 2차원 배열로 변환한다.
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0) # (9, 75)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0) # (90, 75)
"""
필터 크기인 (3X5X5)와 변경된 데이터 cols 의 크기가 같다.
"""

class convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N,C,H,W = x.shape

        out_h = int(1 + (H+ 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W+ 2*self.pad - FW) / self.stride)

        col = im2col(x, FN, FW, self.stride, self.pad)
        col_w = self.W.reshape(FN, -1).T # (원소모든 곱, 필터 개수) 로 필터를 2차원으로 만든다.

        out = np.dot(col, col_w) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
