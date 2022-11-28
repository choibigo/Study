import input_setting

import sys
for _ in range(int(sys.stdin.readline())):
    h, w, n = map(int, sys.stdin.readline().split(' '))
    div, mod = divmod(n, h)
    div+=1

    if mod ==0:
        mod = h
    if div==0:
        div = h
    elif 0<div<10:
        div = '0'+str(div)
    print(str(mod)+str(div))
