import input_setting

import sys
for _ in range(int(sys.stdin.readline())):
    h, w, n = map(int, sys.stdin.readline().split(' '))
    div, mod = divmod(n, h)

    if mod ==0:
        a = str(h)
        b = str(div)
    else:
        a = str(mod)
        b = str(div+1)

    if len(b)<2:
        b = '0'+b
        
    print(f"{a}{b}")
