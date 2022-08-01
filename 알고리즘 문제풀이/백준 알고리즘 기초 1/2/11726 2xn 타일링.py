import input_setting

import sys
n = int(input())

if n == 1:
    print(1)
    sys.exit()
elif n == 2:
    print(3)
    sys.exit()
else:
    res = [0] * (n+1)
    res[1] = 1
    res[2] = 3

    for i in range(3, n+1):
        res[i] = res[i-1] + res[i-2]*2

    
    print(res[n] % 10007)
