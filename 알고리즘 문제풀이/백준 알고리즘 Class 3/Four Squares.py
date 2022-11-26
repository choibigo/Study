import input_setting

'''

1 2 3 4 5 6 7 8 9
1 2 3 1 2 3 4 2 1

'''
n = int(input())
res = [0] * (n+1)
res[1] = 1
flag = 1

for i in range(2, n+1):
    if int(i**0.5) == i:
        res[i] = 1

    elif 

