import input_setting

n = int(input())
for i in range(n):
    temp = ' '*(n-i-1) +  '*'*(i+1)
    print(temp)