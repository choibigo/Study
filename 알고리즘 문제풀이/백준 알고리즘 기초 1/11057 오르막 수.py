import input_setting

n = int(input())

res = [[0 for _ in range(10)] for _ in range(n)]
res[0] = [1] * 10

for count in range(1, n):
    for i in range(10):
        if i==0:
            res[count][i] = sum(res[count-1])
        else:
            res[count][i] = (res[count][i-1] - res[count-1][i-1]) 

print(sum(res[n-1])% 10007)
