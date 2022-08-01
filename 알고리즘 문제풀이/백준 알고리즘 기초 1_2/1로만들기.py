import sys, os

root_path = os.path.dirname(os.path.abspath(__file__))
sys.stdin = open(root_path+"\\input.txt", "r")

n = int(input())

res = [sys.maxsize] * (n+1)

res[1] = 0

for index in range(1, n+1):
    for j in [index+1, index*2, index*3]:
        if j > n:
            break
        else:
            if res[index] + 1 < res[j]:
                res[j] = res[index] + 1

    
print(res[-1])