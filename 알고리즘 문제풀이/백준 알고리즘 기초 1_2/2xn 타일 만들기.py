import sys, os

root_path = os.path.dirname(os.path.abspath(__file__))
sys.stdin = open(root_path+"\\input.txt", "r")

def DFS(v):
    if v <=3:
        return v

    if res[v] != 0:
        return res[v]

    res[v] = DFS(v-1) + DFS(v-2)

    return res[v]

n = int(input())
res = [0] * (n+1)

print(DFS(n)% 10007)
