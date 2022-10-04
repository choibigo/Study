import input_setting

n = int(input())
row = [0] * n

def check(x):
    for i in range(x):
        if row[x] == row[i] or abs(row[x] - row[i]) == abs(x - i):
            return False
    return True


answer = 0
def DFS(v):
    global answer
    if v == n:
        answer += 1
        return 

    for i in range(n):
        row[v] = i
        
        if check(v):
            DFS(v+1)
        
DFS(0)
print(answer)