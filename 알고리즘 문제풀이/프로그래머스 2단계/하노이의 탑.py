def DFS(frm, to, mid, n, answer):
    if n == 1:
        answer.append([frm, to])
        return
    
    DFS(frm, mid, to, n-1, answer)
    answer.append([frm, to])
    DFS(mid, to, frm, n-1, answer)
    
    
def solution(n):
    
    answer = list()
    DFS(1, 3, 2, n, answer)
    
    print(answer)
    
    return answer