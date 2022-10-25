def solution(k, dungeons):
    check = [False] * len(dungeons)
    answer = 0
    def DFS(next_k, count):
        nonlocal answer
        answer = max(count, answer)
        for i in range(len(dungeons)):
            if check[i] == False and next_k >= dungeons[i][0]:
                check[i] = True
                DFS(next_k-dungeons[i][1], count+1)
                check[i] = False
                
    DFS(k, 0)

    return answer