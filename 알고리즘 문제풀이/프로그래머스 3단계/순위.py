from collections import defaultdict

def solution(n, results):
    
    lose_graph = defaultdict(set)
    win_graph = defaultdict(set)
    
    for win, lose in results:
        lose_graph[win].add(lose)
        win_graph[lose].add(win)
        
    for i in range(1, n+1):
        for loser in lose_graph[i]:
            win_graph[loser].update(win_graph[i])
            
        for winer in win_graph[i]:
            lose_graph[winer].update(lose_graph[i])
    
    answer = 0
    for i in range(1, n+1):
        if len(win_graph[i])+len(lose_graph[i]) == n-1:
            answer +=1
    return answer