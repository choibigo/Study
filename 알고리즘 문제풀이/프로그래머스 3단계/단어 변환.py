from collections import deque

def check(word1, word2):
    count = 0
    for w1, w2 in zip(word1, word2):
        if w1 != w2:
            count +=1
        if count >1 : 
            return False
    return True
            
def solution(begin, target, words):

    if target not in words:
        return 0
    
    graph = {key:[] for key in words}
    visited = {key:False for key in words}
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if check(words[i], words[j]):
                graph[words[i]].append(words[j])
                graph[words[j]].append(words[i])
    
    nodes = deque()
    for key in words:
        if check(begin, key):
            nodes.append([key, 1])
            visited[key] = True
    
    while nodes:
        pop_word, count = nodes.popleft()
        
        if pop_word == target:
            return count
        
        for next_word in graph[pop_word]:
            if not visited[next_word]:
                visited[next_word] = True
                nodes.append([next_word, count+1])
                
    return 0