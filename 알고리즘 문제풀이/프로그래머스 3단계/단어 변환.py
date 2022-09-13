# region BFS
from collections import deque

def generate_graph(begin, target, words):
    
    if begin not in words:
        words.append(begin)
    if target not in words:
        words.append(target)
    
    graph = dict()
    for word in words:
        graph[word] = list()
    
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            
            count = len(words[i])
            for w1, w2 in zip(words[i], words[j]):
                if w1 == w2:
                    count -= 1
            
            if count <=1:
                graph[words[i]].append(words[j])
                graph[words[j]].append(words[i])
    
    return graph
        
            
def solution(begin, target, words):

    if target not in words: return 0
    
    graph = generate_graph(begin, target, words)
    
    visited = dict()
    for word in words:
        visited[word] = False
    
    nodes = deque()
    nodes.append([begin, 0])
    
    while nodes:
        pop_word, pop_count = nodes.popleft()
        
        if pop_word == target:
            return pop_count
        
        for g in graph[pop_word]:
            if visited[g] == False:
                visited[g] = True
                nodes.append([g, pop_count+1])
    
    return 0
# endregion

# region DFS
def generate_graph(begin, target, words):
    
    if begin not in words:
        words.append(begin)
    if target not in words:
        words.append(target)
    
    graph = dict()
    for word in words:
        graph[word] = list()
    
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            
            count = len(words[i])
            for w1, w2 in zip(words[i], words[j]):
                if w1 == w2:
                    count -= 1
            
            if count <=1:
                graph[words[i]].append(words[j])
                graph[words[j]].append(words[i])
    
    return graph

global min_count

def DFS(v, word, target, graph, visited):
    global min_count
    
    if word == target:
        if min_count > v:
            min_count = v
            return 
    
    for g in graph[word]:
        if visited[g] == False:
            visited[g] = True
            DFS(v+1, g, target, graph, visited)
            visited[g] = False
            
def solution(begin, target, words):

    if target not in words: return 0
    
    graph = generate_graph(begin, target, words)
    
    visited = dict()
    for word in words:
        visited[word] = False
    
    global min_count
    min_count = len(words)+1
    
    visited[begin] = True
    
    DFS(0, begin, target, graph, visited)
    
    return min_count
# endregion
