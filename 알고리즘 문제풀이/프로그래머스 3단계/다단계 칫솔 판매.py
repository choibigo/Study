from collections import defaultdict
import sys
sys.setrecursionlimit(1000000)

def solution(enroll, referral, seller, amount):

    graph = defaultdict(list)
    parent_info = dict()
    sell_info = dict()
    visited = dict()
    result = dict()
    
    for child, parent in zip(enroll, referral):
        graph[parent].append(child)
        parent_info[child] = parent
    
    for key in enroll+['-']:
        sell_info[key] = []
        visited[key] = False
        result[key] = 0
        
    for key, value in zip(seller, amount):
        sell_info[key].append(value) 
    
    def DFS(node):
        for g in graph[node]:
            if not visited[g]:
                visited[g] = True
                for s in sell_info[g]:
                    money_give(g, s*100)
                DFS(g)
    
    def money_give(child, money):
        if child=='-':
            result['-'] += money
            return
        
        if money<1:
            return 
        
        money_10 = money//10
        result[child] += (money - money_10)
        money_give(parent_info[child], money_10)
        
    DFS('-')
    return [result[e] for e in enroll]