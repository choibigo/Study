def solution(cards):

    len_cards = len(cards)
    graph = [-1 for _ in range(len_cards+1)]
    check = [False for _ in range(len_cards+1)]
    
    
    non_cycle_flag = False
    def cycle(v, start):
        nonlocal non_cycle_flag
        nonlocal value
        value +=1
        
        nv = graph[v]
        if check[nv]:
            if nv != start:
                non_cycle_flag = True
            return 
        
        check[nv] = True
        cycle(nv, start)        
    
    
    for idx, c in enumerate(cards, 1):
        graph[idx] = c
    
    result = list()
    for i in range(1, len_cards+1):
        if not check[i]:
            value = 0
            check[i] = True
            cycle(i, i)
            result.append(value)
            
    result.sort(reverse=True)
    if len(result)== 1:
        return 0
    else:
        return result[0]*result[1]
        
    