def solution(routes):

    routes = sorted(routes, key=lambda x : x[1])
    
    answer = 1
    camera = routes[0][1]
    
    for route in routes[1:]:
        start, end = route
        
        if start > camera:
            answer += 1
            camera = end
        
    return answer