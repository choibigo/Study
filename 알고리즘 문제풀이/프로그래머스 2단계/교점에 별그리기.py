from itertools import combinations 

def intersection_point(line1, line2):
    
    a,b,e = line1
    c,d,f = line2
    if a*d == b*c:
        return None
    
    x = (b*f-e*d)/(a*d-b*c)
    y = (e*c-a*f)/(a*d-b*c)
    if x == int(x) and y == int(y):
        return (int(x),int(y))
    
    return None
    
def solution(line):
    
    x_min = +1001
    x_max = -1001
    y_min = +1001
    y_max = -1001
    point_list = set()
    
    for line1, line2 in list(combinations(line, 2)):
        point = intersection_point(line1, line2)
        if point:
            point_list.add(point)
            
    xs = [p[0] for p in point_list]
    x_min = min(xs)
    x_max = max(xs)
    
    ys = [p[1] for p in point_list]
    y_min = min(ys)
    y_max = max(ys)
    
    answer = [['.' for _ in range(x_max-x_min+1)] for _ in range (y_max-y_min+1)]
    
    for x, y in point_list:
        answer[y_max-y] = answer[y_max-y][:x-x_min] + ['*'] + answer[y_max-y][x-x_min+1:]
    
    return [''.join(ans) for ans in answer]
    