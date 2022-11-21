def solution(n):

    result = [[0 for _ in range(i+1)] for i in range(n)]
    
    move_info = dict()
    move_info[0] = (0, 1)
    move_info[1] = (1, 0)
    move_info[2] = (-1, -1)
    
    pos = [0,-1]
    num = 0
    
    for i in range(n):
        for _ in range(n-i):
            nx = pos[0] + move_info[i%3][0]
            ny = pos[1] + move_info[i%3][1]
            pos = [nx, ny]
            num+=1
            result[ny][nx] = num

    answer = list()
    for row in result:
        answer += row
        
    return answer