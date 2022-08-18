def solution(a, b):
    
    res = [0, 31, 29, 31, 30, 31, 30, 31, 30, 30, 31, 30, 31]
    
    day = sum(res[:a])
    day += (b-1)
    day %= 7 
    
    fuck = ["FRI","SAT","SUN","MON","TUE","WED","THU"]
    
    return fuck[day]