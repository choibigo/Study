def my_bin(n):
    if n // 2 == 0:
        return str(n%2)
    
    return my_bin(n//2)+str(n%2)

def solution(n):
    c = my_bin(n).count('1')
    for m in range(n+1,1000001):
        if my_bin(m).count('1') == c:
            return m