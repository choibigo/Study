def solution(a, b, n):
    answer = 0
    while n>=a:
        d, m = divmod(n, a)
        n = d*b+m
        answer += d*b
    return answer