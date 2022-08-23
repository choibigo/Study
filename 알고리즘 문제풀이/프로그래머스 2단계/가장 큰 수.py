def solution(numbers):
    
    numbers = list(map(str, numbers))
    
    numbers = sorted(numbers, key = lambda x : x*3, reverse=True)
    
    return str(int("".join(numbers)))