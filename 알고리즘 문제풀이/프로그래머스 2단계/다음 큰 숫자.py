def my_two(n):
    if n<2:
        return str(n)
    return my_two(n//2) + str(n%2) 

def solution(n):
    one_count = my_two(n).count("1")
    
    while True:
        n = n+1
        i_two = my_two(n)
        if i_two.count("1") == one_count:
            return n
