def solution(s):
    
    int_list = list(map(int, s.split()))
    
    
    return f"{str(min(int_list))} {str(max(int_list))}"