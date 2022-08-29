def my_bin(num):
    if num // 2 == 0:
        return str(num%2)
    
    return my_bin(num//2) + str(num%2)

def solution(s):
    
    total_zero_count = 0
    count = 0
    while True:
        
        if s == "1":
            break
        
        zero_count = s.count("0")
        total_zero_count+= zero_count
        
        zero_remove_len = len(s.replace("0",""))
        s = str(my_bin(zero_remove_len))
    
        count +=1
    
    return [count, total_zero_count]