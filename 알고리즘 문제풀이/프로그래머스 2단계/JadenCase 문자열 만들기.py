def solution(s):
    
    split_s_list = s.split(" ")
    
    for i in range(len(split_s_list)):
        if len(split_s_list[i]) != 0:
            split_s_list[i] = split_s_list[i][0].upper()+split_s_list[i][1:].lower()

        
    return " ".join(split_s_list)