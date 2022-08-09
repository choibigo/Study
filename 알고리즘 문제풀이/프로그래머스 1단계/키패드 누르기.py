def solution(numbers, hand):
    
    answer=""
    
    # 키패드 좌표료 변경
    dic = {1: [0, 0], 2: [0, 1], 3: [0, 2],
           4: [1, 0], 5: [1, 1], 6: [1, 2],
           7: [2, 0], 8: [2, 1], 9: [2, 2],
           '*':[3, 0], 0: [3, 1], '#': [3, 2]}
    
    # 시작 위치
    left_s = dic['*']
    right_s = dic['#']
    
    for num in numbers:
        now = dic[num]
        if num in [1,4,7]:
            answer+="L"
            left_s = now
        elif num in [3, 6, 9]:
            answer+="R"
            right_s = now
        else:
            left_diff = abs(left_s[0] - now[0]) + abs(left_s[1] - now[1])
            right_diff = abs(right_s[0] - now[0]) + abs(right_s[1] - now[1])
            
            if left_diff < right_diff:
                answer+="L"
                left_s = now
                
            elif left_diff > right_diff:
                answer+="R"
                right_s = now
                
            else:
                if hand == "left":
                    answer+="L"
                    left_s = now
                else:
                    answer+="R"
                    right_s = now
    
    return answer