def solution(food):

    result = ""
    for idx, count in enumerate(food[1:], 1):
        count = int(count)
        if count>1:
            result+= str(idx)*(count//2)
        
    return result+"0"+result[::-1]