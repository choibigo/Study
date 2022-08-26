def solution(number, k):
    
    numbers = list(map(int, list(number)))
    
    count = k
    stack_list = [numbers[0]]
    
    for num in numbers[1:]:
        while len(stack_list) and stack_list[-1] < num and count != 0:
            stack_list.pop()
            count -= 1

        stack_list.append(num)
    
    if count == 0:
        return "".join(list(map(str, stack_list)))
    else:
        return "".join(list(map(str, stack_list[:-count])))    
    
