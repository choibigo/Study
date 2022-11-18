def solution(number, k):
    stack = [number[0]]
    for pop_num in number[1:]:
        while stack and k and stack[-1] < pop_num:
            k-=1
            stack.pop()
        stack.append(pop_num)
    
    if k:
        stack = stack[:-k]
    
    return "".join(stack)