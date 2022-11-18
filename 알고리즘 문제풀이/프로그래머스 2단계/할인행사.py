def check(discount_info, my_info):
    for key in my_info.keys():
        if discount_info.get(key, 0) < my_info[key]:
            return 0
    return 1

def solution(want, number, discount):
    discount_info = dict.fromkeys(discount, 0)
    my_info = {key:value for key, value in zip(want, number)}
    count = 0
    
    for i in range(10):
        discount_info[discount[i]] += 1
    count += check(discount_info, my_info)
    
    for i in range(len(discount)):
        discount_info[discount[i]] -= 1
        if i+10 < len(discount):
            discount_info[discount[i+10]] += 1
        count += check(discount_info, my_info)
    return count