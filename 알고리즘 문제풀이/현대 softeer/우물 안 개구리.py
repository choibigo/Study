import input_setting

people_count, relative = map(int, input().split())
weight_list = list(map(int, input().split()))

check = [1] * (people_count)

for _ in range(relative):
    p1, p2 = map(int, input().split())
    p1 -= 1
    p2 -= 1

    if weight_list[p1] == weight_list[p2]:
        check[p1] = 0
        check[p2] = 0
    elif weight_list[p1] > weight_list[p2]:
        check[p2] = 0
    elif weight_list[p1] < weight_list[p2]:
        check[p1] = 0
    
print(sum(check))