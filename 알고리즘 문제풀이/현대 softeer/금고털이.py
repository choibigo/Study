import input_setting

total_weight, n = map(int, input().split())
weight_price_list = list()
for _ in range(n):
    weight_price_list.append(list(map(int, input().split())))

weight_price_list = sorted(weight_price_list, key=lambda x : -x[1])

print(weight_price_list)

answer = 0
for weight, price in weight_price_list:
    if total_weight < weight:
        answer += (total_weight*price)
        break

    answer += (weight*price)
    total_weight -= weight
    
print(answer)
