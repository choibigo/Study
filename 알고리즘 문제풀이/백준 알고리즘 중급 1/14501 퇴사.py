import input_setting

time_list = [-1]
pay_list = [-1]

dday = int(input())

for _ in range(dday):
    time, pay = map(int, input().split())
    time_list.append(time)
    pay_list .append(pay)

ch_time = [0] * (dday+1)
ch_pay = [0] * (dday+1)

for i in range(1, dday+1):
    ch_time[i] = i + time_list[i] - 1

    if ch_time[i] > dday:
        ch_time[i] = -1
    
    else:
        max_pay = 0
        for j in range(1, i):
            if ch_time[j] < i:
                max_pay = max(max_pay, ch_pay[j])

        ch_pay[i] = pay_list[i] + max_pay

print(max(ch_pay))