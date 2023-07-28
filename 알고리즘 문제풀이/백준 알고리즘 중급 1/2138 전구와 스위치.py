import input_setting

n = int(input())
state = list(map(int, input()))
hope = list(map(int, input()))

"""
000

110
001
010

"""

def charge_zero(light):
    count = 1
    light[0] = light[0]^1
    light[1] = light[1]^1

    for i in range(1, n):
        if light[i-1] != hope[i-1]:
            count +=1
            light[i-1] = light[i-1]^1
            light[i] = light[i]^1

            if i!=n-1:
                light[i+1] = light[i+1]^1

    if light == hope:
        return count

    return 1000000
def no_charge_zero(light):
    count = 0

    for i in range(1, n):
        if light[i-1] != hope[i-1]:
            count +=1
            light[i-1] = light[i-1]^1
            light[i] = light[i]^1

            if i!=n-1:
                light[i+1] = light[i+1]^1
    if light == hope:
        return count

    return 1000000
count1 = charge_zero(state[:])
count2 = no_charge_zero(state[:])

answer = min(count1, count2)

if answer == 1000000:
    answer =-1

print(answer)