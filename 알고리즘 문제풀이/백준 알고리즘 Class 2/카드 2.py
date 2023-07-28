import input_setting

from collections import deque

n = int(input())
card_list = deque(list(range(1, n+1)))
status = 0

while len(card_list) > 1:
    if status ==0:
        status = 1
        card_list.popleft()

    elif status ==1:
        pop_card = card_list.popleft()
        card_list.append(pop_card)
        status = 0

print(card_list[0])