import input_setting

from collections import Counter

card_count = int(input())
cards = list(map(int, input().split()))
_ = input()
cards_info =dict(Counter(cards))
result = list(map(lambda x: cards_info.get(x, 0), list(map(int, input().split()))))
print(*result)