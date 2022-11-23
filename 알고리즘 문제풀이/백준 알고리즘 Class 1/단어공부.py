import input_setting

from collections import Counter
import sys

def func(string):
    if len(string) == 1:
        return string.upper()

    counter = Counter(string.lower())
    counter = sorted(counter.items(), key=lambda x : (-x[1], x[0]))

    if counter[0][1] == counter[1][1]:
        return '?'
    else:
        return counter[0][0].upper()

if __name__ =="__main__":
    print(func(input()))