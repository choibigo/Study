import input_setting

from itertools import combinations

n, k = map(int, input().split())
word_list = [input()[4 : -4] for _ in range(n)]

info = dict()
for c in range(97, 97+26):
    info[chr(c)] = 0
info['a'] = 1
info['n'] = 1
info['t'] = 1
info['c'] = 1
info['i'] = 1

alphabet = [chr(x) for x in range(97, 97+26)]
alphabet.remove('a')
alphabet.remove('n')
alphabet.remove('t')
alphabet.remove('i')
alphabet.remove('c')

def check():
    
    count = n
    for word in word_list:
        for w in word:
            if info[w] == 0:
                count -=1
                break
    return count
        
max_count = 0
if k >=5:
    for temp in list(combinations(alphabet, k-5)):
        for t in temp: info[t] = 1
        count = check()
        max_count = max(max_count, count)
        for t in temp: info[t] = 0
    print(max_count)

else:
    print(0)