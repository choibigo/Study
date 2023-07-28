import input_setting

import sys
while True:
    line_list = list(map(int, sys.stdin.readline().split(' ')))

    if line_list[0] == 0:
        break

    max_line = max(line_list)
    line_list.remove(max_line)

    if max_line**2 == sum(list(map(lambda x: x**2 , line_list))):
        print('right')

    else:
        print('wrong')

