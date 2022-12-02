import input_setting

'''
1   2   3   4   5   6   7   8   9
1   3   6   10  15  21  28  36  45
1   4   10  20  35  56  


'''

for _ in range(int(input())):
    rows = int(input())
    cols = int(input())


    board = [[c+1 for c in range(cols+1)] for _ in range(rows+1)]

    print(*board, sep='\n')
    print()