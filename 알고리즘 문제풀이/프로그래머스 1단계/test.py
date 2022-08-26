from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement

data = [1,2,3,4,5]

for select in list(combinations_with_replacement(data, 2)):
    print(select)