a, b = map(int, input().strip().split(' '))

# print(*["*" * a for _ in range(b)], sep="\n")
print(("*" * a + "\n") * b)