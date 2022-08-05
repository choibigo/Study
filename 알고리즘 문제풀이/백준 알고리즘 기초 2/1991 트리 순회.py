import input_setting

n = int (input())

def front_DFS(v):
    child1, child2 = graph[v]
    print(v, end="")

    if child1 != ".":
        front_DFS(child1)
    
    if child2 != ".":
        front_DFS(child2)

def mid_DFS(v):
    child1, child2 = graph[v]

    if child1 != ".":
        mid_DFS(child1)
    
    print(v, end="")

    if child2 != ".":
        mid_DFS(child2)

def back_DFS(v):
    child1, child2 = graph[v]

    if child1 != ".":
        back_DFS(child1)
    
    if child2 != ".":
        back_DFS(child2)

    print(v, end="")

graph = dict()

for _ in range(n):
    parent, child1, child2 = input().split()
    graph[parent] = [child1, child2]

front_DFS("A")
print()
mid_DFS("A")
print()
back_DFS("A")