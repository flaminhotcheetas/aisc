# bfs

def bfs(m, n, target):
    queue = [(0, 0, [(0, 0)])]
    visited = [(0, 0)]

    while queue:
        x, y, path = queue.pop(0)

        if x == target or y == target:
            print("Solution Path:")
            for state in path:
                print(state)
            return

        next_states = [
            (m, y),
            (x, n),
            (0, y),
            (x, 0),
            (x + min(y, m-x), y - min(y, m-x)),
            (x - min(x, n-y), y + min(x, n-y))
        ]

        for state in next_states:
            if state not in visited:
                visited.append(state)
                queue.append((state[0], state[1], path + [state]))

    print("No solution possible")


bfs(3, 4, 2)



# dfs

def dfs(m, n, target):
    stack = [(0, 0, [(0, 0)])]
    visited = []

    while stack:
        x, y, path = stack.pop()

        if (x, y) in visited:
            continue

        visited.append((x, y))

        if x == target or y == target:
            print("Solution Path:")
            for state in path:
                print(state)
            return

        next_states = [
            (m, y),
            (x, n),
            (0, y),
            (x, 0),
            (x + min(y, m-x), y - min(y, m-x)),
            (x - min(x, n-y), y + min(x, n-y))
        ]

        for state in next_states:
            if state not in visited:
                stack.append((state[0], state[1], path + [state]))

    print("No solution possible")


dfs(3, 4, 2)
