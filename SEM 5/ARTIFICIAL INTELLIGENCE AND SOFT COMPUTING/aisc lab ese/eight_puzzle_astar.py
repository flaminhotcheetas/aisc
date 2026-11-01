goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Count misplaced tiles
def heuristic(state):
    count = 0

    for i in range(9):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1

    return count


def solve(start):
    # Each entry contains state, g value and path
    pq = [(start, 0, [start])]
    visited = {}

    while pq:

        # Find state with smallest f = g + h
        best = 0

        for i in range(1, len(pq)):
            if pq[i][1] + heuristic(pq[i][0]) < pq[best][1] + heuristic(pq[best][0]):
                best = i

        state, g, path = pq.pop(best)

        if state in visited and visited[state] <= g:
            continue

        visited[state] = g

        # Check goal
        if state == goal:
            print("Solution Path:")

            for step, board in enumerate(path):
                h = heuristic(board)
                print("Move:", step, "g:", step, "h:", h, "f:", step+h)

                for i in range(0, 9, 3):
                    print(board[i:i+3])

                print()

            print("Goal State reached!")
            return

        # Find blank position
        pos = state.index(0)
        r = pos // 3
        c = pos % 3

        # Up, Down, Left, Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            nr = r + dr
            nc = c + dc

            if 0 <= nr < 3 and 0 <= nc < 3:

                new_state = state.copy()
                new_pos = nr * 3 + nc

                # Swap blank with adjacent tile
                new_state[pos], new_state[new_pos] = new_state[new_pos], new_state[pos]

                new_g = g + 1

                if new_state not in visited or new_g < visited[new_state]:
                    pq.append((new_state, new_g, path + [new_state]))

    print("No solution found")


print("Enter initial state (3 rows):")

start = []

for i in range(3):
    start += list(map(int, input().split()))

solve(start)