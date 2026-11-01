# AND Gate using Perceptron Learning Rule

data = [
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, -1]
]

w1 = 0
w2 = 0
b = 0
alpha = 1

epoch = 0

while True:

    epoch = epoch + 1
    error = 0

    print("Epoch =", epoch)
    print("x1\tx2\tt\tw1\tw2\tb\tYin\tY\tdw1\tdw2\tdb")

    for x1, x2, t in data:


        old_w1 = w1
        old_w2 = w2
        old_b = b

        yin = b + (x1 * w1) + (x2 * w2)

        if yin > 0:
            y = 1
        elif yin < 0:
            y = -1
        else:
            y = 0

        if y != t:

            error = error + 1

            delta_w1 = alpha * t * x1
            delta_w2 = alpha * t * x2
            delta_b = alpha * t

            w1 = w1 + delta_w1
            w2 = w2 + delta_w2
            b = b + delta_b

        else:
            delta_w1 = 0
            delta_w2 = 0
            delta_b = 0

        print(
            x1, "\t",
            x2, "\t",
            t, "\t",
            old_w1, "\t",
            old_w2, "\t",
            old_b, "\t",
            yin, "\t",
            y, "\t",
            delta_w1, "\t",
            delta_w2, "\t",
            delta_b
        )

    print()

    if error == 0:
        break

print("Training completed")
print("Total Epochs =", epoch)
print("Final W1 =", w1)
print("Final W2 =", w2)
print("Final Bias =", b)