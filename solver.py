import numpy as np
import rotations

ACTIONS = ["L","L'","R","R'","U","U'","D","D'","F","F'","B","B'"]

def relu(X):
    return np.maximum(0,X)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def train(X, y_onehot, loadWeights):
    Xtest = np.load("Xtest.npy")
    ytest = np.load("ytest.npy")
    H1 = 256
    H2 = 256
    input_nodes = X.shape[1]
    output_nodes = 12

    lr = 0.01
    epochs = 100
    m = X.shape[0]

    # weights
    W1 = np.random.randn(input_nodes, H1) * np.sqrt(2 / input_nodes)
    W2 = np.random.randn(H1, H2) * np.sqrt(2 / H1)
    W3 = np.random.randn(H2, output_nodes) * np.sqrt(2 / H2)
    b1 = np.zeros((1, H1))
    b2 = np.zeros((1, H2))
    b3 = np.zeros((1, output_nodes))

    if loadWeights:
        W1 = np.load("W1.npy")
        W2 = np.load("W2.npy")
        W3 = np.load("W3.npy")
        b1 = np.load("b1.npy")
        b2 = np.load("b2.npy")
        b3 = np.load("b3.npy")

    for i in range(epochs):
        # test set
        Z1 = Xtest @ W1 + b1
        A1 = np.maximum(0, Z1)

        Z2 = A1 @ W2 + b2
        A2 = np.maximum(0, Z2)

        Z3 = A2 @ W3 + b3
        A3 = softmax(Z3)

        predictions = np.argmax(A3, axis=1)
        actual = np.argmax(ytest, axis=1)
        accuracy = np.mean(predictions == actual)

        # forward
        Z1 = X @ W1 + b1
        A1 = np.maximum(0, Z1)

        Z2 = A1 @ W2 + b2
        A2 = np.maximum(0, Z2)

        Z3 = A2 @ W3 + b3
        A3 = softmax(Z3)
        # loss
        loss = cross_entropy(y_onehot, A3)

        # backprop
        dZ3 = (A3 - y_onehot) / m
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * (Z2 > 0)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # update
        W1 -= lr * dW1
        W2 -= lr * dW2
        W3 -= lr * dW3

        b1 -= lr * db1
        b2 -= lr * db2
        b3 -= lr * db3

        if i % 20 == 0:
            print(
                f"Epoch {i}/{epochs} - "
                f"loss: {loss:.6f} - "
                f"accuracy: {accuracy:.2%}"
            )
            

    return W1, b1, W2, b2, W3, b3

def predict(X, W1, b1, W2, b2, W3, b3, sample=False):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)

    Z2 = A1 @ W2 + b2
    A2 = np.maximum(0, Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return A3[0]
    if sample:
        return np.random.choice(len(A3[0]), p=A3[0])
    else:
        return np.argmax(A3[0])

def solve(faces, W1, b1, W2, b2, W3, b3, max_steps=150):
    solution = []

    solved_cube = {
        "U": [["Y"]*3 for _ in range(3)],
        "D": [["W"]*3 for _ in range(3)],
        "F": [["G"]*3 for _ in range(3)],
        "B": [["B"]*3 for _ in range(3)],
        "L": [["R"]*3 for _ in range(3)],
        "R": [["O"]*3 for _ in range(3)],
    }

    current = faces
    visited = set()

    for step in range(max_steps):
        print(step)

        # Check whether solved
        if current == solved_cube:
            print("Solved!")
            break

        # Convert cube state into something hashable
        key = encode_cube(current).tobytes()

        # Check for repeated state
        if key in visited:
            print("Loop detected!")
            break

        visited.add(key)

        X = encode_cube(current)

        probs = predict(
            X, W1, b1, W2, b2, W3, b3
        )

        order = np.argsort(probs)[::-1]

        print("Top predictions:")
        for idx in order[:3]:
            print(f"  {ACTIONS[idx]}: {probs[idx]:.3f}")

        move_index = order[0]
        move = ACTIONS[move_index]

        print("Chosen:", move)

        solution.append(move)

        current = rotations.rotate(current, move)

    return solution

def encode_cube(faces):
    color_map = {"W":0,"Y":1,"R":2,"O":3,"G":4,"B":5}

    X = []
    for face in ["U","D","F","B","L","R"]:
        for r in range(3):
            for c in range(3):
                one_hot = [0]*6
                one_hot[color_map[faces[face][r][c]]] = 1
                X.extend(one_hot)

    return np.array(X)