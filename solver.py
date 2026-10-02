import numpy as np
import rotations
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"

ACTIONS = ["L","L'","R","R'","U","U'","D","D'","F","F'","B","B'"]

def relu(X):
    return np.maximum(0,X)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(x):
    x = x - x.max(dim=1, keepdim=True).values
    exp = torch.exp(x)
    return exp / exp.sum(dim=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -((y_true * torch.log(y_pred + 1e-8)).sum(dim=1)).float().mean()

def adam_step(W, dW, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m.mul_(beta1).add_(dW, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(dW, dW, value=1 - beta2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W.sub_(lr * m_hat / (v_hat.sqrt() + eps))

def train(X_full, y_onehot_full, loadWeights):
    with torch.no_grad():
        Xtest = torch.from_numpy(np.load("Xtest.npy")).float().to(dev)
        ytest = torch.from_numpy(np.load("ytest.npy")).float().to(dev)
        print("X", X_full.shape, "y", y_onehot_full.shape, "Xtest", Xtest.shape, "ytest", ytest.shape)

        H1 = 4196
        H2 = 2048
        input_nodes = X_full.shape[1]
        output_nodes = 12

        lr = 0.5
        epochs = 100
        m = X_full.shape[0]

        # weights
        W1 = (torch.randn(input_nodes, H1) * (2 / input_nodes)**0.5).to(dev)
        W2 = (torch.randn(H1, H2) * (2 / H1)**0.5).to(dev)
        W3 = (torch.randn(H2, output_nodes) * (2 / H2)**0.5).to(dev)
        b1 = torch.zeros(1, H1, device=dev)
        b2 = torch.zeros(1, H2, device=dev)
        b3 = torch.zeros(1, output_nodes, device=dev)
        #ADAM
        W1m = torch.zeros_like(W1)
        W1v = torch.zeros_like(W1)
        W2m = torch.zeros_like(W2)
        W2v = torch.zeros_like(W2)
        W3m = torch.zeros_like(W3)
        W3v = torch.zeros_like(W3)
        b1m = torch.zeros_like(b1)
        b1v = torch.zeros_like(b1)
        b2m = torch.zeros_like(b2)
        b2v = torch.zeros_like(b2)
        b3m = torch.zeros_like(b3)
        b3v = torch.zeros_like(b3)

        if loadWeights:
            torch.from_numpy(np.load("X.npy")).float().to(dev)
            W1 = torch.from_numpy(np.load("W1.npy")).float().to(dev)
            W2 = torch.from_numpy(np.load("W2.npy")).float().to(dev)
            W3 = torch.from_numpy(np.load("W3.npy")).float().to(dev)
            b1 = torch.from_numpy(np.load("b1.npy")).float().to(dev)
            b2 = torch.from_numpy(np.load("b2.npy")).float().to(dev)
            b3 = torch.from_numpy(np.load("b3.npy")).float().to(dev)

        for i in range(epochs):
            batch_size = 512
            idx = torch.randint(0, m, (batch_size,), device=dev)
            X = X_full[idx]
            y_onehot = y_onehot_full[idx]
            # test set
            Z1 = Xtest @ W1 + b1
            A1 = Z1.clamp(min=0)

            Z2 = A1 @ W2 + b2
            A2 = Z2.clamp(min=0)

            Z3 = A2 @ W3 + b3
            A3 = softmax(Z3)

            predictions = A3.argmax(dim=1)
            actual = ytest.argmax(dim=1)
            accuracy = (predictions == actual).float().mean().item()

            # forward
            Z1 = X @ W1 + b1
            A1 = Z1.clamp(min=0)

            Z2 = A1 @ W2 + b2
            A2 = Z2.clamp(min=0)

            Z3 = A2 @ W3 + b3
            A3 = softmax(Z3)
            # loss
            loss = cross_entropy(y_onehot, A3)

            # backprop
            dZ3 = (A3 - y_onehot) / batch_size
            dW3 = A2.T @ dZ3
            db3 = dZ3.sum(dim=0, keepdim=True)

            dA2 = dZ3 @ W3.T
            dZ2 = dA2 * (Z2 > 0)
            dW2 = A1.T @ dZ2
            db2 = dZ2.sum(dim=0, keepdim=True)

            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * (Z1 > 0)
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(dim=0, keepdim=True)
            
            # update
            t=i+1
            adam_step(W1, dW1, W1m, W1v, t)
            adam_step(b1, db1, b1m, b1v, t)
            adam_step(W2, dW2, W2m, W2v, t)
            adam_step(b2, db2, b2m, b2v, t)
            adam_step(W3, dW3, W3m, W3v, t)
            adam_step(b3, db3, b3m, b3v, t)

            if i % 20 == 0:
                print(
                    f"Epoch {i}/{epochs} - "
                    f"loss: {loss.item():.6f} - "
                    f"accuracy: {accuracy:.2%}"
                )
                

        return W1, b1, W2, b2, W3, b3

def predict(X, W1, b1, W2, b2, W3, b3, sample=False):
    Z1 = X @ W1 + b1
    A1 = Z1.clamp(min=0)

    Z2 = A1 @ W2 + b2
    A2 = Z2.clamp(min=0)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return A3[0]
    if sample:
        return np.random.choice(len(A3[0]), p=A3[0])
    else:
        return np.argmax(A3[0])

def solve(faces, W1, b1, W2, b2, W3, b3, max_steps=20000):
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
        key = str(current)

        # Check for repeated state
        if key in visited:
            print("Loop detected!")
            break

        visited.add(key)

        X = encode_cube(current)

        probs = predict(
            X, W1, b1, W2, b2, W3, b3
        )

        order = torch.argsort(probs, descending=True).tolist()

        print("Top predictions:")
        for idx in order[:3]:
            print(f"  {ACTIONS[idx]}: {probs[idx]:.3f}")

        for a in order:
            potential_state = rotations.rotate(current, ACTIONS[a])
            if str(potential_state) not in visited:
                print("Chosen:", ACTIONS[a])
                break

        solution.append(ACTIONS[a])

        current = potential_state

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

    return torch.tensor(X, dtype=torch.float32, device=dev)