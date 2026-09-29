import numpy as np
import random
import rotations

ACTIONS = ["L","L'","R","R'","U","U'","D","D'","F","F'","B","B'"]

INV = {
    "L":"L'", "L'":"L",
    "R":"R'", "R'":"R",
    "U":"U'", "U'":"U",
    "D":"D'", "D'":"D",
    "F":"F'", "F'":"F",
    "B":"B'", "B'":"B",
}

COLOR_MAP = {"W":0,"Y":1,"R":2,"O":3,"G":4,"B":5}


# -----------------------------
# SOLVED CUBE
# -----------------------------
def solved_cube():
    return {
        "U": [["Y"]*3 for _ in range(3)],
        "D": [["W"]*3 for _ in range(3)],
        "F": [["G"]*3 for _ in range(3)],
        "B": [["B"]*3 for _ in range(3)],
        "L": [["R"]*3 for _ in range(3)],
        "R": [["O"]*3 for _ in range(3)],
    }


# -----------------------------
# ENCODE
# -----------------------------
def encode_cube(faces):
    X = []

    for face in ["U", "D", "F", "B", "L", "R"]:
        for r in range(3):
            for c in range(3):
                one_hot = [0] * 6
                one_hot[COLOR_MAP[faces[face][r][c]]] = 1
                X.extend(one_hot)

    return X

def get_face(move):
    return move[0]

# -----------------------------
# ONE SCRAMBLE PATH
# -----------------------------
def generate_scramble_path(length=20):
    cube = solved_cube()

    scramble = []
    X_data = []
    y_data = []

    previous_face = None

    # -------------------------
    # Generate scramble
    # -------------------------
    for _ in range(length):

        possible_moves = [
            move for move in ACTIONS
            if get_face(move) != previous_face
        ]

        move = random.choice(possible_moves)

        scramble.append(move)
        previous_face = get_face(move)

        cube = rotations.rotate(cube, move)

    # -------------------------
    # Generate reverse solution
    # -------------------------
    solution = [INV[move] for move in reversed(scramble)]

    # -------------------------
    # Record:
    #
    # current state -> next
    # solution move
    # -------------------------
    for move in solution:

        X_data.append(encode_cube(cube))
        y_data.append(ACTIONS.index(move))

        cube = rotations.rotate(cube, move)

    return X_data, y_data, scramble, solution


# -----------------------------
# FULL DATASET
# -----------------------------
def generate_dataset(samples=5000, scramble_len=20):

    X_all = []
    y_all = []

    for i in range(samples):

        X, y, scramble, solution = generate_scramble_path(scramble_len)

        X_all.extend(X)
        y_all.extend(y)

        if i % 100 == 0:
            print(f"Generated {i}/{samples} scrambles")

    X_all = np.array(X_all)

    y_all = np.array(y_all)

    # One-hot encode labels
    y_onehot = np.zeros(
        (len(y_all), len(ACTIONS))
    )

    y_onehot[
        np.arange(len(y_all)),
        y_all
    ] = 1

    return X_all, y_onehot

if __name__ == "__main__":

    # Test one scramble
    X, y, scramble, solution = generate_scramble_path(20)

    print()
    print("SCRAMBLE:")
    print(" ".join(scramble))

    print()
    print("REVERSE SOLUTION:")
    print(" ".join(solution))

    print()
    print("Number of training examples:", len(X))

    # Generate full dataset
    X, y = generate_dataset(
        samples=1,
        scramble_len=20
    )

    np.save("X.npy", X)
    np.save("y.npy", y)

    print()
    print("Dataset saved.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)