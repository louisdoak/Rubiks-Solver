import copy
import torch

def rotate_face_cw(face):
    return [
        [face[2][0], face[1][0], face[0][0]],
        [face[2][1], face[1][1], face[0][1]],
        [face[2][2], face[1][2], face[0][2]],
    ]

def rotate_face_acw(face):
    return [
        [face[0][2], face[1][2], face[2][2]],
        [face[0][1], face[1][1], face[2][1]],
        [face[0][0], face[1][0], face[2][0]],
    ]


def rotate(faces, move):
    faces = copy.deepcopy(faces)

    def row(face, i):
        return face[i][:]

    def col(face, i):
        return [face[r][i] for r in range(3)]

    def set_row(face, i, vals):
        face[i] = vals

    def set_col(face, i, vals):
        for r in range(3):
            face[r][i] = vals[r]

    # ------------------------
    # FRONT (F)
    # ------------------------
    if move == "F" or move == "F'":
        cw = move == "F"

        if cw:
            faces["F"] = rotate_face_cw(faces["F"])
        else:
            faces["F"] = rotate_face_acw(faces["F"])

        U, D, L, R = faces["U"], faces["D"], faces["L"], faces["R"]

        if cw:
            temp = U[2][:]
            U[2] = [L[2][2], L[1][2], L[0][2]]
            L[0][2], L[1][2], L[2][2] = D[0][:]
            D[0] = [R[2][0], R[1][0], R[0][0]]
            R[0][0], R[1][0], R[2][0] = temp
        else:
            temp = U[2][:]
            U[2] = [R[0][0], R[1][0], R[2][0]]
            R[2][0], R[1][0], R[0][0] = D[0][:]
            D[0] = [L[0][2], L[1][2], L[2][2]]
            L[2][2], L[1][2], L[0][2] = temp

    # ------------------------
    # UP (U)
    # ------------------------
    elif move == "U" or move == "U'":
        cw = move == "U"

        if cw:
            faces["U"] = rotate_face_cw(faces["U"])
        else:
            faces["U"] = rotate_face_acw(faces["U"])

        F, B, L, R = faces["F"], faces["B"], faces["L"], faces["R"]

        if cw:
            temp = F[0][:]
            F[0] = R[0][:]
            R[0] = B[0][:]
            B[0] = L[0][:]
            L[0] = temp
        else:
            temp = F[0][:]
            F[0] = L[0][:]
            L[0] = B[0][:]
            B[0] = R[0][:]
            R[0] = temp

    # ------------------------
    # DOWN (D)
    # ------------------------

    elif move == "D" or move == "D'":
        cw = move == "D"

        if cw:
            faces["D"] = rotate_face_cw(faces["D"])
        else:
            faces["D"] = rotate_face_acw(faces["D"])

        F, R, B, L = faces["F"], faces["R"], faces["B"], faces["L"]

        if cw:
            temp = F[2][:]

            F[2] = L[2][:]
            L[2] = B[2][:]
            B[2] = R[2][:]
            R[2] = temp

        else:
            temp = F[2][:]

            F[2] = R[2][:]
            R[2] = B[2][:]
            B[2] = L[2][:]
            L[2] = temp

    # ------------------------
    # RIGHT (R)
    # ------------------------
    elif move == "R" or move == "R'":
        cw = move == "R"

        if cw:
            faces["R"] = rotate_face_cw(faces["R"])
        else:
            faces["R"] = rotate_face_acw(faces["R"])

        U, D, F, B = faces["U"], faces["D"], faces["F"], faces["B"]

        if cw:
            temp = [U[i][2] for i in range(3)]
            for i in range(3):
                U[i][2] = F[i][2]
                F[i][2] = D[i][2]
                D[i][2] = B[2-i][0]
                B[2-i][0] = temp[i]
        else:
            temp = [U[i][2] for i in range(3)]
            for i in range(3):
                U[i][2] = B[2-i][0]
                B[2-i][0] = D[i][2]
                D[i][2] = F[i][2]
                F[i][2] = temp[i]

    # ------------------------
    # LEFT (L)
    # ------------------------
    elif move == "L" or move == "L'":
        cw = move == "L"

        if cw:
            faces["L"] = rotate_face_cw(faces["L"])
        else:
            faces["L"] = rotate_face_acw(faces["L"])

        U, D, F, B = faces["U"], faces["D"], faces["F"], faces["B"]

        if cw:
            temp = [U[i][0] for i in range(3)]
            for i in range(3):
                U[i][0] = B[2-i][2]
                B[2-i][2] = D[i][0]
                D[i][0] = F[i][0]
                F[i][0] = temp[i]
        else:
            temp = [U[i][0] for i in range(3)]
            for i in range(3):
                U[i][0] = F[i][0]
                F[i][0] = D[i][0]
                D[i][0] = B[2-i][2]
                B[2-i][2] = temp[i]

    # ------------------------
    # BACK (B)
    # ------------------------
    elif move == "B" or move == "B'":
        cw = move == "B"

        if cw:
            faces["B"] = rotate_face_cw(faces["B"])
        else:
            faces["B"] = rotate_face_acw(faces["B"])

        U, D, L, R = faces["U"], faces["D"], faces["L"], faces["R"]

        if cw:
            # Save U top
            temp = U[0][:]

            # U top <- R right
            U[0] = [R[0][2], R[1][2], R[2][2]]

            # R right <- D bottom, reversed
            R[0][2], R[1][2], R[2][2] = D[2][2], D[2][1], D[2][0]

            # D bottom <- L left
            D[2] = [L[0][0], L[1][0], L[2][0]]

            # L left <- old U top, reversed
            L[0][0], L[1][0], L[2][0] = temp[2], temp[1], temp[0]

        else:
            # Save U top
            temp = U[0][:]

            # U top <- L left, reversed
            U[0] = [L[2][0], L[1][0], L[0][0]]

            # L left <- D bottom, reversed
            L[0][0], L[1][0], L[2][0] = D[2][0], D[2][1], D[2][2]

            # D bottom <- R right, reversed
            D[2] = [R[2][2], R[1][2], R[0][2]]

            # R right <- old U top
            R[0][2], R[1][2], R[2][2] = temp

    return faces

def _build_perms():
    FACES = ["U", "D", "F", "B", "L", "R"]
    ACTIONS = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
    """Move m sends new_state[i] = old_state[perm[m][i]]. Found by rotating a cube
    whose 54 stickers are labelled 0..53, using YOUR rotations.rotate."""
    labelled = {f: [[k*9 + r*3 + c for c in range(3)] for r in range(3)] for k, f in enumerate(FACES)}
    perms = []
    for m in ACTIONS:
        out = rotate(labelled, m)
        perms.append([out[f][r][c] for f in FACES for r in range(3) for c in range(3)])
    return torch.tensor(perms, dtype=torch.long)    # (12, 54)

PERMS = _build_perms()