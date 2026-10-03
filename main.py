import tkinter as tk
import rotations
import solver
import numpy as np
import time
import torch
import random as rnd
dev = "cuda" if torch.cuda.is_available() else "cpu"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rubiks Cube NN Solver")
        self.geometry("1200x750")
        self.configure(bg="#1e1e1e")
        self.resizable(True, True)
        self.loadWeights = tk.BooleanVar()

        # Easy-to-edit face colors
        self.color_map = {
            "W": "white",
            "Y": "yellow",
            "R": "red",
            "O": "orange",
            "G": "green",
            "B": "blue",
        }

        # Cube state (editable)
        self.faces = {
            "U": [["Y"]*3 for _ in range(3)],
            "D": [["W"]*3 for _ in range(3)],
            "F": [["G"]*3 for _ in range(3)],
            "B": [["B"]*3 for _ in range(3)],
            "L": [["R"]*3 for _ in range(3)],
            "R": [["O"]*3 for _ in range(3)],
        }

        self.buttons = {}  # store button refs
        
        self._build_functions_toolbar()
        self._build_main()
        self._build_rotations_toolbar()

    def _build_functions_toolbar(self):
        bar = tk.Frame(self, bg="#2d2d2d", pady=8)
        bar.pack(fill="x")
        tk.Button(
                bar, text="Solve",
                command=self._solve,
                bg="#F0F0F0", fg="#000000",
                font=("Segoe UI", 16, "bold"),
                relief="flat", padx=10, pady=4,
                cursor="hand2"
            ).pack(side="left", padx=12)
        tk.Button(
                bar, text="Scramble",
                command=self._scramble,
                bg="#F0F0F0", fg="#000000",
                font=("Segoe UI", 16, "bold"),
                relief="flat", padx=10, pady=4,
                cursor="hand2"
            ).pack(side="left", padx=12)
        tk.Button(
                bar, text="Reset",
                command=self._reset,
                bg="#F0F0F0", fg="#000000",
                font=("Segoe UI", 16, "bold"),
                relief="flat", padx=10, pady=4,
                cursor="hand2"
            ).pack(side="left", padx=12)
        tk.Button(
                bar, text="Train",
                command=self._train,
                bg="#F0F0F0", fg="#000000",
                font=("Segoe UI", 16, "bold"),
                relief="flat", padx=10, pady=4,
                cursor="hand2"
            ).pack(side="left", padx=12)
        tk.Checkbutton(
            bar, text="Continue old",
            bg="#F0F0F0", fg="#000000",
            font=("Segoe UI", 16, "bold"),
            relief="flat", padx=10, pady=4,
            cursor="hand2", variable=self.loadWeights
        ).pack(side="left", padx=12)
        tk.Button(
                bar, text="Debug",
                command=self._debug,
                bg="#F0F0F0", fg="#000000",
                font=("Segoe UI", 16, "bold"),
                relief="flat", padx=10, pady=4,
                cursor="hand2"
            ).pack(side="left", padx=12)

    def _build_rotations_toolbar(self):
        bar = tk.Frame(self, bg="#2d2d2d", pady=8)
        bar.pack(fill="x")

        tk.Label(
            bar, text="Rotate Faces",
            bg="#2d2d2d", fg="#ffffff",
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=12)
        turns = ['L','R','U','D','F','B','L\'','R\'','U\'','D\'','F\'','B\'']
        for move in turns:
            tk.Button(
                bar, text=move,
                command=lambda m=move: self._rotate_side(m),
                bg="#F0F0F0", fg="#000000",
                font=("Segoe UI", 16, "bold"),
                relief="flat", padx=10, pady=4,
                cursor="hand2"
            ).pack(side="left", padx=12)
    
    def _build_main(self):
        self.main_frame = tk.Frame(self, bg="#1e1e1e")
        self.main_frame.pack(expand=True)

        layout = [
            (0, 3, "U"),
            (3, 0, "L"),
            (3, 3, "F"),
            (3, 6, "R"),
            (3, 9, "B"),
            (6, 3, "D"),
        ]

        for r_offset, c_offset, face in layout:
            self._build_face(r_offset, c_offset, face)

    def _build_face(self, r0, c0, face):
        for r in range(3):
            for c in range(3):
                color_code = self.faces[face][r][c]
                btn = tk.Button(
                    self.main_frame,
                    bg=self.color_map[color_code],
                    width=4,
                    height=2,
                    command=lambda f=face, i=r, j=c: self._cycle_color(f, i, j),
                    relief="flat"
                )
                btn.grid(row=r0 + r, column=c0 + c, padx=1, pady=1)

                self.buttons[(face, r, c)] = btn

    def _cycle_color(self, face, r, c):
        order = list(self.color_map.keys())
        current = self.faces[face][r][c]
        idx = order.index(current)
        new = order[(idx + 1) % len(order)]

        self.faces[face][r][c] = new
        self.buttons[(face, r, c)].configure(bg=self.color_map[new])
    
    def _rotate_side(self, move):
        self.faces = rotations.rotate(self.faces, move)
        self.update_ui()

    def _solve(self):
        self.solution = solver.solve(
            self.faces,
            torch.from_numpy(np.load("W1.npy")).float().to(dev),
            torch.from_numpy(np.load("b1.npy")).float().to(dev),
            torch.from_numpy(np.load("W2.npy")).float().to(dev),
            torch.from_numpy(np.load("b2.npy")).float().to(dev),
            torch.from_numpy(np.load("W3.npy")).float().to(dev),
            torch.from_numpy(np.load("b3.npy")).float().to(dev)
        )

        self.step = 0
        self._animate_solve()
            
    def _animate_solve(self):
        if self.step >= len(self.solution):
            self.update_ui()
            return

        move = self.solution[self.step]

        self.faces = rotations.rotate(self.faces, move)
        if len(self.solution)-self.step < 100:
            print(move)
            self.update_ui()
        if self.step % 50 == 0:
            print(self.step)
            self.update_ui()

        self.step += 1
        if len(self.solution)-self.step < 100:
            self.after(80, self._animate_solve)
        else:
            self.after(0, self._animate_solve)

    def _scramble(self):
        turns = ['L','R','U','D','F','B','L\'','R\'','U\'','D\'','F\'','B\'']
        for i in range(20):
            self._rotate_side(turns[rnd.randint(0,11)])

    def _reset(self):
        self.faces = {
            "U": [["Y"]*3 for _ in range(3)],
            "D": [["W"]*3 for _ in range(3)],
            "F": [["G"]*3 for _ in range(3)],
            "B": [["B"]*3 for _ in range(3)],
            "L": [["R"]*3 for _ in range(3)],
            "R": [["O"]*3 for _ in range(3)],
        }
        self.update_ui()

    def _train(self):
        X = torch.from_numpy(np.load("X.npy")).float().to(dev)
        y = torch.from_numpy(np.load("y.npy")).float().to(dev)
        print(self.loadWeights.get())
        W1, b1, W2, b2, W3, b3 = solver.train(X, y, self.loadWeights.get())

        np.save("W1.npy", W1.cpu().numpy())
        np.save("b1.npy", b1.cpu().numpy())
        np.save("W2.npy", W2.cpu().numpy())
        np.save("b2.npy", b2.cpu().numpy())
        np.save("W3.npy", W3.cpu().numpy())
        np.save("b3.npy", b3.cpu().numpy())

        print("Training complete")

    def _debug(self):
        W1=torch.from_numpy(np.load("W1.npy")).float().to(dev)
        b1=torch.from_numpy(np.load("b1.npy")).float().to(dev)
        W2=torch.from_numpy(np.load("W2.npy")).float().to(dev)
        b2=torch.from_numpy(np.load("b2.npy")).float().to(dev)
        W3=torch.from_numpy(np.load("W3.npy")).float().to(dev)
        b3=torch.from_numpy(np.load("b3.npy")).float().to(dev)

        ACTIONS = ["L","L'","R","R'","U","U'","D","D'","F","F'","B","B'"]
        depthTest = False
        if depthTest:
            depth = 2
            for depth in range(1,20):
                success = 0
                for i in range(1000):
                    self._reset()
                    for i in range(depth):
                        self.faces = rotations.rotate(self.faces, ACTIONS[rnd.randint(0,11)])
                        self.update_ui()

                    tempSolution = solver.solve(
                        self.faces,
                        W1,b1,W2,b2,W3,b3,
                        verbose=False,
                        max_steps=20
                    )
                    self.step = 0
                    if len(tempSolution) == depth:
                        success+=1
                print(f"Accuracy on depth {depth}: {(success/1000) *100}")

        solveTest=True
        if solveTest:
            moveNums = np.array([])
            success = 0
            for i in range(1000):
                print(i)
                self._reset()
                self._scramble()
                tempSolution, solved = solver.solve(
                    self.faces,
                    W1,b1,W2,b2,W3,b3,
                    verbose=False,
                    max_steps=20000
                )
                self.step = 0
                if solved:
                    print(f"solved in {len(tempSolution)} moves")
                    moveNums = np.append(moveNums, len(tempSolution))
                    success+=1
            print(f"Solved {(success/1000) *100}% of cubes")
            print(f"mean:{np.mean(moveNums)}")
            print(f"median:{np.median(moveNums)}")
            print(moveNums)

    def update_ui(self):
        for face in self.faces:
            for r in range(3):
                for c in range(3):
                    color_code = self.faces[face][r][c]
                    self.buttons[(face, r, c)].configure(
                        bg=self.color_map[color_code]
                    )

if __name__ == "__main__":
    app = App()
    app.mainloop()