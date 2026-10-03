# Rubik's Cube Neural Network Solver

A neural network that learns to solve a 3x3 Rubik's cube, with a Tkinter GUI to scramble and solve it.
The network predicts, for any scrambled cube, which move would undo the *last* scramble move. A simple
search loop then follows those predictions until the cube is solved.

The training loop (forward pass, backpropagation and Adam) is handwritten on top of PyTorch tensors,
so the maths is explicit and the arithmetic runs on the GPU. PyTorch is used for GPU tensors and basic
tensor operations, not for autograd, torch.optim or nn layers.

> **Credit/Motivation:** the overall approach (predict the inverse of the last move, then follow the network with a
> visited-state check) was inspired by [kongaskristjan/rubik](https://github.com/kongaskristjan/rubik). When I first
> decided to attempt this project for my A-Level coursework it was the "simplest/stupidest approach", in kongaskristjan's
> own words, that I could find. To my disappointment at the time, I didn't get it working before the deadline, but two years
> on, a C# to python switch, and a dozen features/implementation details that I only wish I knew at the time, and here we are.

## Results

All results are for the model after 5,000 training steps.

**Fully scrambled cubes** (1,000 random 20-move scrambles, budget of 20,000 moves):

| Metric | Value |
|---|---:|
| Solved within 20,000 moves | 51.6% |
| Mean moves (solved cubes only) | 7,808 |
| Median moves (solved cubes only) | 6,571 |
| Max moves (solved cubes only) | 19912 |
| Min moves (solved cubes only)| 6 |

Solutions are long: the solver is a guided random walk, not an optimal solver.

**Strict test:** 1,000 scrambles per depth, counted as solved only if the solver finishes in no more
moves than the scramble used.

| Depth | Solved |
|---:|---:|
|  1 | 100.0% |
|  2 |  91.5% |
|  3 |  79.2% |
|  4 |  72.0% |
|  5 |  66.0% |
|  6 |  50.8% |
|  7 |  36.1% |
|  8 |  25.1% |
|  9 |  14.3% |
| 10 |   5.8% |
| 11 |   2.4% |
| 12 |   1.2% |
| 13 |   0.4% |
| 14 |   0.3% |
| 15 |   0.1% |
| 16-20 | 0.0% |

**Per-move accuracy** on 1,000 held-out test cubes per depth. This is how often the network picks the
move that undoes the last scramble move, not the solve rate. "Correct" is subjective with this approach,
since several moves can be equally good, which is part of why solutions are so long.

| Depth | Accuracy |
|---:|---:|
|   1 | 100.00% |
|   2 |  89.40% |
|   3 |  88.10% |
|   5 |  85.20% |
|   8 |  66.80% |
|  10 |  48.10% |
|  15 |  22.10% |
|  20 |  14.90% |

overall: 50.41%

## How it works

**Cube encoding.** 54 stickers x 6 colours, one-hot encoded, gives 324 inputs.

**Network.** A fully connected network, 324 -> 4096 -> 2048 -> 12 (ReLU hidden layers, softmax output
over the 12 moves `L L' R R' U U' D D' F F' B B'`).

**Training data.** Generated fresh on the GPU at every step, so the network never sees the same cube twice:

1. Every move is stored as a 54-entry permutation table, built once by running `rotations.rotate` on a cube
   whose stickers are labelled 0-53.
2. A batch of solved cubes is scrambled by a random depth of 1-20 moves, with a heavier weight on middling
   depths to aid training.
3. The label is the inverse of the last scramble move. Half of each batch is drawn only from depths 8-16 where
   the network has the most to learn, the rest uniformly from 1-20.

**Training.** Mini-batches of 512, cross-entropy loss, handwritten backpropagation, handwritten Adam.

**Solving.** Repeatedly take the network's highest-ranked move that does not lead to a recently visited state.
The visited set is reset every 100 steps so the solver can undo its own mistakes.

## What I learned

- Two hidden layers 128->128 was enough in the beginning to solve cubes that were 2-3 moves from solved, but
  I settled on 4096->2048 in the end, which interestingly was one less hidden layer, and half the width per layer
  of kongaskristjan's network (but he had shorter and more solves than me).
- One of the main improvements over my original C# project came from pytorch's GPU tensor operations. My C#
  project used handwritten matrix functions, and was brutally slow all round.
- Adam optimisation similarly seemed to give a good boost to the testing accuracy.
- Overfitting: Originally I used a fixed dataset of 20 × 5,000 states with no test set. Once I added one, training accuracy
  pulled ahead of test accuracy, which showed the network was overfitting. Generating new cubes at every step fixed this.
- Per-depth accuracy showed that deep scrambles (about depth 14+) were close to unpredictable, because the last
  move is almost impossible to infer from a heavily scrambled cube. That made mid-depth states the best place to
  spend training effort.
- Overall accuracy was misleading for a while, as it averaged easy shallow states with effectively unlearnable deep ones,
  so per-depth accuracy was helpful alongside this.

## Running it

Requires Python 3 and PyTorch with a CUDA build that supports your GPU.

```
pip install torch
python generator.py   # creates the fixed test set (Xtest.npy, ytest.npy), also creates an unused training set
python main.py        # opens the GUI: Scramble, Train, Solve
```

Trained weights and data (`*.npy`) are not stored in the repo. Train once with the Train button. You can continue training from
weights in the project folder by checking "continue old".

## Files

- `main.py` - Tkinter GUI
- `solver.py` - network, training loop, Adam, solve loop
- `rotations.py` - cube move logic and the permutation table
- `generator.py` - builds the fixed test set

## Limitations

- Solutions are very long.
- Solving is effectively a random walk until a state is found that the model can make predictions on. Some scrambles take far longer than the median.
- Only 3x3 cubes and quarter-turn moves are supported.
- Nearly half the cubes could not be solved in under 20k moves.
