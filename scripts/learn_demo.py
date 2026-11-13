# learn_demo.py
# The smallest possible model: one number, w, learns the rule "answer = 3 * x".
# Run: python scripts/learn_demo.py
# train.py repeats these same four steps with a real model that has 483,153 numbers like w.

import torch

# Four example inputs, and the correct answers the model must learn to give (3, 6, 9, 12).
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
answer = 3 * x

# The whole model is one number, w. It starts with a bad guess: 0.
# requires_grad=True tells PyTorch to keep track of w, so it can say later which way w should move.
w = torch.tensor(0.0, requires_grad=True)

# A loop repeats the indented lines below it. range(1, 9) counts 1 to 8, so they run 8 times.
for step in range(1, 9):
    # 1. The model guesses by multiplying every input by the current w.
    guess = w * x                           # 1. the model guesses

    # 2. The loss measures how wrong the guess is: the average squared error.
    #    A loss of 0 means a perfect guess.
    loss = ((guess - answer) ** 2).mean()   # 2. measure how wrong it is

    # 3. PyTorch works out whether w should go up or down to make the loss smaller.
    #    The answer is stored in w.grad.
    loss.backward()                         # 3. work out which way to move w

    # 4. Move w a small step in that direction. 0.05 is the learning rate, the size of the step.
    #    Then clear w.grad so the next round starts fresh.
    with torch.no_grad():
        w -= 0.05 * w.grad                  # 4. nudge w a little that way
        w.grad.zero_()

    print("step", step, "  w =", round(w.item(), 3), "  loss =", round(loss.item(), 3))
