# learn_demo.py: a tiny model learns the rule "multiply by 3".
# Run: python scripts/learn_demo.py

import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])    # The questions.
answer = 3 * x                             # The right answers: 3, 6, 9, 12.

# The whole model is one number, w. It starts at 0, which is wrong.
w = torch.tensor(0.0, requires_grad=True)

for step in range(1, 9):                    # Repeat 8 times.
    guess = w * x                           # 1. The model makes a guess.
    loss = ((guess - answer) ** 2).mean()   # 2. Measure how wrong the guess is.
    loss.backward()                         # 3. Work out if w should go up or down.
    with torch.no_grad():
        w -= 0.05 * w.grad                  # 4. Move w a small step that way.
        w.grad.zero_()                      # Forget the old direction before the next round.
    print("step", step, "  w =", round(w.item(), 3), "  loss =", round(loss.item(), 3))
