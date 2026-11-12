import torch

# The secret rule the model must discover: answer = 3 * x
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
answer = 3 * x

# The whole model is one number, w. It starts with a bad guess: 0
w = torch.tensor(0.0, requires_grad=True)

for step in range(1, 9):
    guess = w * x                           # 1. the model guesses
    loss = ((guess - answer) ** 2).mean()   # 2. measure how wrong it is
    loss.backward()                         # 3. work out which way to move w
    with torch.no_grad():
        w -= 0.05 * w.grad                  # 4. nudge w a little that way
        w.grad.zero_()
    print("step", step, "  w =", round(w.item(), 3), "  loss =", round(loss.item(), 3))
