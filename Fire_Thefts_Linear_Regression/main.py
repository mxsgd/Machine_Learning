import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(
    "fires_thefts.csv",
    sep=",",
    names=["fires", "burglary"])


m, n_plus_1 = data.values.shape
n = n_plus_1 - 1
Xn = data.values[:, :1].reshape(m, n)


X = np.matrix(np.concatenate((np.ones((m, 1)), Xn), axis=1)).reshape(m, n_plus_1)
y = np.matrix(data.values[:, 1]).reshape(m, 1)

def h(theta, x):
    return theta[0] + x * theta[1]

def J(theta, X, y):
    m = len(y)
    cost = 1.0 / (2.0 * m) * ((X * theta - y).T * (X * theta - y))
    return cost.item()

def dJ(theta, X, y):
    return 1.0 / len(y) * (X.T * (X * theta - y))

def gradient_descent(fJ, fdJ, theta, X, y, alpha, eps):
    current_cost = fJ(theta, X, y)
    history = [[current_cost, theta]]
    while True:
        theta = theta - alpha * fdJ(theta, X, y)  # implementacja wzoru
        current_cost, prev_cost = fJ(theta, X, y), current_cost
        if abs(prev_cost - current_cost) <= eps:
            break
        if current_cost > prev_cost:
            print("Długość kroku (alpha) jest zbyt duża!")
            break
        history.append([current_cost, theta])
    return theta, history

theta_start = np.zeros((n + 1, 1))

theta_best, history = gradient_descent(J, dJ, theta_start, X, y, alpha=0.0007, eps=9)


print("parametry krzywej regresyjnej: \n", theta_best)

# jak zmienia sie koszt w zaleznoscio od alpha
# najkorzystniejsze alpha +- 0.0007

aalpha = np.linspace(0.005, 0.0001, 9)
costs = []
lengths = []
for alpha in aalpha:
    theta_best, history = gradient_descent(
        J, dJ, theta_start, X, y, alpha=alpha, eps=9
    )
    cost = history[-1][0]
    steps = len(history)
    costs.append(cost)
    lengths.append(steps)

def alpha_cost_steps_plot(eps, costs, steps):
    """Wykres kosztu i liczby kroków w zależności od alphy"""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(eps, steps, "--s", color="green")
    ax2.plot(eps, costs, ":o", color="orange")
    ax1.set_xscale("log")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("liczba kroków", color="green")
    ax2.set_ylabel("koszt", color="orange")
    plt.show()

alpha_cost_steps_plot(aalpha, costs, lengths)

# jak zmienia sie koszt w zaleznoscio od epsilon

epss = [10.0**n for n in range(-1, 5)]
costs = []
lengths = []
for eps in epss:
    theta_best, history = gradient_descent(
        J, dJ, theta_start, X, y, alpha=0.0007, eps=eps
    )
    cost = history[-1][0]
    steps = len(history)
    print(f"{eps=:7},  {cost=:15.3f},  {steps=:6}")
    costs.append(cost)
    lengths.append(steps)

def eps_cost_steps_plot(eps, costs, steps):
    """Wykres kosztu i liczby kroków w zależności od eps"""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(eps, steps, "--s", color="green")
    ax2.plot(eps, costs, ":o", color="orange")
    ax1.set_xscale("log")
    ax1.set_xlabel("eps")
    ax1.set_ylabel("liczba kroków", color="green")
    ax2.set_ylabel("koszt", color="orange")
    plt.show()

eps_cost_steps_plot(epss, costs, lengths)

# najkorzystniejsze epsilon +- 9

theta_best, history = gradient_descent(J, dJ, theta_start, X, y, alpha=0.0007, eps=9)
print("parametry krzywej regresyjnej: \n", theta_best)
example_x = [50,100,200]
for x in example_x:
    print("na",x, "pożarów", h(theta_best, x), "włamań")

