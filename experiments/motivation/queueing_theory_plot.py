import numpy as np
import matplotlib.pyplot as plt

def alpha(r):
    return ((-8 + r ** 2 + np.sqrt(64 - 96 * r + 56 * (r ** 2) - 12 * (r ** 3) + r ** 4))
            /(3 * (-2 * r + r ** 2)))

def beta(r):
    return (r - np.sqrt(8 - 4 * r + r ** 2))/(-2 + r)

if __name__ == "__main__":
    x = np.linspace(0.0, 2.0, 1000)
    y_alpha = alpha(x)
    y_beta = beta(x)
    # print("y_alpha", y_alpha)
    # print("y_beta", y_beta)
    plt.figure(figsize=(3, 2))
    plt.plot(x, y_alpha, label=r"$\alpha$")
    plt.plot(x, y_beta, label=r"$\beta$")
    plt.xlabel(r"$\lambda D$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("queueing_theory.pdf")