import math
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("1/plots", exist_ok=True)

def f(x):
    return x**2 + 5

def df(x):
    return 2*x

def ddf(x):
    return 2

# Algorithm A: Newton-Raphson Method (Fastest if differentiable)
def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100):
    x = initial_guess
    history = [x]
    for n in range(max_iter):
        D_prime = 2*(x-x0) + 2*(f(x)-y0)*df(x)
        D_double_prime = 2 + 2*(df(x)**2) + 2*(f(x)-y0)*ddf(x)
        next_x = x - D_prime/D_double_prime
        history.append(next_x)
        if abs(next_x - x)<tolerance: break
        x = next_x
    shortest_distance = ((x-x0)**2 + (f(x)-y0)**2)**0.5
    return shortest_distance, x, history

# Algorithm B: Golden Section Search (Best if derivatives are unknown)
def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    phi = (1 + math.sqrt(5))/2
    resphi = 2 - phi
    x1 = a + resphi*(b-a)
    x2 = b - resphi*(b-a)

    def dist_sq(x): return (x-x0)**2 + (f(x)-y0)**2

    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    history = [(a, b)]
    while abs(b-a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi*(b-a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi*(b-a)
            f_x2 = dist_sq(x2)
        history.append((a, b))
    best_x = (a+b)/2
    return math.sqrt(dist_sq(best_x)), best_x, history

points = [(0,0), (-4,0), (-8,0), (2,0), (6,0)]
print(f"{'Point':<10} | {'Newton Dist':<12} | {'Golden Dist':<12} | {'Newton x':<10} | {'Golden x':<10}")
print("-" * 65)

for x0, y0 in points:
    d_newton, x_newton, _= find_distance_newton(x0, y0, f, df, ddf)
    d_golden, x_golden, _= golden_section_search(x0, y0, f, -10, 10)
    print(f"({x0:>2}, {y0})     | {d_newton:<12} | {d_golden:<12} | {x_newton:<10} | {x_golden:<10}")