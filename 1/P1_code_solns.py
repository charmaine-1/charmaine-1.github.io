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

def plot_newton(x0, y0, history, filename):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    #for the oteration table
    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(10, 6.5), gridspec_kw={'width_ratios': [3, 1]})

    ax.plot(x_vals, y_vals, label='$y = x^2 + 5$', color='blue')
    ax.scatter([x0], [y0], color='red', zorder=5, label=f'Target Point ({x0}, {y0})')

    prev_x = None
    table_data = []

    for n, x_hist in enumerate(history):
        y_hist = f(x_hist)
        symbol = f"S{n}"
        table_data.append([symbol, f"{x_hist:.5f}", f"{y_hist:.5f}"])

        if prev_x is None or abs(x_hist - prev_x) > 0.001 or n == len(history) - 1:
            ax.scatter([x_hist], [y_hist], color='orange', zorder=4)
            
            # Alternate arrow direction to avoid cluttering adjacent points
            x_offset = 20 if n % 2 == 0 else -40
            y_offset = 20 if n % 2 == 0 else -25
            
            ax.annotate(
                symbol, 
                xy=(x_hist, y_hist), 
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='orange', lw=1.2),
                fontsize=8,
                fontweight='bold'
            )
            ax.plot([x0, x_hist], [y0, y_hist], 'g--', alpha=0.2)
            prev_x = x_hist

    final_x = history[-1]
    final_d = math.sqrt((final_x - x0)**2 + (f(final_x) - y0)**2)

    ax.plot([x0, final_x], [y0, f(final_x)], 'r-', linewidth=2, label='Shortest Distance Line')
    ax.set_title(f'Newton-Raphson approach for pt ({x0}, {y0})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend(loc='upper right')

    #build side table
    ax_table.axis('off')
    table = ax_table.table(
        cellText = table_data,
        colLabels = ['Step', 'x', 'y'],
        loc = 'center',
        cellLoc = 'center'
    )
    table.scale(1, 1.2)
    table.set_fontsize(8)

    fig.text(0.05, 0.05, f'min d = {final_d:.5f} at x = {final_x:.5f}',
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) #make room for the min d line outside plot
    plt.savefig(filename)
    plt.close()

def plot_golden(x0, y0, history, filename):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, y_vals, label='$y = x^2 + 5$', color='blue')
    ax.scatter([x0], [y0], color='red', zorder=5, label=f'Target pt ({x0}, {y0})')

    steps_to_show = [0, len(history)//2, len(history)-1]
    colors = ['purple', 'orange', 'green']
    for n, step_n in enumerate(steps_to_show):
        a_val, b_val = history[step_n]
        ax.axvline(a_val, linestyle=':', color=colors[n], label=f'Bracket Step {step_n}: [{a_val:.2f}, {b_val:.2f}]')
        ax.axvline(b_val, linestyle=':', color=colors[n])
        
    best_x = (history[-1][0] + history[-1][1]) / 2
    final_d = math.sqrt((best_x - x0)**2 + (f(best_x) - y0)**2)
    ax.plot([x0, best_x], [y0, f(best_x)], 'r-', linewidth=2, label='Shortest Distance Line')
    
    ax.set_title(f'Golden Section Search Bracketing Steps for Point ({x0}, {y0})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.text(0.05, 0.05, f'min d = {final_d:.5f} at x = {best_x:.5f}', 
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(filename)
    plt.close()

points = [(0,0), (-4,0), (-8,0), (2,0), (6,0)]

for n, (x0, y0) in enumerate(points):
    d_newton, x_newton, hist_newton= find_distance_newton(x0, y0, f, df, ddf)
    d_golden, x_golden, hist_golden= golden_section_search(x0, y0, f, -10, 10)

    plot_newton(x0, y0, hist_newton, f'1/plots/newton_{n}.jpg')
    plot_golden(x0, y0, hist_golden, f'1/plots/golden_{n}.jpg')

    print(f"completed generating plots for ({x0}, {y0}) !")
