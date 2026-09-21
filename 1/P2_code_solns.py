import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 2, 1, 3], dtype=float)
y = np.array([0.5, 3.5, 1.5, 7.5], dtype=float)
n = len(x)

def MSE_line(m, c):
    return np.mean((m*x + c - y)**2)

def MSE_parabola(a, b, c):
    return np.mean((a*(x**2) + b*x + c - y)**2)

def solve_line_analytical():
    # solve by using matrix
    """
    [m1 c1] [m] = [xy]
    [m2 c2] [c] = [y]
    """
    #to find coefficients of m & c in dmse/dm & dmse/dc
    A = np.array([
        [np.sum(x**2),  np.sum(x)], #dmse/dm
        [np.sum(x),     n]  #dmse/dc
    ])
    B = np.array([
        np.sum(x*y),
        np.sum(y)
    ])
    m, c = np.linalg.solve(A, B)
    MSE = MSE_line(m, c)
    return m, c, MSE

def solve_parabola_analytical():
    A = np.array([
        [np.sum(x**4),  np.sum(x**3),   np.sum(x**2)],
        [np.sum(x**3), np.sum(x**2),   np.sum(x)],
        [np.sum(x**2),  np.sum(x),      n]
    ])
    B = np.array([
        np.sum((x**2)*y),
        np.sum(x*y),
        np.sum(y)
    ])
    a, b, c = np.linalg.solve(A, B)
    MSE = MSE_parabola(a, b, c)
    return a,b, c, MSE

def solve_line_numerical(m0=-10.0, c0=-10.0, tolerance=1e-5, max_iter=100):
    m, c = m0, c0
    history = [(m, c, MSE_line(m, c))]
    d2_dm2 = 7
    d2_dc2 = 2
    for i in range(max_iter):
        dm = 7*m + 3*c - 31/2
        m_new = m - dm/d2_dm2

        dc = 3*m_new + 2*c - 13/2
        c_new = c - dc/d2_dc2

        if abs(m_new - m)<tolerance and abs(c_new - c)<tolerance:
            m, c = m_new, c_new
            history.append((m,c, MSE_line(m, c)))
            break

        m, c = m_new, c_new
        history.append((m, c, MSE_line(m, c)))
    return m, c, MSE_line(m, c), history

def solve_parabola_numerical(a0=-10.0, b0=-10.0, c0=-10.0, tolerance=1e-5, max_iter=100):
    a, b, c = a0, b0, c0
    history = [(a, b, c, MSE_parabola(a, b, c))]
    d2_da2 = 49
    d2_db2 = 7
    d2_dc2 = 2
    for i in range(max_iter):
        da = 49*a + 18*b + 7*c - 83/2
        a_new = a - da/d2_da2

        db = 18*a_new +7*b +3*c - 31/2
        b_new = b - db/d2_db2

        dc = 7*a_new + 3*b_new + 2*c - 13/2
        c_new = c - dc/d2_dc2

        if abs(a_new - a)<tolerance and abs(b_new - b)<tolerance and abs(c_new - c)<tolerance:
            a, b, c = a_new, b_new, c_new
            history.append((a, b,c, MSE_parabola(a, b, c)))
            break

        a, b, c = a_new, b_new, c_new
        history.append((a, b,c, MSE_parabola(a, b, c)))
    return a, b, c, MSE_parabola(a, b, c), history
#linear line
m1, c1, mse_line1 = solve_line_analytical()
m2, c2, mse_line2, line_hist = solve_line_numerical()
#parabola
a_a, b_a, c_a, mse_parabola1 = solve_parabola_analytical()
a_n, b_n, c_n, mse_parabola2, parabola_hist = solve_parabola_numerical()

plt.figure()
plt.scatter(x, y, color='black', label='Data Points')
x_smooth = np.linspace(-0.5, 3.5, 100)
#after getting those constants, use the x values in x_smooth to get the corresponding y values
y_line1 = m1*x_smooth + c1
y_parabola1 = a_a*(x_smooth**2) + b_a*x_smooth + c_a
plt.plot(x_smooth, y_line1, 'r--', label=f'Line Fit (MSE={mse_line1:.3f})')
plt.plot(x_smooth, y_parabola1, 'b--', label=f'Parabola Fit (MSE={mse_parabola1:.3f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line and Parabola Curve Fits')
plt.grid(True)
plt.legend()
plt.savefig('fitted_curves.jpg')
plt.close()

print("Done!")