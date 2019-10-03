#!/usr/bin/env python
### See details in [Jiang & Luo 2016]

import numpy as np
import sys, argparse, math, os.path, time
from functools import partial
from scipy.integrate import quad
from scipy.interpolate import interp1d, BarycentricInterpolator
from scipy.linalg import solve

parser = argparse.ArgumentParser(description='Solver for the plane Couette-flow problem')
parser.add_argument('-Nu', '--uniform', type=int, default=8, help='uniform division of interval')
parser.add_argument('-Ng', '--geometric', type=int, default=2, help='geometric division of last uniform interval')
parser.add_argument('-d', '--degree', type=int, default=10, help='polynomial degree used in each block')
parser.add_argument('-g', '--geom-ratio', type=float, default=0.5, help='ratio of the geometric progression')
parser.add_argument('-k', '--knudsen', type=float, default=0.1, help='modified Knudsen number')
parser.add_argument('-p', '--interp-points', type=int, default=100, help='for interpolation of Abramowitz functions')
parser.add_argument('-y', '--every', type=int, default=1, help='print every n line of result')
parser.add_argument('-e', '--exact-abra', action='store_true', help='use exact values for the Abramowitz functions')
parser.add_argument('--skip-generalized', action='store_true', help='use the same quadrature rule for diagonal blocks')
parser.add_argument('-t', '--tests', action='store_true', help='run some tests instead')
parser.add_argument('-a', '--plot-abra', action='store_true', help='plot the Abramowitz functions with asymptotics')
parser.add_argument('-q', '--plot-quad', action='store_true', help='plot all used quadratures')
parser.add_argument('-m', '--plot-matrix', action='store_true', help='plot matrix of kernel')
parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
args = parser.parse_args()

if args.verbose or args.plot_abra or args.plot_quad:
    import matplotlib.pyplot as plt

int_xln = lambda n, x: x**(n+1)/(n+1)*(np.log(x) - 1/(n+1))
int_ln0 = partial(int_xln, 0)
int_ln1 = partial(int_xln, 1)
int_ln2 = partial(int_xln, 2)
maxv = lambda x: np.max(np.abs(x))
minv = lambda x: np.min(np.abs(x))

class fixed:
    L = .5
    a = np.sqrt(np.pi)
    log_xmin = -8
    log_xmax = +3
    exact_tol = 1e-15

def interp_log(f):
    X = np.logspace(fixed.log_xmin, fixed.log_xmax, args.interp_points)
    quadvec = lambda f: np.vectorize(lambda x: np.log(quad(partial(f, x), 0, np.infty)[0]))
    # TODO: use Chebyshev polynomials instead of interp1d
    func = interp1d(X, quadvec(f)(X), kind='cubic', fill_value=-np.infty, bounds_error=True)
    return lambda x: np.exp(func(x))

def abra(verbose=False, only_zeros=False):
    a, b = np.sqrt(np.pi), 1.5*np.euler_gamma
    g0 = { -1: np.infty, 0: a/2, 1: 0.5, 2: a/4 }
    if only_zeros:
        return g0
    j = lambda n, x, t: np.exp(-t*t - x/t)*t**n     # integrand of the Abramowitz function
    g = {                                           # asymptotes for small x
        -1: lambda x: -np.log(x/(1+np.exp(b)*x)) - b,
        0: lambda x: g0[0] + x/(1+x) * (np.log(x) + b-1),
        1: lambda x: g0[1] * np.ones_like(x),
        2: lambda x: g0[2] * np.ones_like(x),
    }
    kk = g.keys()
    J0 = dict(zip(kk, [ interp_log(partial(j, n)) for n in kk ]))
    if args.exact_abra:
        J = dict(zip(kk, [ np.vectorize(partial(lambda n, x: quad(partial(j, n, x), 0, np.infty)[0], n)) for n in kk ]))
    else:
        reg_J = dict(zip(kk, [ interp_log(lambda x, t: j(n, x, t) / g[n](x)) for n in kk ]))
        J = dict(zip(kk, [ partial(lambda n, x: reg_J[n](x) * g[n](x), n) for n in kk ]))
    if verbose:
        X = np.logspace(fixed.log_xmin, fixed.log_xmax, 2*args.interp_points)
        for n in kk:
            plt.plot(X, J[n](X), label='J({})'.format(n))
            plt.plot(X, g[n](X), 'k--', lw=.5)
        plt.plot(X, np.abs(J[-1](X) - J0[-1](X)), label='J(-1)_J0(-1)')
        plt.loglog()
        plt.legend()
        plt.show()
    return J

def intervals():
    X_uni = np.linspace(0, fixed.L, args.uniform+1)
    a, b = X_uni[-2], X_uni[-1]
    h1 = (b-a) / (1-args.geom_ratio**args.geometric) * (1-args.geom_ratio)
    X = np.hstack((X_uni[:-1], a + np.cumsum(h1*args.geom_ratio**np.arange(args.geometric))))
    return X

### For int_0^1 P(y) dy
def gauss_laguerre():
    y, w = np.polynomial.legendre.leggauss(args.degree)
    return (y+1)/2, w/2

def check_generalized(quad, x, verbose, a=0, b=1):
    assert(a >= 0 and b <= 1 and a < x and x < b)
    tests = [
        ( lambda x, y: x+y,                 lambda x: (b-a)*(x + (b+a)/2)           ),
        ( lambda x, y: np.abs(x-y),         lambda x: ((x-a)**2 + (b-x)**2)/2       ),
        ( lambda x, y: np.log(np.abs(x-y)), lambda x: int_ln0(x-a) + int_ln0(b-x)   ),
    ]
    for integrand, exact in tests:
        assert math.isclose(quad(partial(integrand, x)), exact(x), rel_tol=fixed.exact_tol)
    if verbose:
        print('Generalized Gauss quadrature for x={} is correct'.format(x))

### For int_0^1 [ P1(y) + |x-y|P2(y) + P3(y)ln|x-y| ] dy
def generalized_gauss(collocation_points, verbose=False):
    p, result1, result2 = args.degree, [], []
    interp = BarycentricInterpolator(collocation_points)
    for n in range(p//2):
        filename = 'weights{}.txt'.format(n+1)
        with open(filename) as f:
            header = f.readline()
            x = float(header.split()[-1])
            # check if quadrature is for the given collacation point
            assert math.isclose(x, collocation_points[n], rel_tol=fixed.exact_tol)
        y, w = np.loadtxt(filename, unpack=True)
        check_generalized(lambda f: np.sum(w*f(y)), x, verbose)
        I, b1, b2 = np.eye(collocation_points.size), [], []
        for i in range(collocation_points.size):
            interp.set_yi(I[i])
            b1.append(interp(y))
            b2.append(interp(1-y[::-1]))
        result1.append(( y, w, np.array(b1) ))
        result2.append(( 1-y[::-1], w[::-1], np.array(b2) ))
    return result1 + result2[::-1]

### Map quadrature from [0, 1] to [a, b]
def map_quad(quad, a, b):
    x, w = quad
    return (b-a)*x + a, (b-a)*w

def make_block(nondiag_quad, diag_quad, kernel, X, i, j):
    x, _ = map_quad(nondiag_quad, X[i], X[i+1])
    if args.verbose:
        sys.stdout.write('Construction progress: {:02.0f}%\r'.format((i*X.size+j)*100/X.size**2))
        sys.stdout.flush()
    if i == j:
        block = []
        for m, (y, w, b) in enumerate(diag_quad):
            y, w = map_quad((y, w), X[j], X[j+1])
            block.append(np.einsum('l,nl,l->n', w, b, kernel(x[m], y)))
        bl = np.array(block)
        return np.array(block)
    else:
        y, w = map_quad(nondiag_quad, X[j], X[j+1])
        return np.einsum('n,nm->mn', w, kernel(*np.meshgrid(x, y)))

def make_x(nondiag_quad, X, i):
    return map_quad(nondiag_quad, X[i], X[i+1])[0]

def make_w(nondiag_quad, X, i):
    return map_quad(nondiag_quad, X[i], X[i+1])[1]

def splot(X, A):
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    xv, yv = np.meshgrid(X, X, sparse=False, indexing='ij')
    ax.plot_wireframe(xv, yv, A)

def construct_matrix(kernel, verbose=False):
    X = intervals()
    J = abra()
    nondiag_quad = gauss_laguerre()
    collocation_points = nondiag_quad[0]
    if args.skip_generalized:
        diag_quad = [ (nondiag_quad[0], nondiag_quad[1], np.eye(args.degree)) for i in range(args.degree) ]
    else:
        diag_quad = generalized_gauss(collocation_points)
    if verbose:
        print('Mesh with {0}x{0} blocks:'.format(X.size-1))
        H = X[1:] - X[:-1]
        print('Interval lengths: min={}, max={}'.format(minv(H), maxv(H)))

    if verbose:
        print(' -- This is a moment just before constructing matrix A'); start_time = time.time()
    A = np.block([[ make_block(nondiag_quad, diag_quad, kernel, X, i, j)
        for j in range(X.size-1) ] for i in range(X.size-1) ])
    if verbose:
        print(' -- Matrix A is constructed. Elapsed time:', time.time() - start_time)
    if args.plot_matrix:
        splot(XX, A)
        plt.show()
    XX = np.hstack([ make_x(nondiag_quad, X, i) for i in range(X.size-1) ])
    return XX, A

def solve_linalg(kernel, func, verbose=False):
    XX, A = construct_matrix(kernel, verbose)
    if verbose:
        print('Shape of the matrix:', A.shape)
        print(' -- This is a moment just before solving system Au = f'); start_time = time.time()
    I = np.identity(XX.size)
    U = solve(I - A, func(XX))
    if verbose:
        print(' -- System Au = f is solved. Elapsed time:', time.time() - start_time)
        print('Residual in L_inf:', np.max(np.dot(A, U) - func(XX)))
    return XX, U

def integrate_with_U(func, U):
    X = intervals()
    quad = gauss_laguerre()
    YY = np.hstack([ make_x(quad, X, i) for i in range(X.size-1) ])
    WW = np.hstack([ make_w(quad, X, i) for i in range(X.size-1) ])
    return np.sum(WW*func(YY)*U)

def plot_solution(XX, U, F, U_exact=None, XX_exact=None):
    if XX_exact is None:
        XX_exact = XX
    plt.plot(XX, U, 'x-', label='U')
    plt.plot(XX, F, '*-', label='F')
    if U_exact is not None:
        plt.plot(XX_exact, U_exact, 'k.--', label='exact', lw=0.5, markersize=1)
    plt.xlim(0, fixed.L)
    plt.ylim(0, fixed.L)
    plt.legend()
    plt.show()

def print_section(title):
    print(''.join(['-' for i in range(50)]))
    print('---', title)

def check_test(n, kernel, func, exact=None):
    print_section('Test #{}.'.format(n))
    XX, U = solve_linalg(kernel, func)
    if exact is None:
        exact = lambda x: x
    if args.verbose:
        plt.title('Test #{}'.format(n))
        plot_solution(XX, U, func(XX), exact(XX))
    assert math.isclose(exact(XX[-1]), U[-1], rel_tol=fixed.exact_tol)

if args.plot_abra:
    print_section('Abramowitz functions')
    J = abra(verbose=True)

if args.plot_quad:
    print_section('Gauss quadratures')
    x, w = gauss_laguerre()
    params = { 'lw': 0.5, 'marker': '.', 'markersize': 2, 'ls': '-' }
    plt.plot(x, w, label='Gauss--Laguerre', **params)
    for n, (y, w, _) in enumerate(generalized_gauss(x, verbose=True)):
        plt.plot(y, w, label='Generalized Gauss (x={})'.format(x[n]), **params)
    plt.xlim(0, 1)
    plt.legend(loc='upper center')
    plt.show()

if args.tests:
    L = fixed.L
    tests = [
        ( lambda x, y: 0*(x+y),     lambda x: x                                     ),
        ( lambda x, y: 1 + 0*(x+y), lambda x: x - L**2/2                            ),
        ( lambda x, y: 0*x + y,     lambda x: x - L**3/3                            ),
        ( lambda x, y: x + 0*y,     lambda x: x*(1 - L**2/2)                        ),
        ( lambda x, y: x**2 - y**2, lambda x: x - (x*L)**2/2 + L**4/4               ),
        ( lambda x, y: x*y,         lambda x: x*(1 - L**3/3)                        ),
        ( lambda x, y: 0*x + y,     lambda x: x**3/L**2 - L**3/5,       lambda x: x**3/L**2             ),
        ( lambda x, y: np.abs(x-y),                     lambda x: -x**3/3 + x*(L**2/2 + 1) - L**3/3,    ),
        ( lambda x, y: np.log(np.abs(x-y))/y*(y-x)**2,  lambda x: x - int_ln2(x) - int_ln2(L-x)         ),
        ( lambda x, y: np.log(np.abs(x-y))/y,           lambda x: x - int_ln0(x) - int_ln0(L-x)         ),
    ]
    for n, params in enumerate(tests):
        check_test(n+1, *params)

else:
    J, J0, L, k = abra(), abra(only_zeros=True), fixed.L, args.knudsen
    if args.verbose:
        print_section('Couette-flow problem for k = {}'.format(k))
    kernel = lambda x, y: ( J[-1](np.abs(x-y)/k) - J[-1]((x+y)/k) ) / k / fixed.a
    func = lambda x: ( J[0]((L-x)/k) - J[0]((L+x)/k) ) / 2 / fixed.a
    XX, U = solve_linalg(kernel, func, args.verbose)
    shear = -2/fixed.a*( k*(J0[2] - J[2](1/k)) + 2*integrate_with_U(lambda y: J[1]((L-y)/k) - J[1]((L+y)/k), U) )
    XX, A = construct_matrix(lambda x, y: J[1](np.abs(x-y)/k) - J[1]((x+y)/k), args.verbose)
    qflux = .5/fixed.a*( J[2]((L-XX)/k) - J[2]((L+XX)/k) + 2/k*np.einsum('nm,m', A, U) ) - U/2
    np.savetxt(sys.stdout, np.transpose((XX, U, -shear*np.ones_like(XX), -qflux))[::args.every], fmt='%1.4e',
        header='       X        v_x      -p_xy       -q_x')

    # Compare with exact solution
    filename = '../k{}.txt'.format(k)
    if os.path.isfile(filename):
        XX_exact, U_exact, shear_exact = np.loadtxt('../k{}.txt'.format(k), unpack=True)
        U_interp = interp1d(XX, U, kind='linear')
        error_U = np.linalg.norm((U_interp(XX_exact[1:-1]) - U_exact[1:-1])/U_exact[1:-1])
        error_Pxy = np.abs(2*shear_exact[0] - shear)
        print('# k = {}, Nu = {}, Ng = {}, N_points = {}, error_U = {:.2e}, error_Pxy = {:.2e}'
            .format(k, args.uniform, args.geometric, XX.size, error_U, error_Pxy))
        if args.verbose:
            print(' -- Error:', (U_interp(XX_exact[1:-1]) - U_exact[1:-1])/U_exact[1:-1])
    else:
        U_exact = XX_exact = None
    if args.verbose:
        plot_solution(XX, U, func(XX), U_exact, XX_exact)
