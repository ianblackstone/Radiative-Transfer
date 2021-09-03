import sympy as sp

F_cl = sp.Symbol('F_cl')
G = sp.Symbol('G')
Sigma_g = sp.Symbol('Sigma_g')
Sigma_new = sp.Symbol('Sigma_new')
Sigma_old = sp.Symbol('Sigma_old')
r = sp.Symbol('r')
H_old = sp.Symbol('H_old')
v = sp.symbols('v', cls=sp.Function)
c = sp.Symbol('c')
t = sp.Symbol('t')

diffeq = sp.Eq( (F_cl/(c*Sigma_g*t) - G / 2 * ( Sigma_g + Sigma_new + Sigma_old * (r/H_old))),v(r)*v(r).diff(r))

_, v = sp.dsolve(diffeq,v(r))



