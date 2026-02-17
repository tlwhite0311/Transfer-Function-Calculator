from sympy import (diff, Function, laplace_transform, laplace_correspondence, laplace_initial_conds, symbols, abc, Eq,
                   solve, simplify)

v_c, V_C, v, V = symbols('v_c, V_C, v, V', cls=Function)
s, L, R, C = symbols('s, L, R, C')
t = symbols('t', real=True)

# create and display differential equation of the system
LHS = L*C*diff(v_c(t), t, 2) + R*C*diff(v_c(t), t) + v_c(t)
RHS = v(t)
Eq(LHS, RHS)

# create and display the transformed equation into the s domain
lhs_lt = laplace_transform(LHS, t, s, noconds=True)
rhs_lt = laplace_transform(RHS, t, s, noconds=True)

lhs_lt_corr = laplace_correspondence(lhs_lt, {v_c: V_C})
rhs_lt_corr = laplace_correspondence(rhs_lt, {v: V})

lhs_init_conds = laplace_initial_conds(lhs_lt_corr, t, {v_c: [0, 0, 0]})
rhs_init_conds = laplace_initial_conds(rhs_lt_corr, t, {v: [0, 0, 0]})

eqn_transformed = Eq(lhs_init_conds, rhs_init_conds)


sol_of_V_s = solve(eqn_transformed, V(s))[0]


tf = 1/sol_of_V_s*V_C(s)

print(tf)