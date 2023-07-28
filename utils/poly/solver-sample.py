import smlp

def print_result(res):
    if isinstance(res, smlp.sat):
        print('SAT with model')
        for var, cnst in res.model.items():
            print(' ', var, '=', smlp.Cnst(cnst))
    elif isinstance(res, smlp.unsat):
        print('UNSAT')
    else:
        print('unknown:', res.reason)

# solver instance, 'True' means: enable incremental solving
slv = smlp.solver(True)

# define some domain
dom = smlp.domain({
    'x': smlp.component(smlp.Real, interval=[0,10]),
    'y': smlp.component(smlp.Integer, grid=[1,4,7]),
})

# initialize the solver with the domain, now it knows about variables 'x' and 'y'
slv.declare(dom)

# get the variables as terms
x = smlp.Var('x')
y = smlp.Var('y')

# add some constraints to the solver
slv.add((x > smlp.Cnst(5)) | (y < smlp.Cnst(5)))
slv.add(x > y)

# solve the problem
res = slv.check()
print_result(res)

slv.add(y > smlp.Cnst(8))
res = slv.check()
print_result(res)
