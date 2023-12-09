import smlp

# Solver class; currently used only to set an external solver to SMLP
class SmlpSolver:
    def __init__(self):
        self._solver = None
        self._solver_path = None
        self._DEF_SOLVER = 'z3'
        self._DEF_SOLVER_PATH = None
        self.solver_params_dict = {
            'solver': {'abbr':'solver', 'default':self._DEF_SOLVER, 'type':str,
                'help':'Solver to use in model exploration modes "verify," "query", "optimize" and "tune". ' +
                        '[default: {}]'.format(str(self._DEF_SOLVER))},
            'solver_path': {'abbr':'solver_path', 'default': self._DEF_SOLVER_PATH, 'type':str,
                'help':'Path to solver to use in model exploration modes "verify," "query", "optimize" and "tune". ' +
                        '[default: {}]'.format(str(self._DEF_SOLVER_PATH))}
        }
    
    def set_solver(self, solver:str):
        self._solver = solver
        
    def set_solver_path(self, solver_path:str):
        if solver_path is not None:
            smlp.options({'inc_solver_cmd': solver_path})
        
        
    
    