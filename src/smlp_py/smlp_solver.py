import smlp
from smlp_py.smlp_utils import str_to_bool
                                
# Solver class; currently used only to set an external solver to SMLP
class SmlpSolver:
    def __init__(self):
        self._solver = None
        self._solver_path = None
        self._DEF_SOLVER = 'z3'
        self._DEF_SOLVER_PATH = None
        self._DEF_SOLVER_LOGIC = 'ALL'
        #self._DEF_SOLVER_INCREMENTAL = True
        
        '''
        When the logic is not given (None), the solver's default (which usually is something similar to ALL)
        is used. Except for Yices: it does not have a default, so you need to give a logic string. 
        
        Useful logics for our settings are:
        - QF_LIRA: quantifier-free linear integers/reals
        - QF_NIRA: quantifier-free polynomial integers/reals
        - QF_LRA / QF_NRA: like above, but without integers (enables faster/other solving strategies)
        - ALL: works for Z3, MathSAT, CVC4, CVC5, Yices
        '''
        self.solver_params_dict = {
            'solver': {'abbr':'solver', 'default':self._DEF_SOLVER, 'type':str,
                'help':'Solver to use in model exploration modes "verify," "query", "optimize" and "tune". ' +
                        '[default: {}]'.format(str(self._DEF_SOLVER))},
            'solver_path': {'abbr':'solver_path', 'default': self._DEF_SOLVER_PATH, 'type':str,
                'help':'Path to solver to use in model exploration modes "verify," "query", "optimize" and "tune". ' +
                        '[default: {}]'.format(str(self._DEF_SOLVER_PATH))},
            'solver_logic': {'abbr':'solver_logic', 'default': self._DEF_SOLVER_LOGIC, 'type':str,
                'help':'SMT2-lib theory with respect to which to solve model exploration task at hand, ' +
                        'in modes "verify," "query", "optimize" and "tune". ' +
                        '[default: {}]'.format(str(self._DEF_SOLVER_LOGIC))},
            #'solver_incr': {'abbr':'solver_incr', 'default': self._DEF_SOLVER_INCREMENTAL, 'type':str_to_bool,
            #    'help':'Should sover be used in incremental mode? ' +
            #            '[default: {}]'.format(str(self._DEF_SOLVER_INCREMENTAL))}
        }
    
    def set_solver(self, solver:str):
        self._solver = solver
        
    def set_solver_path(self, solver_path:str):
        if solver_path is not None:
            #print({'inc_solver_cmd': solver_path}); 
            smlp.options({'inc_solver_cmd': solver_path})
        
        
    
    