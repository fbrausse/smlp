
# INFO: File Structure

- src/smlp_py/NN_verifiers: contains scripts to test the marabou models and verify the validity of the conversion of pb files into h5 files.
- src/smlp_py/marabou: helper files that contain examples of maraboupy commands.
- src/smlp_py/smtlib:
	- parser.py & smt_to_pysmt.py contain helper functions. They are not used in SMLP's pipeline.
	- text_to_sympy.py: Contains all the logic of converting, simplifying and reformatting expressions to a state that is easily translated into marabou expressions.
- src/smlp_py/solvers: This folder will contain all the logic of the external neural network verifiers that are going to be integrated into the SMLP pipeline.
- src/smlp_py/vnnlib: Contains the logic for for the need in the future to utilise the VNNLIB format (solver agnostic, several solvers support this format) to interact with the solvers.


# INFO: The abstract solver
abstract_solver.py defines an abstract solver class that is used to interface all the functionalities that all integrated solvers must support. This is because the main flow has been updated to reference the abstract solver functionalities, and thus all functions must be overridden by every new solver.

Some methods are optionally overridden as their content usually cover most use cases. 

# HOW: Integrating multiple solvers
In the solvers/ folder, each solver must have it own subfolder. Currently, z3 and marabou are the only supported solvers. Each solver must have 2 files:
1) Operations.py :  this file contains the functions that are utilised during the formula building and processing part of the workflow. For example, the method #smlp_and contains the conjunction logic utilised within the formula building phase, in marabou's case, the conjunction operations take place by using pysmt. Whereas is z3's case, the operators library is used to manage the formulas. 
2) solver.py : This file contains the class that extends our abstract solver and operations class. Consequently, it will have to override all the functions mentioned in the abstract solver class in order to function properly. 


# INFO: How the universal solver class is used in the main worklfow
The current PR contains changes in multiple files that contain core functionalities used in the main SMLP workflow. Re-usable parts of the flow or certain cases that are handled differently for each solver have been moved into the z3 solver and have been replaced with the universal solver class's functions. 
The universal solver (universal_solver.py) acts as the intermediary and eventually point to the specified solver (marabou, z3), depending on the given "version" argument. Possible values are:

```
Solver.Version.PYSMT
Solver.Version.FORM2

```

# INFO: Pysmt processing
The file #text_to_sympy .py contains all the logic required to transform the formulas into marabou queries. 
My dissertation can be used as a reference point to understand the underlying methodologies used inside each function. 