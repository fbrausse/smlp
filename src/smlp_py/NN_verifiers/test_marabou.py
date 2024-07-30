from src.smlp_py.NN_verifiers.verifiers import MarabouVerifier
from src.smlp_py.smtlib.text_to_sympy import TextToPysmtParser

from pysmt.shortcuts import Symbol, And, Not, Or, Implies, simplify, LT, Real, Times, Minus, Plus, Equals, GE, ToReal, LE
from pysmt.typing import *
import tf2onnx
import numpy as np
from pysmt.shortcuts import Symbol, Times, Minus, Div, Real
from pysmt.smtlib.parser import get_formula
# from pysmt.oracles import get_logic
from pysmt.typing import REAL
from z3 import simplify, parse_smt2_string
import z3

from maraboupy.MarabouPythonic import *


if __name__ == "__main__":
    from keras.models import load_model

    # model = load_model("/home/kkon/Desktop/smlp/result/abc_smlp_toy_basic_nn_keras_model_complete.h5")
    # model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, opset=13, output_path="smlp_toy.onnx")
    print("SAVING TO ONNX")
    parser = TextToPysmtParser()
    parser.init_variables(symbols=[("x1", "real"), ('x2', 'real'), ('p1', 'real'), ('p2', 'real'),
                                  ('y1', 'real'), ('y2', 'real')])

    mb = MarabouVerifier(parser=parser)
    mb.init_variables(inputs=[("x1", "Real"), ('x2', 'Integer'), ('p1', 'Real'), ('p2', 'Integer')],
                      outputs=[('y1', 'Real'), ('y2', 'Real')])
    mb.initialize()

    smlp_formula = '(let ((|:0| (* (/ 281474976710656 2944425288877159) (- y1 (/ 1080863910568919 4503599627370496))))) (let ((|:1| (* (/ 281474976710656 2559564553220679) (- (* (/ 1 2) (+ y1 y2)) (/ 1170935903116329 1125899906842624))))) (>= (ite (< |:0| |:1|) |:0| |:1|) 1)))'
    smlp_str = f"""
                (declare-fun y1 () Real)
                (declare-fun y2 () Real)
                (assert {smlp_formula})
                """

    smlp_parsed = z3.parse_smt2_string(smlp_str)
    smlp_simplified = z3.simplify(smlp_parsed[0])
    ex = parser.parse(str(smlp_simplified))
    # ex = parser.replace_constants_with_floats_and_evaluate(ex)
    marabou_formula = parser.convert_ite_to_conjunctions_disjunctions(ex)
    print(marabou_formula.serialize())





    y1 = parser.get_symbol("y1_unscaled")
    y2 = parser.get_symbol("y2_unscaled")
    p1 = parser.get_symbol("p1_unscaled")
    p2 = parser.get_symbol("p2_unscaled")
    x1 = parser.get_symbol("x1_unscaled")
    x2 = parser.get_symbol("x2_unscaled")

    x2_int = parser.create_integer_disjunction("x2_unscaled", (-1, 1))
    p2_int = parser.create_integer_disjunction("p2_unscaled", (3, 7))
    # alpha = (((-1 <= x2) & (0.0 <= x1) & (x2 <= 1) & (x1 <= 10.0)) & (((p2 < 5) & (x1 == 10.0)) & (x2 < 12)))
    # beta = ((4 <= y1) & (6 <= y2))


    #  with x as input: y1==6.847101329531717 & y2==10.31207527363552
    #  with x as knob:  y1==4.120704402283359 &
    solution = And(
        Equals(x1, Real(10)),
        Equals(x2, Real(0)),
        Equals(p1, Real(2)),
        Equals(p2, Real(3))
    )

    theta = And(
        GE(p1, Real(6.8)),
        GE(p2, Real(3.8)),
        LE(p1, Real(7.2)),
        LE(p2, Real(4.2))
    )
    alpha = And(
        GE(x2, Real(-1)),
        LE(x2, Real(1)),
        GE(x1, Real(0.0)),
        LE(x1, Real(10.0)),
        And(
            LT(p2, Real(5)),
            Equals(x1, Real(10.0)),
            LT(x2, Real(12))
        )
    )

    beta = And(
        GE(y1, Real(4)),
        GE(y2, Real(8)),
    )

    not_beta = Or(
        LT(y1, Real(4)),
        LT(y2, Real(8))
    )
    eta = And(
        GE(p1, Real(0.0)),
        LE(p1, Real(10.0)),
        GE(p2, Real(3)),
        LE(p2, Real(7)),
        Or(
            p1.Equals(Real(2.0)),
            p1.Equals(Real(4.0)),
            p1.Equals(Real(7.0))
        )
    )
    mb.apply_restrictions(x2_int)
    mb.apply_restrictions(p2_int)
    # mb.apply_restrictions(beta)
    mb.apply_restrictions(alpha)
    mb.apply_restrictions(eta)
    # mb.apply_restrictions(marabou_formula)
    # mb.apply_restrictions(solution)

    # mb.apply_restrictions(theta)

    witness= mb.solve()
    print(witness)

##################  TEST PARSER ###########################
# if __name__ == "__main__":
#     parser = TextToPysmtParser()
#     parser.init_variables(inputs=[("x1", "real"), ('x2', 'int'), ('p1', 'real'), ('p2', 'int'),
#                                   ('y1', 'real'), ('y2', 'real')])
#
#     mb = MarabouVerifier(parser=parser)
#     mb.init_variables(inputs=[("x1", "Real"), ('x2', 'Integer'), ('p1', 'Integer'), ('p2', 'Integer')],
#                       outputs=[('y1', 'Real'), ('y2', 'Real')])
#
#
#     # (1<=(ite)) and (y<=4) and (y>=8)
#     # ite_without_ite = Or(And(c, t), And(Not(c), f))
#
#     y1 = parser.get_symbol("y1")
#     y2 = parser.get_symbol("y2")
#
#     ex = parser.parse('(y1+y2)/2')
#
#     c = y1 > y2
#     t = y1
#     f = y2
#     # ite_without_ite = Or(And(c, t), And(Not(c), f))
#
#     condition_true = Times(ToReal(c), y1)  # y1 if y1 > y2
#     condition_false = Times(ToReal(Not(c)), y2)  # y2 if y1 <= y2
#
#     # Combine them
#     ite_without_ite = Plus(condition_true, condition_false)
#
#     # Final expression (ite_without_ite >= 1)
#     inequality = GE(ite_without_ite, Real(1))
#
#     # Combine with the inequality 1 <= ITE(c, t, f)
#     # inequality = Real(1) <= ite_without_ite
#     print(inequality)


