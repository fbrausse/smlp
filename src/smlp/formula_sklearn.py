# utilities to convert sklearn tree and polynomial models into formulas / rules


import numpy as np
from sklearn.tree import _tree


# Generation and exporting model formulae for tree models, tree forests, and polynomial models
class SklearnFormula:
    # generate rules from a single decision or regression tree that predicts a single response
    def _get_rules(self, tree, feature_names, resp_names, class_names, rounding=-1):
        #logic_impl = '->'
        #logic_and = 'and'
        tree_ = tree.tree_
        #print(tree_.feature) ; print(_tree.TREE_UNDEFINED) ; print(feature_names)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #smlp.Var(feature_name)
        paths = []
        path = []

        def recurse(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                # path is a tuple of the form ['(feature1 > 0.425)', '(feature2 > 0.875)', (array([[0.19438973],[0.28151123]]), 1)]
                # where first n-1 elements of the list describe a branch in the tree and the last element has the array of response
                # values for that branch of the tree; the length of that array must coincide with the number of the responses.
                #print('path', path, 'path[-1][0]', path[-1][0], resp_names)
                assert len(path[-1][0]) == len(resp_names)
                #print('+++++++++ path :\n', path, path[-1][0]); 
                responses_values = [path[-1][0][i][0] for i in range(len(resp_names))]
                #print('responses_values', responses_values)
                if rounding > 0:
                    #response_value = np.round(response_value, rounding)
                    responses_values = np.round(responses_values, rounding)
                if len(resp_names) == 1:
                    rule += '(' + resp_names[0] + " = "+str(responses_values[0]) + ')'
                else:
                    conjunction = []
                    for i, rn in enumerate(resp_names):
                        conjunction.append('(' + rn + " = "+str(responses_values[i]) + ')')
                    rule += ' and '.join(conjunction)
                #print('rule', rule); #assert False
            else:
                # TODO: this branch has not been tested; likely will not work with multiple reonses as classes
                # in th enext line is defined as hard coded for resp_names[0] -- meaning, the last index [0].
                classes = path[-1][0][0]
                l = np.argmax(classes)
                # was hard-coded rounding: class_probability = np.round(100.0*classes[l]/np.sum(classes),2)
                class_probability = 100.0*classes[l]/np.sum(classes)
                if rounding > 0:
                    class_probability = np.round(class_probability, rounding)
                rule += f"class: {class_names[l]} (proba: {class_probability}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]

        return rules

    # Print on standard output or in file rules that describe a set of decision or regression trees that
    # predict a single response, by extracting the rules from each individual tree using get_rules().
    # Argument tree_estimators is a set of objects tree_est that as *.tree_ contain trees trained for 
    # predicting the response. Argument class_names specifies levels of the response in case of classificarion. 
    def trees_to_rules(self, tree_estimators, feature_names, response_names, class_names, log, rules_filename):
        if not rules_filename is None:
            save = True
            print('Writing tree rules into file ' + rules_filename)
            rules_file = open(rules_filename, "w")
        else:
            save = False

        # write a preamble: number of trees and tree semantics (how responses are computed using many trees)
        if log:
            print('#Forest semantics: {}\n'.format('majority vote'))
            print('#Number of trees: {}\n\n'.format(len(tree_estimators)))
        if save:
            rules_file.write('#Forest semantics: {}\n'.format('majority vote'))
            rules_file.write('#Number of trees: {}\n\n'.format(len(tree_estimators)))

        # traverse trees, generate and print rules per tree (each rule correponds to a full branch in the tree)
        for indx, tree_est in enumerate(tree_estimators):
            rules = self._get_rules(tree_est, feature_names, response_names, class_names)
            if log:
                print('#TREE {}\n'.format(indx))
                for rule in rules:
                    print(rule)
                print('\n')
            if save:
                rules_file.write('#TREE {}\n'.format(indx))
                for rule in rules:
                    rules_file.write(rule)
                    rules_file.write('\n')
        if save:
            rules_file.close()

    '''        
    # Print on standard output or in file rules that describe a set of decision or regression trees that
    # predict a single response; do that separately for each response in response_names, by applying function 
    # trees_to_rules_single_response() to each response in response_names. See that function for the
    # description of the arguments. 
    def trees_to_rules(tree_estimators, feature_names, response_names, class_names, log, rules_filename):
        for response_name in response_names:
            rules_filename_resp = rules_filename + '_' + response_name + "_tree_rules.txt"
            trees_to_rules_single_response(tree_estimators, feature_names, response_name, class_names, log, rules_filename_resp)
    '''        

    # print and export to file a plonomial model formula        
    def poly_model_to_formula(self, inputs, outputs, coefs, powers, resp_id, log, formula_filename):
        #print('Polynomial model coef\n', coefs.shape, '\n', coefs)
        #print('Polynomial model terms\n', powers.shape, '\n', powers)
        #print('Polynomial model inps\n', len(inputs), inputs)
        #print('Polynomial model outp\n', len(outputs), outputs)
        #assert False
        if len(inputs) != powers.shape[1]:
            raise Exception('Error in poly_model_to_formula')
        formula_str = ''
        for r in range(powers.shape[0]):
            #print('r', powers[r], 'coef', coefs[0][r])
            if coefs[resp_id][r] == 0:
                continue
            curr_term = str(coefs[resp_id][r])
            for i in range(len(inputs)):
                #print('i', inputs[i], coefs[resp_id][r], powers[r][i])
                if powers[r][i] == 0:
                    continue
                elif powers[r][i] == 1:
                    curr_term = curr_term + ' * ' + inputs[i]
                else:
                    curr_term = curr_term + ' * ' + inputs[i] + '^' + str(powers[r][i])
            #print('curr_term', curr_term)
            if formula_str == '':
                formula_str = curr_term
            else:
                formula_str = formula_str + ' + ' + curr_term

        # add the output name
        formula_str = outputs[resp_id] + ' = ' + formula_str

        # print thr formula as text
        if log:
            print('formula', formula_str)

        # save formula into file
        if not formula_filename is None:
            model_file = open(formula_filename, "w")
            model_file.write(formula_str)
            model_file.close()

        return formula_str