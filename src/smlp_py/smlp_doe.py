from pyDOE import *
# build failed, need to rebuild from designofexperiment import *
from doepy import build, read_write

from typing import Union
import os
import pandas as pd
import numpy as np
#import textwrap
from smlp_py.smlp_utils import list_unique_unordered


# What are main effects, simple effects, and interactions?
# Main effects are those that occur between an independent variable and a dependent response variable. 
# They are also called simple effects. Interaction effects are between two independent variables and 
# how their relationship influences a dependent variable.
class SmlpDoepy:
    def __init__(self):
        self._doepy_logger = None
        self.FULL_FACTORIAL = 'full_factorial'
        self.TWO_LEVEL_FRACTIONAL_FACTORIAL = 'fractional_factorial' #two_level_
        self.PLACKET_BURMAN = 'plackett_burman'
        self.SUKHAREV_GRID = 'sukharev_grid'
        self.BOX_BEHNKEN = 'box_behnken'
        self.BOX_BEHNKEN_CENTERS = 1
        self.BOX_WILSON = 'box_wilson'
        self.BOX_WILSON_CENTER = '2,2'
        self.BOX_WILSON_ALPHA = 'o'
        self.BOX_WILSON_FACE = 'ccf'
        self.LATIN_HYPERCUBE = 'latin_hypercube'
        self.LATIN_HYPERCUBE_SPACE_FILLING = 'latin_hypercube_sf'
        self.LATIN_HYPERCUBE_PROB_DISTR = 'Normal'
        self.RANDOM_K_MEANS = 'random_k_means'
        self.MAXMIN_RECONSTRUCTION = 'maximin_reconstruction'
        self.HALTON_SEQUENCE = 'halton_sequence'
        self.UNIFORM_RANDOM_MATRIX = 'uniform_random_matrix'
        self.TWO_LEVEL_DESIGNS = [self.TWO_LEVEL_FRACTIONAL_FACTORIAL, self.PLACKET_BURMAN, 
            self.SUKHAREV_GRID, self.BOX_WILSON, self.LATIN_HYPERCUBE, self.LATIN_HYPERCUBE_PROB_DISTR,
            self.RANDOM_K_MEANS, self.MAXMIN_RECONSTRUCTION, self.HALTON_SEQUENCE,
            self.UNIFORM_RANDOM_MATRIX]
        self.doepy_params_dict = {
            'doe_algo':{'abbr':'doe_algo', 'type':str, 
                'help':'Design of experiment (DOE) algorithm from doepy package. ' +
                    'The supported algorithms are: "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}"'.format(self.FULL_FACTORIAL, self.TWO_LEVEL_FRACTIONAL_FACTORIAL, self.PLACKET_BURMAN, self.SUKHAREV_GRID, self.BOX_BEHNKEN, self.BOX_WILSON, self.LATIN_HYPERCUBE, self.LATIN_HYPERCUBE_SPACE_FILLING, self.HALTON_SEQUENCE, self.UNIFORM_RANDOM_MATRIX)},
            'doe_factor_level_ranges': {'abbr':'doe_factor_level_ranges', 'type':dict, 
                'help':'A dictionary of levels per feature for building experiments for all ' +
                    'supported DOE algorithms. Here experiments are lists feature-value assignments ' +
                    '[(feature_1, value_1),...,(feature_n, value_n)], and they are rows of the matrix ' +
                    'of experiments returned by the supported DOE algorithms. The features are integer ' +
                    'features (thus the levels (values) are integers). The keys in that dictionary are ' +
                    'names of  features and the associated values are lists [val_1, .., val_k] ' +
                    'from which value for that feature are selected to build an experiment. ' +
                    'Example: {"Pressure":[50,60,70],"Temperature":[290, 320, 350],"Flow rate":[0.9,1.0]}. ' +
                    'DOE algorithms that work with two levels only treat these levels as the min and max '
                    'of the rage of a numeric variable. [default: {}]'.format(str(None))},
            'doe_num_samples': {'abbr':'doe_samples', 'type':int,
                'help':'Number of samples (experiments) to be generated [default: {}]'.format(str(None))},
            'doe_design_resolution':{'abbr':'doe_resolution', 'default':None, 'type':int,
                'help':'Desired design resolution. ' +
                    'The resolution of a design is defined as the length of the shortest ' +
                    'word in the defining relation. The resolution describes the level of ' +
                    'confounding between factors and interaction effects, where higher ' +
                    'resolution indicates lower degree of confounding. ' +
                    'For example, consider the 2^4-1-design defined by ' +
                    '   gen = "a b c ab" ' +
                    'The factor "d" is defined by "ab" with defining relation I="abd", where ' +
                    'I is the unit vector. In this simple example the shortest word is "abd" ' +
                    'meaning that this is a resolution III-design. ' +
                    'In practice resolution III-, IV- and V-designs are most commonly applied. ' +
                    '* III: Main effects may be confounded with two-factor interactions. ' +
                    '* IV: Main effects are unconfounded by two-factor interactions, but ' +
                    '      two-factor interactions may be confounded with each other. ' +
                    '* V: Main effects unconfounded with up to four-factor interactions, ' +
                    '     two-factor interactions unconfounded with up to three-factor ' +
                    '     interactions. Three-factor interactions may be confounded with ' +
                    '     each other. ' +
                    '[default: Half of the total feature count in doe_factor_level_ranges]'},
            'doe_spec_file': {'abbr':'doe_spec', 'type':str,
                'help':'File in csv format that specifies factor, level ranges used for ' +
                    'building design of experiment (DOE) samples using function sample_doepy(). ' +
                    'If not provided, a dictionary of factor / level ranges must be supplied to ' +
                    ' sample_doepy() directly instead of the file.'},
            'doe_box_behnken_centers': {'abbr':'doe_bb_centers', 'default':self.BOX_BEHNKEN_CENTERS, 'type':int, 
                'help':'Number of center points to include in the final design ' +
                    '[default: {}]'.format(str(self.BOX_BEHNKEN_CENTERS))},
            'doe_central_composite_center': {'abbr':'doe_cc_center', 'default':self.BOX_WILSON_CENTER, 'type':str,
                'help':'A 1-by-2 array of integers, the number of center points in each block of the design. ' +
                    '[default]'.format(str(self.BOX_WILSON_CENTER))},
            'doe_central_composite_alpha': {'abbr':'doe_cc_alpha', 'default':self.BOX_WILSON_ALPHA, 'type':str,
                'help':'A string describing the effect of alpha has on the variance. "alpha" can take on two ' +
                    'values: "orthogonal" or "o", and "rotatable" or "r" [default {}]'.format(str(self.BOX_WILSON_ALPHA))},
            'doe_central_composite_face': {'abbr':'doe_cc_face', 'default':self.BOX_WILSON_FACE, 'type':str,
                'help':'The relation between the start points and the corner (factorial) points. There are three ' +
                    'options for this input: ' + 
                    ' 1.   "circumscribed" or "ccc": This is the original form of the central composite design. The star ' +
                    '      points are at some distance "alpha" from the center, based on the properties desired for the ' +
                    '      design. The start points establish new extremes for the low and high settings for all factors. ' +
                    '      These designs have circular, spherical, or hyperspherical symmetry and require 5 levels for ' +
                    '      each factor. Augmenting an existing factorial or resolution V fractional factorial design ' +
                    '      with star points can produce this design. ' +   
                    ' 2.   "inscribed" or "cci": For those situations in which the limits specified for factor settings ' +
                    '      are truly limits, the CCI design uses the factors settings as the star points and creates a ' +
                    '      factorial or fractional factorial design within those limits (in other words, a CCI design ' +
                    '      is a scaled down CCC design with each factor level of the CCC design divided by "alpha" to ' +
                    '      generate the CCI design). This design also requires 5 levels of each factor. ' +
                    ' 3.   "faced" or "ccf": In this design, the star points are at the center of each face of the factorial ' +
                    '      space, so ''alpha" = 1. This variety requires 3 levels of each factor. Augmenting an existing ' +
                    '      factorial or resolution V design with appropriate star points can  also produce this design. ' +
                    '[default {}]'.format(str(self.BOX_WILSON_CENTER))},
            'doe_prob_distribution': {'abbr':'doe_prob_distr', 'default':self.LATIN_HYPERCUBE_PROB_DISTR, 'type':str,
                'help':'Analytical probability distribution to be applied over the randomized sampling. Takes strings: ' +
                    '"Normal", "Poisson", "Exponential", "Beta", "Gamma" [default {}]'.format(str(self.LATIN_HYPERCUBE_PROB_DISTR))}
        }
    '''
    usage of textwrap.dedent() to format text in argparse help messages is suggested here; does not work in 
    our setting as we use dictionaries rather than using arg[arse in the standard way.
    "doe_algo" full help message:
    textwrap.dedent('Design of experiment (DOE) algorithm from doepy package. ' +
                    'The supported algorithms are: ' +
                    self.FULL_FACTORIAL + ': Builds a full factorial design dataframe ' + 
                    ' from a dictionary doe_factor_level_ranges of factor/level ranges. ' +
                    self.TWO_LEVEL_FRACTIONAL_FACTORIAL + ': Builds a 2-level fractional ' +
                    'factorial design dataframe from a dictionary doe_factor_level_ranges ' +
                    'of factor/level ranges and given resolution. ' +
                    'Parameters: ' +
                    ' *  doe_factor_level_ranges: Dictionary of factors and ranges. ' +
                    '    Only min and max values of the range are required. If more than two ' + 
                    '    levels are given, the extreme values will be set to the low/high levels. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]} ' +
                    ' *  resolution: int, desired design resolution. ' +
                    '    [default: half of the total factor count] ' +
                    'Notes: ' +
                    '    The resolution of a design is defined as the length of the shortest ' +
                    '    word in the defining relation. The resolution describes the level of ' +
                    '    confounding between factors and interaction effects, where higher ' +
                    '    resolution indicates lower degree of confounding. ' +
                    '    For example, consider the 2^4-1-design defined by ' +
                    '       gen = "a b c ab" ' +
                    '    The factor "d" is defined by "ab" with defining relation I="abd", where ' +
                    '    "I" is the unit vector. In this simple example the shortest word is "abd" ' +
                    '    meaning that this is a resolution III-design. ' +
                    '    In practice resolution III-, IV- and V-designs are most commonly applied. ' +
                    '    * III: Main effects may be confounded with two-factor interactions. ' +
                    '    * IV: Main effects are unconfounded by two-factor interactions, but ' +
                    '         two-factor interactions may be confounded with each other. ' +
                    '   * V: Main effects unconfounded with up to four-factor interactions, ' +
                    '         two-factor interactions unconfounded with up to three-factor ' +
                    '         interactions. Three-factor interactions may be confounded with ' +
                    '         each other. ' +
                    'Example: ' +
                    ' *  >>> d1 = {"A":[1,5],"B":[0.3,0.7],"C":[10,15],"D":[3,7],"E":[-2,-1]}' +
                    '    >>> build_frac_fact_res(d1,3) ' +
                    '         A    B     C    D    E ' +
                    '    0  1.0  0.3  10.0  7.0 -1.0 ' +
                    '    1  5.0  0.3  10.0  3.0 -2.0 ' +
                    '    2  1.0  0.7  10.0  3.0 -1.0 ' +
                    '    3  5.0  0.7  10.0  7.0 -2.0 ' +
                    '    4  1.0  0.3  15.0  7.0 -2.0 ' +
                    '    5  5.0  0.3  15.0  3.0 -1.0 ' +
                    '    6  1.0  0.7  15.0  3.0 -2.0 ' +
                    '    7  5.0  0.7  15.0  7.0 -1.0 ' +
                    ' ' +
                    '    It builds a dataframe with only 8 rows (designs) from a dictionary with ' +
                    '    6 factors. A full factorial design would have required 2^6 = 64 designs.' +
                    self.PLACKET_BURMAN +  
                    ' Builds a Plackett-Burman design dataframe from a dictionary of factor/level ' +
                    'ranges. Only min and max values of the range are required. ' +
                    'Example of the dictionary which is needed as the input: ' +
                    '{"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]} ' +
                    'Notes: ' +
                    'Plackett–Burman designs are experimental designs presented in 1946 by ' +
                    'Robin L. Plackett and J. P. Burman while working in the British Ministry of Supply. ' +
                    '(Their goal was to find experimental designs for investigating the dependence of some ' +
                    'measured quantity on a number of independent variables (factors), each taking L levels, ' +
                    'in such a way as to minimize the variance of the estimates of these dependencies using ' +
                    'a limited number of experiments. Interactions between the factors were considered negligible. ' +
                    'The solution to this problem is to find an experimental design where each combination of ' +
                    'levels for any pair of factors appears the same number of times, throughout all the ' +
                    'experimental runs (refer to table). A complete factorial design would satisfy this ' +
                    'criterion, but the idea was to find smaller designs. These designs are unique in that ' +
                    'the number of trial conditions (rows) expands by multiples of four (e.g. 4, 8, 12, etc.). ' +
                    'The max number of columns allowed before a design increases the number of rows is always ' +
                    'one less than the next higher multiple of four. ' +
                    self.SUKHAREV_GRID + ' Builds a Sukharev-grid hypercube design dataframe from a dictionary ' +
                    'of factor/level ranges. ' +
                    'Parameters: ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]} ' +
                    '  * num_samples: Number of samples to be generated. Number of samples raised ' +
                    '    to the power of (1/dimension), where dimension is the number of variables, ' +
                    '    must be an integer. Otherwise num_samples will be increased to meet this criterion' +
                    'Notes: ' +
                    'Special property of this grid is that points are not placed on the boundaries of ' +
                    'the hypercube, but at centroids of the  subcells constituted by individual samples. ' +
                    'This design offers optimal results for the covering radius regarding distances ' +
                    'based on the max-norm. ' +
                    self.BOX_BEHNKEN + ' Builds a Box-Behnken design dataframe from a dictionary of factor/level ' +
                    'ranges. ' + 
                    'Parameters: ' +
                    ' *  factor_level_ranges: 3 levels of factors are necessary. If not given, the function ' +
                    '    will automatically create 3 levels by linear mid-section method. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,60,70],"Temperature":[290, 320, 350],"Flow rate":[0.9,1.0,1.1]}' +
                    ' *  center: The number of center points to include [default 1] ' +   
                    'Notes: ' +
                    'In statistics, Box–Behnken designs are experimental designs for response surface ' +
                    'methodology, devised by George E. P. Box and Donald Behnken in 1960, to achieve ' +
                    'the following goals: ' +
                    ' *  Each factor, or independent variable, is placed at one of three equally spaced values,' +
                    '    usually coded as −1, 0, +1. (At least three levels are needed for the following goal.) ' +
                    ' *  The design should be sufficient to fit a quadratic model, that is, one containing ' +
                    '    squared terms, products of two factors, linear terms and an intercept. ' +
                    ' *  The ratio of the number of experimental points to the number of coefficients in the ' +
                    '    quadratic model should be reasonable (in fact, their designs kept it in the range of ' +
                    '    1.5 to 2.6).*estimation variance should more or less depend only on the distance from ' +
                    '    the centre (this is achieved exactly for the designs with 4 and 7 factors), and should ' +
                    '    not vary too much inside the smallest (hyper)cube containing the experimental points. ' +
                    self.BOX_WILSON + 'Builds a Box-Wilson central-composite design dataframe from a dictionary ' +
                    'of factor/level ranges. '
                    'Notes: ' +
                    'In statistics, a central composite design is an experimental design, useful in response ' +
                    'surface methodology, for building a second order (quadratic) model for the response variable ' +
                    'without needing to use a complete three-level factorial experiment. ' +
                    'The design consists of three distinct sets of experimental runs: ' +
                    ' *  A factorial (perhaps fractional) design in the factors studied, each having two levels; ' +
                    ' *  A set of center points, experimental runs whose values of each factor are the medians of ' +
                    '    the values used in the factorial portion. This point is often replicated in order to ' +
                    '    improve the precision of the experiment; ' +
                    ' *  A set of axial points, experimental runs identical to the centre points except for one factor, ' +
                    '    which will take on values both below and above the median of the two factorial levels, and ' +
                    '    typically both outside their range. All factors are varied in this way. ' +
                    'Parameters: ' +
                    ' * factor_level_ranges: dictionary of factors and ranges. Only min and max values of the range ' +
                    '    are required. Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]} ' + 
                    ' * center  A 1-by-2 array of integers, the number of center points in each block of the design. ' +
                    '    [default (2, 2)] ' +
                    ' * alpha:  A string describing the effect of alpha has on the variance. "alpha" can take on the ' +
                    '    following values: "orthogonal", or "o", and "rotatable", or "r" [default "o"] ' +
                    ' * face: The relation between the start points and the corner (factorial) points. There are three ' +
                    '   options for this input: ' + 
                    '   1. "circumscribed" or "ccc": This is the original form of the central composite design. The star ' +
                    '      points are at some distance "alpha" from the center, based on the properties desired for the ' +
                    '      design. The start points establish new extremes for the low and high settings for all factors. ' +
                    '      These designs have circular, spherical, or hyperspherical symmetry and require 5 levels for ' +
                    '      each factor. Augmenting an existing factorial or resolution V fractional factorial design ' +
                    '      with star points can produce this design. ' +   
                    '   2. "inscribed" or "cci": For those situations in which the limits specified for factor settings ' +
                    '      are truly limits, the CCI design uses the factors settings as the star points and creates a ' +
                    '      factorial or fractional factorial design within those limits (in other words, a CCI design ' +
                    '      is a scaled down CCC design with each factor level of the CCC design divided by "alpha" to ' +
                    '      generate the CCI design). This design also requires 5 levels of each factor. ' +
                    '   3. "faced" or "ccf": In this design, the star points are at the center of each face of the factorial ' +
                    '      space, so ''alpha" = 1. This variety requires 3 levels of each factor. Augmenting an existing ' +
                    '      factorial or resolution V design with appropriate star points can  also produce this design. ' +
                    self.LATIN_HYPERCUBE + 'Builds simple Latin Hypercube from a dictionary of factor/level ranges. ' +
                    'Parameters ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]}.  ' +
                    '  * num_samples: Number of samples to be generated. ' +
                    '  * prob_distribution: Analytical probability distribution to be applied over the ' +
                    '    randomized sampling. Takes strings: "Normal", "Poisson", "Exponential", "Beta", "Gamma". ' +
                    'Notes: ' +
                    'Latin hypercube sampling (LHS) is a form of stratified sampling that can be applied to '
                    'multiple variables. The method commonly used to reduce the number or runs necessary for ' +
                    'a Monte Carlo simulation to achieve a reasonably accurate random distribution. ' +
                    'LHS can be incorporated into an existing Monte Carlo model fairly easily, and work ' +
                    'with variables following any analytical probability distribution. ' +
                     self.LATIN_HYPERCUBE_SPACE_FILLING + 'Builds a space-filling Latin Hypercube design ' +
                    'dataframe from a dictionary of factor/level ranges. ' +
                    'Parameters: ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input:. ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]}. ' +
                    '  * num_samples: Number of samples to be generated. ' +
                    'Notes: ' +
                    'Unlike ' + self.LATIN_HYPERCUBE + ' method, the space filling method does not use ' + 
                    ' probability distributions of the involved parameters. ' +
                    self.RANDOM_K_MEANS + 'Builds designs with random _k-means_ clusters from a dictionary ' +
                    'of factor/level ranges. ' +
                    'Parameters: ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input:  ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]}.' +
                    '  * num_samples: Number of samples to be generated. ' +
                    'Notes: ' +
                    'This function aims to produce a centroidal Voronoi tesselation ' +
                    'of the unit random hypercube and generate k-means clusters. ' +
                    self.MAXMIN_RECONSTRUCTION + 'Builds maximin reconstruction matrix from a dictionary ' +
                    'of factor/level ranges. ' +
                    'Parameters: ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]}. ' +
                    '  * num_samples: Number of samples to be generated ' +
                    'Notes: ' +
                    'This algorithm carries out a user-specified number of iterations to maximize the minimal ' +
                    'distance of a point in the set to: ' +
                    '  * other points in the set, ' +
                    '  * existing (fixed) points, ' + 
                    '  * the boundary of the hypercube. ' +
                    self.HALTON_SEQUENCE + 'Builds Halton matrix based design from a dictionary of ' +
                    'factor/level ranges. ' +
                    'Parameters: ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]}. '+
                    '  * num_samples: Number of samples to be generated. ' +
                    'Notes: ' +
                    'Builds a quasirandom dataframe from a dictionary of factor/level ranges using ' +
                    'prime numbers as seed. Quasirandom sequence using the default initialization ' +
                    'with first n prime numbers equal to the number of factors/variables. ' +
                    self.UNIFORM_RANDOM_MATRIX + 'Builds uniform random design matrix from a dictionary ' +
                    'of factor/level ranges. ' +
                    'Parameters: ' +
                    '  * factor_level_ranges: Only min and max values of the range are required. ' +
                    '    Example of the dictionary which is needed as the input: ' +
                    '    {"Pressure":[50,70],"Temperature":[290, 350],"Flow rate":[0.9,1.0]}. ' +
                    '  * num_samples: Number of samples to be generated. ' +
                    'Notes: ' +
                    'Builds a design dataframe with samples drawn from uniform random distribution ' +
                    'based on a dictionary of factor/level ranges. ')
            
            doe_factor_level_ranges full help message:
            'A dictionary of levels per feature for building experiments for all ' +
                    'supported DOE algorithms. Here experiments are lists feature-value assignments' +
                    '[(feature_1, value_1),...,(feature_n, value_n)], and they are rows of the matrix ' +
                    'of experiments returned by the supported DOE algorithms. The features are integer ' +
                    'features (thus the levels (values) are integers). The keys in that dictionary are ' +
                    'names of  features and the associated values are lists [val_1, .., val_k] ' +
                    'from which value for that feature are selected to build an experiment.' +
                    'Example: {"Pressure":[50,60,70],"Temperature":[290, 320, 350],"Flow rate":[0.9,1.0]}. ' +
                    'DOE algorithms that work with two levels only treat these levels as the min and max '
                    'of the rage of a numeric variable. [default: {}]'.format(str(None)
    '''
    # set logger from a caller script
    def set_logger(self, logger):
        self._doepy_logger = logger 
    
    def get_doe_results_file_name(self, report_file_prefix):
        return report_file_prefix + '_doe.csv'
    
    # Drop nan values in doe_spec_dict; they might originate from missing values in doe_spec csv file.
    # If doe_algo requires two level doe spec, drop all levels except min and max and isssue a warning; 
    # otherwise sort all values. Return the processed doe dictionary.
    def _process_doe_spec(self, doe_algo, doe_spec_dict):
        two_level_design = doe_algo in self.TWO_LEVEL_DESIGNS
        for k, v in doe_spec_dict.items():
            v_nonan = list(filter(lambda x: not np.isnan(x), doe_spec_dict[k]))
            v_nonan_unique = list_unique_unordered(v_nonan)
            if len(v_nonan) > len(v_nonan_unique):
                self._doepy_logger.warning('factor ' + str(k) + 
                    ' contains duplicate elements in the DOE spec; they will be dropped')
            if two_level_design:
                if len(v_nonan_unique) > 2:
                    self._doepy_logger.warning('factor ' + str(k) + 
                        ' contains more than two levels in the DOE spec; only min / max values will be used')
                vmn = min(v_nonan)
                vmx = max(v_nonan)
                doe_spec_dict[k] = [vmn, vmx]
            else:
                doe_spec_dict[k] = v_nonan
                doe_spec_dict[k].sort()
        return doe_spec_dict
        
    # main doypu function, applies doe_algo to generate experiemntal design (tests) in a smart way.
    # All supported doe algorithms require doe_spec as an argument to specify sampling points for 
    # each feature in the data, and most functions also take num_samples as argument to spcify 
    # how many tests (featuer-value tuples) to generate, while for other algorithms this number
    # is determined directly from doe_spec. Argument report_file_prefix, after adding suffix .csv, 
    # is path to the output file where the denerated design / tests dataframe are saved.
    def sample_doepy(self, doe_algo:str, doe_spec, num_samples:int, report_file_prefix:str, #:Union([dict, str])
                prob_distribution:str, fractional_factorial_resolution:int, 
                central_composite_center, central_composite_face:str, 
                central_composite_alpha:str, box_behnken_centers:int): #Union([dict, str])
        if type(doe_spec) == str:
            doe_spec_fname = doe_spec# + '.csv'
            if os.path.isfile(doe_spec_fname):
                doe_spec_dict = pd.read_csv(doe_spec_fname); #print('doe_spec_df\n', doe_spec_dict)
                doe_spec_dict = doe_spec_dict.to_dict(orient='list'); #print('doe_spec_df\n', doe_spec_dict)
            else:
                raise Exception('DOE levels grid file ' + str(doe_spec) + ' does not eist')
        elif type(doe_spec) == dict:
            doe_spec_dict = doe_spec
        else:
            raise Exception('doe_spec argument in function sample_doepy is ' + 
                str(type(doe_spec)) + ' (must be either file path or a dictionary')
        doe_spec_dict = self._process_doe_spec(doe_algo, doe_spec_dict)
        #print('doe_spec_dict after processing\n', doe_spec_dict)
        
        if doe_algo == self.FULL_FACTORIAL:
            doe_out_df = build.full_fact(doe_spec_dict)
        elif doe_algo == self.TWO_LEVEL_FRACTIONAL_FACTORIAL:
            doe_out_df = build.frac_fact_res(doe_spec_dict, fractional_factorial_resolution)
        elif doe_algo == self.PLACKET_BURMAN:
            doe_out_df = build.plackett_burman(doe_spec_dict)
        elif doe_algo == self.SUKHAREV_GRID:
            doe_out_df = build.sukharev(doe_spec_dict, num_samples=num_samples)
        elif doe_algo == self.BOX_BEHNKEN:
            for v in doe_spec_dict.values():
                assert len(v) == 2 or len(v) == 3
            doe_out_df = build.box_behnken(doe_spec_dict, box_behnken_centers)
        elif doe_algo == self.BOX_WILSON:
            #print('central_composite_center', type(central_composite_center), central_composite_center)
            assert isinstance(central_composite_center, str)
            # central_composite_center is a string of a form a,b where a and be are integers. 
            # We need to convert this string to 1-by-2 np.ndarray object 
            center_pair = central_composite_center.split(",")
            center_pair = ([int(e) for e in center_pair])
            assert len(center_pair) == 2
            center_pair = np.array(center_pair); #print(type(center_pair), center_pair)
            doe_out_df = build.central_composite(doe_spec_dict, center_pair, 
                central_composite_alpha, central_composite_face)
        elif doe_algo == self.LATIN_HYPERCUBE:
            doe_out_df = build.lhs(doe_spec_dict, num_samples=num_samples, 
                prob_distribution=prob_distribution)
        elif doe_algo == self.LATIN_HYPERCUBE_SPACE_FILLING:
            doe_out_df = build.space_filling_lhs(doe_spec_dict, num_samples=num_samples)
        elif doe_algo == self.RANDOM_K_MEANS:
            doe_out_df = build.random_k_means(doe_spec_dict, num_samples=num_samples)
        elif doe_algo == self.MAXMIN_RECONSTRUCTION:
            doe_out_df = build.maximin(doe_spec_dict, num_samples=num_samples)
        elif doe_algo == self.HALTON_SEQUENCE:
            doe_out_df = build.halton(doe_spec_dict, num_samples=num_samples)
        elif doe_algo == self.UNIFORM_RANDOM_MATRIX:
            doe_out_df = build.uniform_random(doe_spec_dict, num_samples=num_samples)
        else:
            raise Exception('Unsupported DOE algorithm ' + str(doe_algo))
        #print('doe_out_df\n', doe_out_df); 
        self._doepy_logger.info('DOE table with ' + str(doe_out_df.shape[0]) + ' entries has been generated')
        doe_out_df.to_csv(self.get_doe_results_file_name(report_file_prefix), index=False)
        return doe_out_df

        
'''
        doepy_example_dict = {'Pressure':[40, 50, 70], 'Temperature':[290, 320, 350], 'FlowRate':[0.2, 0.3, 0.2], 'Time':[5, 8, 5]}
        doepy_example_df = pd.DataFrame.from_dict(doepy_example_dict); #print('doepy_example_df\n', doepy_example_df); 
        doe_out_df = doeInst.sample_doepy(doeInst.LATIN_HYPERCUBE_SPACE_FILLING, doepy_example_dict, num_samples=100); #print('doe_out_df\n', doe_out_df); 
        doe_out_df = build.space_filling_lhs(doepy_example_dict, num_samples=100); #print('doe_out_df\n', doe_out_df); 
        assert False
        ranges_dict = {'a':[5,7], 'b':[-1,1], 'c':[0,1]}
        ranges_df = pd.DataFrame.from_dict(ranges_dict);
        ranges_df.to_csv('.../doe_ranges.csv', index=False)
        doe_ranges_dict = read_write.read_variables_csv('.../doe_ranges.csv')
        #print('doe_ranges_dict\n', doe_ranges_dict)
        doe_out_df = build.space_filling_lhs(doe_ranges_dict, num_samples=100); #print('doe_out_df\n', doe_out_df); 
        assert False
        #example_ranges_df = read_write.read_variables_csv('ranges.csv'); 
        read_write.write_csv(
            build.space_filling_lhs(read_write.read_variables_csv('ranges.csv'),
            num_samples=100),
            filename='DOE_table.csv'
        )
        ff2n(3)
        #print(levels)
        abc = fullfact(levels); #print('abc', abc)
        assert False
'''
        