import tensorflow as tf
from keras.utils.vis_utils import plot_model

import numpy as np
from fractions import Fraction
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from maraboupy import Marabou, MarabouUtils
from maraboupy import MarabouCore
# t-SNE visualization 

incorrect_points = pd.read_csv('h.csv')
X_embedded = TSNE(n_components=2).fit_transform(incorrect_points)
print(X_embedded)

kmeans = KMeans(n_clusters=5).fit(X_embedded)

plt.scatter(X_embedded.T[0],X_embedded.T[1])
plt.scatter(kmeans.cluster_centers_.T[0], kmeans.cluster_centers_.T[1])
plt.show()

points = []

for x, y in zip(X_embedded.T[0],X_embedded.T[1]):
    current_index = -1
    best_distance = np.inf
    for i in range(len(kmeans.cluster_centers_.T[0])):
        curr_dist = np.sqrt((x - kmeans.cluster_centers_.T[0][i]) * \
             (x - kmeans.cluster_centers_.T[0][i]) + \
             (y - kmeans.cluster_centers_.T[1][i]) * \
             (y - kmeans.cluster_centers_.T[1][i]))

        if curr_dist <= best_distance:
            current_index = i
            best_distance = curr_dist

    points.append(current_index)

incorrect_points['Cluster'] = points

'''
Timing
LP4_DIMM_RON
LP4_SOC_ODT
ICOMP
CTLE_C
CTLE_R
'''

data = [[29.0/62.0, 4438438805624748448999337308207763882634487451851301174787195442861508844855043.0/14931908842363365481082399272710292486093281706904523288379370042402241764864640.0, 1.0, 3.0/16.0, 3.0/16.0, 148654145722939449575863060394596206172576682020318207372048057473528740826043397.0/ 164250997265997020291906391999813217347026098775949756172173070466424659413511040.0],
[Fraction(0.4758064516129032), Fraction(0.88), Fraction(0.7875), Fraction(1.0),  Fraction(0.45), Fraction(0.1875)],
[Fraction("29/62"), Fraction("148654145722939449575863060394596206172576682020318207372048057473528740826043397/164250997265997020291906391999813217347026098775949756172173070466424659413511040"), Fraction("4438438805624748448999337308207763882634487451851301174787195442861508844855043/14931908842363365481082399272710292486093281706904523288379370042402241764864640"),Fraction("1/1"), Fraction("3/10"),Fraction("3/16")]
,[Fraction("59/124"), Fraction("22/25"), Fraction("63/80"), Fraction("1.0"), Fraction("9/20"), Fraction("3/16")]] 
#data = incorrect_points[["n_Timing", "n_LP4_DIMM_RON","n_LP4_SOC_ODT","n_ICOMP","n_CTLE_C", "n_CTLE_R"]]



def relu(x):
    if x < 0:
        return Fraction(0.0)
    return Fraction(x)

def linear(x):
    return Fraction(x)

def m(a, b):
    ret = Fraction(0.)
    for ax, bx in zip(a,b):
        ax = Fraction(ax)
        bx = Fraction(bx)
        ret += ax * bx
    return ret

def transpose(arr):
    return np.array(arr).T.tolist()

tf_results = []

i = 0
#data = data.to_numpy()
for d in data:
    x = np.array(list(map(lambda x: float(x),d)))
    x = x.reshape((1,-1))

    new_model = tf.keras.models.load_model('.')
    tf_results.append(new_model.predict(x))
    i += 1





layer1w = [[-0.08890143781900406, -1.3002418279647827, -1.2109712362289429, -0.6945668458938599, 0.019196180626749992, -0.005701065994799137, 1.1026861667633057, 0.13191932439804077, 0.35577598214149475, 0.09860929846763611, 0.4421059191226959, 0.29669567942619324], [0.44493311643600464, 0.02877730131149292, -0.05811310186982155, 0.25924137234687805, 0.399913489818573, -0.057366933673620224, 0.3172338604927063, -0.050938960164785385, -0.3961658775806427, -0.5948590040206909, -0.10297174751758575, 0.060411855578422546], [-0.10205871611833572, 0.012303481809794903, -0.022998131811618805, 0.12297612428665161, -0.19084478914737701, -1.0172311067581177, 0.165262833237648, -0.013768389821052551, -0.1046021580696106, -0.6198437213897705, 0.008288810960948467, 0.06950774043798447], [0.015924353152513504, 0.036563850939273834, -0.009511173702776432, 0.02626209706068039, -0.3279072344303131, -0.052484698593616486, -0.04099908098578453, -0.060946881771087646, -0.1408153623342514, -1.1569105386734009, 0.2777300477027893, 0.039331648498773575], [-0.0024525721091777086, -0.052179768681526184, -0.019908271729946136, -0.03219045698642731, 0.36681684851646423, 0.017640627920627594, -0.16749221086502075, 0.8600183725357056, 0.2826616168022156, -0.02439781464636326, 0.1785336285829544, -0.3643292784690857], [0.30215632915496826, -0.03434370085597038, -0.00803697481751442, 0.1203712448477745, -0.3312080502510071, 0.11175493896007538, 0.1412035971879959, 0.4068172872066498, -0.142279714345932, 0.4030674695968628, -0.15031297504901886, -1.796433925628662]]
layer1b = [-0.09060380607843399, 0.6138206720352173, 0.8539918065071106, 0.6073952317237854, 0.05925266072154045, 0.34397244453430176, -0.49556446075439453, -0.4423728585243225, 0.3016418516635895, -0.14107848703861237, -0.05472536012530327, 0.3067191243171692]
layer2w = [[-0.26751160621643066, -0.4335198700428009, 0.2450881153345108, -0.6017016768455505, -0.1606462299823761, -0.026504233479499817], [1.52304208278656, -0.23902064561843872, -0.4941531717777252, -0.02712055668234825, 0.8591068983078003, 0.5278595089912415], [-0.28793758153915405, 0.08308027684688568, 0.6994970440864563, 0.20362280309200287, -0.2471090406179428, 0.7303434610366821], [-0.24234287440776825, -0.21951669454574585, 0.588628888130188, 0.10316584259271622, 0.2942146062850952, 0.7812458276748657], [-0.2251725196838379, -2.995680570602417, 0.21603021025657654, -0.3771885335445404, 0.0890156701207161, -0.03789760172367096], [0.38434746861457825, 0.5210357904434204, -0.1023879200220108, 0.03165186941623688, 0.6609517335891724, 0.04274032264947891], [0.3625856935977936, 0.2746960520744324, -0.7341524362564087, -0.6102184653282166, 0.45284348726272583, -0.7471290826797485], [0.35038676857948303, -1.5727148056030273, -0.612347424030304, -0.37857791781425476, -0.6098761558532715, 0.04524724557995796], [-0.3499625027179718, -0.39043137431144714, 0.43123358488082886, -0.26532939076423645, 0.527114987373352, -0.1153956949710846], [-0.3322322368621826, -0.6754734516143799, 0.032224953174591064, 0.11093885451555252, -0.5595612525939941, -0.18320080637931824], [-0.16271387040615082, -0.6444727182388306, 0.11819768697023392, -0.47386282682418823, 0.5083276629447937, -0.2664438784122467], [0.8115272521972656, -0.10702082514762878, -0.1377408504486084, -0.0060714855790138245, 0.7799659371376038, -0.038739822804927826]]
layer2b = [0.08135966211557388, 0.21889625489711761, 0.48404428362846375, -0.29297158122062683, -0.3007769286632538, -0.09027373790740967]
layer3w = [[0.5940441489219666], [-0.5719011425971985], [0.9981497526168823], [0.26241040229797363], [-0.4677869975566864], [-1.0118790864944458]]
layer3b = [0.20477154850959778]

layer1w = transpose(layer1w)
layer2w = transpose(layer2w)
layer3w = transpose(layer3w)

z3_results = [] 

for d in data:
    input = d
    print("Layer 1")
    x1 = []
    for layer, b in zip(layer1w, layer1b):

        x1.append(m(input,layer) + Fraction(b))
    print(x1)
    z1 = list(map(relu, x1))
    print(z1)
    print("Layer 2")
    x2 = []

    for layer, b in zip(layer2w, layer2b):
        
        x2.append(m(z1,layer) + b)

    z2 = list(map(relu, x2))

    print("Layer 3")
    x3 = []

    for layer, b in zip(layer3w, layer3b):
        x3.append(m(z2,layer) + b)

    output = list(map(linear, x3))
    z3_results.append(output)


error = 0

def weird_function(x):
    return (x * 852.0/5.0 - 512.0/5.0 + 544.0/5.0) / (949.0 / 5.0)

def inv_weird_function(x):
    return (x * (949.0 / 5.0) + 512/5.0 - 544.0/5.0) / (852.0/5.0)


for t, z in zip(tf_results, z3_results):
    print("tf: {0},  z3:{1}".format(t, weird_function(float(z[0]))))
    error += abs(t- z)
#

print(weird_function(0.0))
print(weird_function(1.0))
z = input()

print(error)
for a in incorrect_points[["Timing", "LP4_DIMM_RON","LP4_SOC_ODT","ICOMP","CTLE_C", "CTLE_R"]]:
    print(a)

network = Marabou.read_tf(".",modelType="savedModel_v2")
for inputB in network.inputVars[0][0]:
    network.setLowerBound(inputB, 0)
    network.setUpperBound(inputB, 1.0)

normalize_dict = {"Timing":(124.0, -63.0),
    "LP4_DIMM_RON":(200.0, 40.0),
    "LP4_SOC_ODT":(40.0, 40.0),
    "ICOMP":(9.0, 6.0),
    "CTLE_C":(420.0, 0.0),
    "CTLE_R":(2400.0, 0.0)
}


def normalize(key, val):
    bounds = normalize_dict[key]
    return (val - bounds[1]) / bounds[0]


disjunctions = []
for i in range(-63,62):
    eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
    eq.addAddend(1.0, 0)
    eq.setScalar(normalize("Timing", i))
    disjunctions.append([eq])

network.addDisjunctionConstraint(disjunctions)

#################
## 0 and 1
#################

# LP4_DIMM_RON
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 1)
eq.setScalar(normalize("LP4_DIMM_RON", 40.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 1)
eq.setScalar(normalize("LP4_DIMM_RON", 240.0))
network.addEquation(eq)


# LP4_DIMM_ODT
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 2)
eq.setScalar(normalize("LP4_SOC_ODT", 15.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 2)
eq.setScalar(normalize("LP4_SOC_ODT", 80.0))
network.addEquation(eq)


# ICOMP
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 3)
eq.setScalar(normalize("ICOMP", 6.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 3)
eq.setScalar(normalize("ICOMP", 15.0))
network.addEquation(eq)

# CTLE_C
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 4)
eq.setScalar(normalize("CTLE_C", 0.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 4)
eq.setScalar(normalize("CTLE_C", 420.0))
network.addEquation(eq)

# CTLE_R
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 5)
eq.setScalar(normalize("CTLE_R", 0.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 5)
eq.setScalar(normalize("CTLE_R", 2400.0))
network.addEquation(eq)

#########################
## Restriction of variables
#########################

# Timing
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 0)
eq.setScalar(normalize("Timing", -4))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 0)
eq.setScalar(normalize("Timing", 6))
network.addEquation(eq)


# LP4_DIMM_RON
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 1)
eq.setScalar(normalize("LP4_DIMM_RON", 216.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 1)
eq.setScalar(normalize("LP4_DIMM_RON", 264.0))
network.addEquation(eq)


# LP4_DIMM_ODT
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 2)
eq.setScalar(normalize("LP4_SOC_ODT", 58.5))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 2)
eq.setScalar(normalize("LP4_SOC_ODT", 71.5))
network.addEquation(eq)


# ICOMP
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 3)
eq.setScalar(normalize("ICOMP", 13.5))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 3)
eq.setScalar(normalize("ICOMP", 16.5))
network.addEquation(eq)

# CTLE_C
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 4)
eq.setScalar(normalize("CTLE_C", 189.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 4)
eq.setScalar(normalize("CTLE_C", 231.0))
network.addEquation(eq)

# CTLE_R
eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 5)
eq.setScalar(normalize("CTLE_R", 450.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 5)
eq.setScalar(normalize("CTLE_R", 550.0))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.LE)
eq.addAddend(1, 6)
eq.setScalar(np.nextafter(float(0.7), -np.inf))
network.addEquation(eq)

eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EquationType.GE)
eq.addAddend(1, 6)
eq.setScalar(0.0)
network.addEquation(eq)


ipq = network.getMarabouQuery()
ipq.dump()
b, stats = network.solve()

print(b)

plot_model(new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)