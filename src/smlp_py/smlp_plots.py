import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import cm
import seaborn as sns
from math import ceil
import os, sys
import numpy as np
import pandas as pd
#from logs_common import *
from sklearn.metrics import mean_squared_error, r2_score

# the block below defines two distribution plotting functions which do the same
# except one receives inst as the first argument which is used to retreive the
# output file prefix (including the directory path) while the other function 
# receives this output file prefix directly as a string. Usage of the latter
# function allows functions of these module to not be dependent on inst.
# Enforcing usage of the latter function: TODO !!!: remove the first version.
if [0,9] <= list(map(int, sns.__version__.split('.'))) < [0,10]:
    '''
    def distplot_dataframe(inst, y, resp_names, interactive):
        plt.figure('Distribution of response variables')
        cc = plt.cm.get_cmap('hsv')
        sns.distplot(y, hist=True, kde=False, bins=ceil(max(10, 50/len(resp_names))),
                     color=[cc(c/len(resp_names)) for c,n in enumerate(resp_names)])
        plt.legend(resp_names, loc='upper right')
        plot('resp-distr', interactive, inst._filename_prefix, block=False)
    '''    
    def response_distribution_plot(out_dir, y, resp_names, interactive):
        plt.figure('Distribution of response features')
        cc = plt.cm.get_cmap('hsv')
        sns.distplot(y, hist=True, kde=False, bins=ceil(max(10, 50/len(resp_names))),
                     color=[cc(c/len(resp_names)) for c,n in enumerate(resp_names)])
        plt.legend(resp_names, loc='upper right')
        plot('resp-distr', interactive, out_dir, block=False)
        
elif [0,10] <= list(map(int, sns.__version__.split('.'))) < [0,14]:
    '''
    def distplot_dataframe(inst, y, resp_names, interactive):
        plt.figure('Distribution of response features')
        for c in y:
            sns.distplot(y[c], hist=True, kde=False, bins=ceil(max(10, 50/len(resp_names))))
        plot('resp-distr', interactive, inst._filename_prefix, block=False)
    '''    
    def response_distribution_plot(out_dir, y, resp_names, interactive):
        plt.figure('Distribution of response features')
        for c in y:
            sns.distplot(y[c], hist=True, kde=False, bins=ceil(max(10, 50/len(resp_names))))
        plot('resp-distr', interactive, out_dir, block=False)
else:
    print('seaborn version',sns.__version__,'not supported; required: 0.9.x or 0.10.x',
          file=sys.stderr)
    sys.exit(1)


def plot(name, interactive, out_prefix=None, **show_kws):
    #print('saved figure filename: ', 'out_prefix', out_prefix, 'name', name)
    #print('interactive', interactive); print('show_kws', show_kws)
    if out_prefix is not None:
        #print('Saving plot ' + out_prefix + '_' + name + '.png')
        plt.savefig(out_prefix + '_' + name + '.png')
    if interactive:
        #print('HERE2', show_kws)
        plt.show(**show_kws)
    plt.clf()


def plot_data_columns(data):
   # columns = list(data)
    for c in data:
        #print(c,data[c])
        plt.figure() # start a new figure rather than draw on top of each other
        sns.distplot(data[c], hist=True, kde=False, bins=50)
        #print(c)

# see this link to understand why empty plots are created sometimes 
# https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
# for us it happens when {'block': False} is passed for show_kws; unclear where / how show_kws
# gets this value {'block': False} as plot() is not called with any extra arguments
def plot_true_pred_runtime1(y, y_pred, interactive, title, out_prefix=None, log_scale=False):
    #print("{1} msqe: {0:.3f}".format(mean_squared_error(y, y_pred), title))
    #print("{1} r2_score: {0:.3f}".format(r2_score(y, y_pred), title))

    #print("Train rmse: {0:.3f}".format(rmse(y_pred, y)))
    #print('y\n', y, 'y_pred\n', y_pred)
    if log_scale:
        y = np.log10(y)
        y_pred = np.log10(y_pred)

    #l_plot = np.linspace(y.min(), y.max(), 100)
    y_pred = pd.DataFrame(y_pred, columns=y.columns)
    for c in y.columns:
        # scale each plot individually based on min/max of y[c] (not that of y)
        l_plot = np.linspace(y[c].min(), y[c].max(), 100)
        #print('c=', c, 'x=', y[c].values, 'y=',y_pred[c].values, y_pred.columns)
        ax = sns.scatterplot(x=y[c].values, y=y_pred[c].values, marker='+', label=c)
        plot_title = title + '_resp_%s' % c
        ax.set_title(plot_title) # (title + ' col_%s' % c)
        ax.set_xlabel('True values')
        ax.set_ylabel('Predicted values')

        plt.gcf().set_size_inches(16, 9)

        plt.plot(l_plot, l_plot, color='r')
        #plt.ylim(0, max(y) * 1.1)
        #plt.xlim(0, max(y) * 1.1)
        plot('eval_' + title + '-col-%s' % c, interactive, out_prefix)
        #plot('eval-' + plot_title, out_prefix)

def plot_true_pred_runtime(y, y_pred, interactive, title, out_prefix=None, log_scale=False):
    try:
        plot_true_pred_runtime1(y, y_pred, interactive, title, out_prefix=out_prefix, log_scale=log_scale)
    except ValueError as e:
        print(e, file=sys.stderr)

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, interactive, out_prefix=None,
                   log_scale=False):
    ## Evaluate on training data
    plot_true_pred_runtime(y_train, model.predict(X_train), interactive, model_name + '_train',
                           out_prefix=out_prefix, log_scale=log_scale)

    ## Evaluate on test data
    plot_true_pred_runtime(y_test, model.predict(X_test), interactive, model_name + '_test',
                           out_prefix=out_prefix, log_scale=log_scale)


# plot error between response resp and its prediction pred. 
# Argument data_version denotes on which data was the prediction performed:
# usually can be training, test, entire labeled data on new, unseen data
def evaluate_prediction(model_name, resp, pred, data_version, 
        interactive=False, out_prefix=None, log_scale=False):
    if not resp is None and not pred is None:
        plot_true_pred_runtime(resp, pred, interactive, model_name + '_' + data_version,
                               out_prefix=out_prefix, log_scale=log_scale)
        
