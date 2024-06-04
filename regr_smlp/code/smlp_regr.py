#!/usr/bin/env python3

import os
from os import path, chdir, sep, remove, listdir, kill
from argparse import ArgumentParser
from shutil import copytree, rmtree, copyfile
from csv import reader

from multiprocessing import Process, Queue, Lock
from subprocess import Popen, check_output, PIPE
from time import  time
import csv_comparator as csv_cmp

from threading import Timer

# from difflib import ndiff, context_diff

TUI_DIFF = 'diff'
GUI_DIFF = 'tkdiff'

DEBUG = False
# used for excluding from diff reports that involve randomness
files_to_ignore_from_diff = ['Test41_doe_two_levels_doe.csv', 'Test42_doe_two_levels_doe.csv']

RELEASE = False # to run regression with SMLP from release area
def ignored_files(src, filenames):
    """
    Copy code and regression scripts required to run regression from a different directory
    """
    return [filename for filename in filenames if
            not (filename.endswith('.py') or filename == "tests.csv" or filename.endswith('.exe') or filename.endswith('.json'))]


def get_all_files_from_dir(dir_path):
    return listdir(dir_path)


def fetch_test(test_id, tests_data):
    with open(tests_data, 'r') as rFile:
        csvreader = reader(rFile, delimiter=',')
        next(csvreader, None)
        for row in csvreader:
            if row[0] == test_id:
                return row
    raise NameError('Test {0} not found!'.format(test_id))


def conf_identifier(switches):
    return '-config' in switches


def get_conf_name(switches):
    i = switches.find('-config ')
    sub_switches = switches[i + 8:].strip()
    j = sub_switches.find(' -')
    if j != -1:
        return sub_switches[:j]
    else:
        return sub_switches


def get_conf_path(conf, path1):
    return path.join(path1, conf)

# This function works both with json and txt config files
def get_switches_with_conf(switches, path1):
    i = switches.find('-config ')
    sub_switches = switches[i + 8:].strip()
    j = sub_switches.find(' -')
    if j != -1:
        conf = sub_switches[:j]
    else:
        conf = sub_switches
    
    name_len = len(conf)
    conf = path.join(path1, conf)

    if j != -1:
        return switches[0:i + 8] + conf + switches[i + 8 + name_len:]
    else:
        return switches[0:i + 8] + conf


def extract_smlp_error(error_string):
    error_list = error_string.splitlines()
    for line in error_list:
        if line.startswith('Error:'):
            error_msg = line[7:]
            return error_msg
    return 'OK'


def mode_identifier(switches):
    #print('switches in mode identifier', switches)
    if '-mode ' in switches or '--mode ' in switches:
        #print('mode', switches.find('-mode '), switches.find('--mode '))
        if '--mode ' in switches:
            mode = switches.find('--mode ')
            if mode + 7 > len(switches) - 1:
                return 'no mode'
            mode_prefix3 = switches[mode + 7:mode + 10]; #print('mode_prefix3', mode_prefix3)
            mode_prefix4 = switches[mode + 7:mode + 11]; #print('mode_prefix4', mode_prefix4);
            mode = switches[mode + 7]
        else:
            mode = switches.find('-mode ')
            if mode + 6 > len(switches) - 1:
                return 'no mode'
            mode_prefix3 = switches[mode + 6:mode + 9]; #print('mode_prefix3', #mode_prefix3)
            mode_prefix4 = switches[mode + 6:mode + 10]; #print('mode_prefix4', #mode_prefix4)
            mode = switches[mode + 6]
        #print('mode', mode, 'mode_prefix3', mode_prefix3, 'mode_prefix4', mode_prefix4)
        if mode == 's':
            if mode_prefix3.startswith('syn'):
                return 'synthesize'
            else:
                return 'subgroups'
        if mode == 't':
            if mode_prefix3.startswith('tra'):
                return 'train'
            elif mode_prefix3.startswith('tun'):
                print('mode tune was renamed');
                assert False
                return 'tune'
            else:
                raise Exception("Unknovn mode prefix " + str(mode_prefix3))
            #return 'train'
        elif mode == 'p':
            return 'prediction'
        elif mode == 'f':
            return 'features'
        elif mode == 'o':
            if mode_prefix4.startswith('opti'):
                return 'optimize'
            elif mode_prefix4.startswith('opts'):
                return 'optsyn'
            else:
                print('unknown mode prefix', mode_prefix4);
                assert False
        elif mode == 'v':
            return 'verify'
        elif mode == 'q':
            return 'query'
        elif mode == 'c':
            return 'certify'
        elif mode == 'l':
            return 'level'
        elif mode == 'r':
            return 'representatives'
        elif mode == 'd':
            if mode_prefix3.startswith('dis'):
                return 'discretization'
            elif mode_prefix3 == 'doe':
                return 'doe'
            else:
                assert False
            return 'datainfo'
        elif mode == 'n':
            return 'novelty'
        else:
            return 'unknown'
    else:
        return 'no mode'
    
def spec_identifier(switches):
    if '-spec' in switches or '--spec' in switches:
        #print('spec', switches.find('-spec '), switches.find('--spec '))
        if '--spec ' in switches:
            spec = switches.find('--spec ')
            if spec + 7 > len(switches) - 1:
                return 'no spec'
            spec_id = '--spec '
        else:
            spec = switches.find('-spec ');
            spec_id = '-spec '
        #print('spec', spec, 'spec_id', spec_id)
        sub_switches = switches[spec + len(spec_id)-1:].strip(); #print('sub_switches', sub_switches)
        i = sub_switches.find(' -'); #print('i', i)
        if i != -1:
            return sub_switches[:i]
        else:
            return sub_switches

def solver_path_identifier(switches):
    option_short = '-solver_path '
    option_full = '--solver_path '
    if option_short in switches or option_full in switches:
        #print('solver', switches.find(option_short), switches.find(option_full))
        if option_full in switches:
            solver = switches.find(option_full); #print('solver', solver, len(option_full), len(switches))
            if solver + len(option_full) > len(switches) - 1:
                return 'no solver'
            #solver_prefix3 = switches[solver + len(option_full):solver + len(option_full)+3]; 
            solver = switches[solver + len(option_full)]; #
            sub_switches = switches[solver + (len((option_short))-1):].strip()
        else:
            solver = switches.find(option_short);
            sub_switches = switches[solver + (len((option_short))-1):].strip()
        #print('solver', solver); print('sub_switches', sub_switches)
        i = sub_switches.find(' -'); #print('i', i)
        if i != -1:
            return sub_switches[:i]
        else:
            return sub_switches

def use_model_identifier(switches):
    # return '-use_model' in switches
    if not ('-use_model' in switches or '--use_model ' in switches):
         return False
    if '--use_model' in switches:
         use_model = switches[switches.find('--use_model ') + len('--use_model ')]
    else:
         use_model = switches[ switches.find('-use_model ') + len('-use_model ')]
    
    if use_model.lower().startswith('t'):
            return True
    elif use_model.lower().startswith('f'):
            return False
    else:
            raise Exception('use_model option value cannot be identified')


def save_model_identifier(switches):
    #return '-save_model' in switches
    if not ('-save_model' in switches or '--save_model ' in switches):
         return False
    if '--save_model ' in switches:
        save_model = switches[switches.find('--save_model ') + len('--save_model ')]
    elif '-save_model ' in switches:
        save_model = switches[ switches.find('-save_model ') + len('-save_model ')]
    else:
        save_model = 't' # the default value for svae_model
    
    if save_model.lower().startswith('t'):
            return True
    elif save_model.lower().startswith('f'):
            return False
    else:
            raise Exception('save_model option value cannot be identified')

def get_model_name(switches):
    i = switches.find('-model_name ')
    sub_switches = switches[i + 12:].strip()
    i = sub_switches.find(' -')
    if i != -1:
        return sub_switches[:i]
    else:
        return sub_switches

# This function was adpated to work with json config files
def use_model_in_config(conf):
    with open(conf, 'r') as c:
        lines = c.readlines()
    for line in lines:
        ln = line.lower()
        #if '--use_saved_prediction_model true' in ln or '--use_saved_prediction_model t' in ln or '-use_model true' in ln or '-use_model t' in ln:
        # to work with json config file we are are looking to match slightly different patterns
        if '"use_model": "true"' in ln or '"use_model": "true"' in ln:
            return True
    return False


def main():
    start_time = time()
    file_path = path.dirname(path.abspath(__file__))
    # Regression arguments
    parser = ArgumentParser(description='SMLP regression')
    parser.add_argument('-o', '--output', help='Output directory.')
    parser.add_argument('-t', '--tests', help='Specify tests to run. It can be a comma-separated list of test numbers\
                        like -t 5,8,10; it can be a range of consecutive tests like -t 10:15; one can also run\
                        all toy tests, where toy means that the test data name starts with smlp_toy, by specifying\
                        -t toy; or run all other tests by specifying -t real; or run all the regression tests by\
                        specifying -t all.')
    parser.add_argument('-m', '--modes', help='Specify modes (e.g., verify) of tests to run, default is all modes.')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', '--print_command', action='store_true', help='print the command to run manually;\
                        the test will not be executed.')
    parser.add_argument('-diff', '--diff', action='store_true')
    #parser.add_argument('-c', '--cross_check', action='store_true', help='Cross check specific csv outputs.')
    parser.add_argument('-w', '--workers', help='Number of concurrent tests that will run, default 2.')
    #parser.add_argument('-temp', '--tempdir', help='Specify where to copy and run code, default=temp_dir.')
    parser.add_argument('-i', '--ignore_tests', help='Ignores test/s that are passed as this argument.')
    parser.add_argument('-n', '--no_all', action='store_true', help='Answer no to all file replacements/updates\
                        when a mismatch is found between current and master results.')
    parser.add_argument('-f', '--fail_txt', action='store_true', help='Don\'t compare all files if .txt main log\
                        file comparison fails.')
    parser.add_argument('-time', '--timeout', help='Set the timeout for each test to given value, if not provided,\
                        no timeout.')
    parser.add_argument('-tol', '--tolerance', help='Set the csv comparison tolerance to ignore differences in low\
                        decimal bits.')
    parser.add_argument('-def', '--default', help='Yes/No/Y/N answer to all master file replacements/updates.')
    parser.add_argument('-conf', '--config_default', help='Yes/No/Y/N answer to config file all replacements/updates.')
    parser.add_argument('-g', '--no_graphical_compare', action='store_true', help='Answer no on all replacing.')

    args = parser.parse_args()
    if not args.output:
        output_path = './' #file_path.replace('\\', '/')
    else:
        output_path = args.output.replace('\\', '/')
    if not args.tests:
        tests = 'all'
    else:
        tests = args.tests.replace(" ", "").replace("\'", "")
    if args.debug:
        debug = '-d 1'
    else:
        debug = ''
    ignored_tests = []
    if args.ignore_tests:
        if ',' in args.ignore_tests:
            ignored_tests = args.ignore_tests.replace(" ", "").replace("\'", "").split(',')
        else:
            ignored_tests.append(args.ignore_tests.replace(" ", "").replace("\'", ""))
    #print('ignored_tests', ignored_tests);
    relevant_modes = []
    if args.modes:
        if ',' in args.modes:
            relevant_modes = args.modes.replace(" ", "").replace("\'", "").split(',')
        else:
           relevant_modes.append(args.modes.replace(" ", "").replace("\'", ""))
    #print('relevant_modes',relevant_modes);

    """def read_txt_file_to_list(file_path):
        with open(file_path, 'r') as rFile:
            return rFile.read().splitlines()"""

    global DIFF
    if 'DISPLAY' in os.environ:
        DIFF = GUI_DIFF
    else:
        DIFF = TUI_DIFF

    code_path = file_path  # Path to SMLP regression code - also where smlp_tests.csv file and this script are located.

    tree_path = '../' # Path to regression location (where data, code, specs, master and model directories are located)
    solvers_path = '../../../external' # Path to external solvers
    # Create and migrate code to temp dir
    if False: # currently use development code w/o copying to temp area #not args.print_command:
        tempdir = 'temp'
        if args.tempdir:
            tempdir += args.tempdir
        elif tests in {'all', 'real', 'toy', 'test'}:
            tempdir += tests
        else:
            tempdir = 'temp_code4'
        temp_code_dir = path.join(tree_path, tempdir)  # Path of temp copied code dir.
        if path.exists(temp_code_dir):
            rmtree(temp_code_dir)
        copytree(dst=temp_code_dir, src=code_path, ignore=ignored_files)  # Copies code to temp dir.
        chdir(temp_code_dir)  # Changes working dir to temp code dir.
    else:
        temp_code_dir = code_path
    master_path = path.join(tree_path, 'master')  # Path to master results (to compare with)
    models_path = path.join(tree_path, 'models')  # Path to saved models and everything required to re-run it
    data_path = path.join(tree_path, 'data')      # Path to the data
    doe_path = path.join(tree_path, 'grids')      # Path to the doe grids data
    specs_path = path.join(tree_path, 'specs')    # Path to the domain spec for model exploration
    tests_data = path.join(temp_code_dir, 'smlp_regr.csv')  # Path of the tests config file
    
    
    diff = 'diff'

    if args.tolerance:
        csv_cmp.set_threshold(int(args.tolerance))
        
    tests_list = []
    tests_queue = Queue()
    print_lock = Lock()

    if tests == "all":
        with open(tests_data, 'r') as rFile:
            csvreader = reader(rFile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                if row[0] not in ignored_tests:
                    tests_list.append(row)
                    tests_queue.put(row)
    elif tests == 'toy':
        with open(tests_data, 'r') as rFile:
            csvreader = reader(rFile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                if (row[1].startswith('smlp_toy') or row[1].startswith('mlbt_toy') or row[2].startswith('smlp_toy') or row[
                    2].startswith('mlbt_toy') or (
                            conf_identifier(row[3]) and get_conf_name(row[3]).startswith('smlp_toy')) or (
                            not conf_identifier(row[3]) and row[1] == '' and row[2] == '')) and (
                        row[0] not in ignored_tests):
                    tests_list.append(row[0])
                    tests_queue.put(row)
    elif tests == 'real':
        with open(tests_data, 'r') as rFile:
            csvreader = reader(rFile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                if (not (row[1].startswith('smlp_toy') or row[1].startswith('mlbt_toy') or row[2].startswith('smlp_toy') or
                         row[2].startswith('mlbt_toy') or (
                                 conf_identifier(row[3]) and get_conf_name(row[3]).startswith('smlp_toy')))) and (
                        row[0] not in ignored_tests):
                    tests_list.append(row)
                    tests_queue.put(row)
    elif tests == 'test':
        with open(tests_data, 'r') as rFile:
            csvreader = reader(rFile, delimiter=',')
            next(csvreader, None)
            i_picks = ['36', '51', '60', '80', '95', '104', '120']
            for row in csvreader:
                if row[0] in i_picks:
                    tests_list.append(row)
                    tests_queue.put(row)
    elif ',' in tests:
        t_list = tests.split(',')
        for e in t_list:
            if ':' in e: # this option to support tests range, eg: 5:10
                t_list.remove(e)
                e_range = e.split(':')
                t_list = t_list + [str(e) for e in list(range(int(e_range[0]), int(e_range[1])+1))]
                #print('t_list', t_list)
        with open(tests_data, 'r') as rFile:
            csvreader = reader(rFile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                if row[0] in t_list:
                    tests_list.append(row)
                    tests_queue.put(row)
    elif ':' in tests: # this option to support tests range, eg: 5:10
        t_range = tests.split(':')
        start = t_range[0]
        end = t_range[1]
        t_list = [str(i) for i in range(int(start), int(end) + 1)]
        with open(tests_data, 'r') as rFile:
            csvreader = reader(rFile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                if row[0] in t_list:
                    tests_list.append(row)
                    tests_queue.put(row)
    else:
        #print('tests', tests, 'tests_data', tests_data)
        tests_list.append(fetch_test(tests,tests_data))
        tests_queue.put(fetch_test(tests,tests_data))


    

    """def fetch_test_outputs(test_id):
        test = fetch_test(test_id)
        new_prefix = 'Test' + test_id
        test_data = test[1]
        test_new_data = test[2]
        test_switches = test[3]
        test_type = mode_identifier(test_switches)
        return test_outputs(test_id, new_prefix, test_data, test_new_data, test_type, test_switches)"""

    '''
    Indexes for test list:
    0 - id
    1 - data
    2 - new data(if exists)
    3 - switches
    4 - description
    '''

    test_id_list = []
    test_out_queue = Queue()

    def kill_process(pr):
        pr.kill()

    def popen_timeout(command, timeout):
        p = Popen(command,shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        my_timer = Timer(timeout, kill_process, [p])
        cm = False
        try:
            my_timer.start()
            cm = p.communicate()
        finally:
            my_timer.cancel()
            return cm

    def worker(q, id_q, print_l):
        while True:
            if q.empty():
                return True
            test = q.get()
            test_id = test[0]
            test_data = test[1]; #print('test_data', test_data)
            test_new_data = test[2]; #print('test_new_data', test_new_data)
            test_switches = test[3]; #print('test_switvhed', test_switches)
            test_description = test[4]
            use_model = use_model_identifier(test_switches); #print('use_model', use_model)
            save_model = save_model_identifier(test_switches); #print('save_model', save_model)
            test_type = mode_identifier(test_switches); #print('test_type', test_type)

            if DEBUG:
                print('test_data', test_data)
                print('test_new_data', test_new_data)
                print('test_switvhes', test_switches)
                print('test_description', test_description)
                print('use_model', use_model)
                print('save_model', save_model)
                print('test_type', test_type)
                
            use_config_file = conf_identifier(test_switches); #print('use_config_file', use_config_file)
            #print('config file', get_conf_path(get_conf_name(test_switches), models_path))
            if use_config_file:
                use_model = use_model_in_config(get_conf_path(get_conf_name(test_switches), models_path));
                #print('use_model updated', use_model)
 
            test_errors = []
            model = False  # flag if test uses model
            status = True  # test run status
            execute_test = True  # flag if test should be executed
            if DEBUG:
                print('use_config_file', use_config_file)
                if use_config_file:
                    print('use_model updated', use_model)
                    print("DEBUG 1")
            if test_type == 'no mode':
                if test_switches not in {'-h', '--help'}:
                    test_type = 'unknown'
                else:
                    test_type = 'help'
            if test_type == 'unknown' and not (use_config_file):
                execute_test = False
                test_errors.append(['Build', 'Unknown mode or was not specified'])
                
            if DEBUG:
                print("DEBUG 2");
                print('execute_test', execute_test)
                
            if execute_test:
                new_prefix = 'Test' + test_id
                #print('test_data', test_data); print('use_model', use_model ); print('use_config_file', use_config_file)
                if (test_data == '' and not use_model and not use_config_file and not '-doe_spec' in test_switches):
                    if test_type != 'help':
                        execute_test = False
                        test_errors.append(['Build', 'No test data specified'])
                    else:
                        if test_switches == '-h':
                            test_data = 'h'
                        else:
                            test_data = 'help'
                        test_out = path.join(output_path, new_prefix + '_' + test_data + '.txt')
                        test_type = 'help'
                elif use_model:
                    # model_name = path.join(data_path, test_data).replace('\\', '/')
                    model_name = path.join(models_path, test_data).replace('\\', '/'); #print('model_name', model_name)
                    test_data_path = '-model_name \"{0}\"'.format(model_name)  # here we use a model instead of data
                    if DEBUG:
                        print('model_name', model_name)
                else:
                    if test_data != "":
                        if test_type == 'doe':
                            test_data_path = path.join(doe_path, test_data).replace('\\', '/')
                            #print('test_data_path', test_data_path); print('test_data', test_data)
                            if path.exists(test_data_path + '.csv'):
                                test_data_path = '-doe_spec \"{0}.csv\"'.format(test_data_path)
                            else:
                                execute_test = False
                                test_errors.append(['Build', 'DOE file does not exist'])
                        else:
                            test_data_path = path.join(data_path, test_data).replace('\\', '/')
                            #print('test_data_path', test_data_path); print('test_data', test_data)
                            if path.exists(test_data_path):
                                test_data_path = '-data \"{0}\"'.format(test_data_path)
                            elif path.exists(test_data_path + '.csv'):
                                test_data_path = '-data \"{0}.csv\"'.format(test_data_path)
                            else:
                                execute_test = False
                                test_errors.append(['Build', 'Data file does not exist'])
                    else:
                        test_data_path = ""
                if DEBUG:
                        print('test_data', test_data)
                        print('test_new_data', test_new_data)
                        print('test_data_path', test_data_path)
                        print('use_config_file', use_config_file )
                        print(test_new_data != "")
                        
                if test_type == 'prediction' or (test_new_data != ""): #use_config_file and
                    if not test_new_data == '':
                        test_new_data_path = path.join(data_path, test_new_data).replace('\\', '/')
                        if path.exists(test_new_data_path):
                            test_new_data_path = '-new_dat \"{0}\"'.format(test_new_data_path)
                        elif path.exists(test_new_data_path + '.csv'):
                            test_new_data_path = '-new_dat \"{0}.csv\"'.format(test_new_data_path)
                        else:
                            execute_test = False
                            test_errors.append(['Build', 'New data file does not exist'])
                    else:
                        execute_test = False
                        test_errors.append(['Build', 'No new data file specified'])
                if len(relevant_modes) > 0 and (test_type not in relevant_modes+['no mode']):
                     execute_test = False
                    
            if DEBUG:
                print("DEBUG 3")
                print('execute_test', execute_test)
                print('test_errors', test_errors)
                
            if execute_test:
                if use_config_file:
                    test_switches = get_switches_with_conf(test_switches, models_path)

                if RELEASE:
                    command = "../../src/run_smlp.py"
                else:
                    command = "../../src/run_smlp.py"
                if DEBUG:
                    print('command (0)', command); print('test_type', test_type)
                if test_type == 'help':
                    command += ' {args} > {output}'.format(args=test_switches, output=test_out)
                else:
                    if test_type in ['optimize', 'verify', 'query', 'optsyn', 'certify', 'synthesize']:
                        # add relative path to spec file name
                        spec_fn = spec_identifier(test_switches)# + '.spec';
                        #print('spec_fn', spec_fn); print('specs_path', specs_path)
                        if spec_fn is not None:
                            spec_file = os.path.join(specs_path, spec_fn)
                            test_switches = test_switches.replace(spec_fn, spec_file) ; #print('test_switches', test_switches)
                        else:
                            raise Exception('spec file must be specified in command line in model exploration modes')
                        # add relative path to external solver name
                        solver_bin = solver_path_identifier(test_switches); #print('solver_bin', solver_bin)
                        if solver_bin is not None:
                            solver_path_bin = os.path.join(solvers_path, solver_bin)
                            test_switches = test_switches.replace(solver_bin, solver_path_bin) ; #print('test_switches', test_switches)
                    #print('test_switches', test_switches); print('test_type', test_type)
                    command += ' {dat} {out_dir} {pref} {args} {debug} '.format(dat=test_data_path,
                                                                                               out_dir='-out_dir {output_path}'.format(
                                                                                                   output_path=output_path),
                                                                                               pref='-pref {prefix}'.format(
                                                                                                   prefix=new_prefix),
                                                                                               args=test_switches,
                                                                                               debug=debug)
                    if DEBUG:
                        print('command (1)', command);
                    #print('test_type', test_type, 'test_new_data',test_new_data) 
                    if test_type == 'prediction'  or (test_new_data != ""): #use_config_file and
                        command += '{new_dat} '.format(new_dat=test_new_data_path)
                        #print('command (2)', command);

                if DEBUG:
                    print('command (2)', command);
                    
                with print_l:
                    print("Running test {0} test type: {1}, description: {2}".format(test_id, test_type,
                                                                                     test_description))
                    print(command + '\n')
                if not args.print_command:
                    if save_model:
                        model = True
                    if args.timeout:
                        pr = popen_timeout(command, int(args.timeout))
                        if pr:
                            outs, errs = pr
                            if args.debug:
                                print('Output: \n' + outs + '\n' + 'Errors: \n' + errs + '\n')
                            if extract_smlp_error(errs) != 'OK':
                                status = False
                                test_errors.append(['Run', errs])
                        else:
                            status = False
                            test_errors.append(['Run', 'Timeout'])
                            execute_test = False
                    else:
                        pr = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                        outs, errs = pr.communicate()
                        if args.debug:
                            print('Output: \n' + outs + '\n' + 'Errors: \n' + errs + '\n')
                        if extract_smlp_error(errs) != 'OK':
                            status = False
                            test_errors.append(['Run', errs])
            if model:
                model = get_model_name(test_switches)
            id_q.put([test_id, execute_test, model, status, test_errors])

    if DEBUG:
        print("DEBUG 4")
    
    process_list = []
    expected_outs = tests_queue.qsize()
    if args.workers:
        workers = int(args.workers)
    else:
        workers = 2  # Number of concurrent processes
    if tests_queue.qsize() < workers:
        workers = tests_queue.qsize()
    print("Calling {workers} workers for multiprocessing...".format(workers=workers))
    for i in range(0, workers):
        t = Process(target=worker, args=(tests_queue, test_out_queue, print_lock))
        process_list.append(t)
        t.start()
        print("Initiating {i} worker...".format(i=i))
    counter = 0
    while counter < expected_outs:
        test_id_list.append(test_out_queue.get())
        counter += 1
        
    if DEBUG:
        print("DEBUG 5")
        
    for process in process_list:
        process.join()
    # fixing output and master path for the system use:
    master_path = master_path.replace('/', path.sep).replace('\"', '').replace('\'', '')
    output_path = output_path.replace('/', path.sep).replace('\"', '').replace('\'', '')
    files_in_master = get_all_files_from_dir(master_path)
    files_in_output = get_all_files_from_dir(output_path)

    if DEBUG:
        print("DEBUG 6")
        
    def get_file_from_list_underscore(prefix_list, list1):
        outs_list = []
        for file1 in list1:
            for pref in prefix_list:
                if file1.startswith(pref):
                    outs_list.append(file1)
        return outs_list

    cross_tests = []
    new_error_ids = [] # test IDs of crashes reporrted in error files -- ones that should not happen / do not occur in masters
    new_error_fns = [] # filenames of crashes reporrted in error files -- ones that should not happen / do not occur in masters
    #missing_errors = []  # crashes that are expected / are part of masters but do not occur in current run -- not implemented yet
    # main log comparing

    def smlp_txt_file(fname):
        if 'config' in fname or 'error' in fname or 'mrmr_features_summary' in fname or '_formula' in fname:
            return False
        elif fname.endswith('.txt'):
            return True
        else:
            return False

    def comapre_files(file1, file2):
        if file1.endswith('.csv'):
            return csv_cmp.compare_csv(file1, file2)
        f1 = open(file1, 'r')
        f2 = open(file2, 'r')
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        len1 = len(lines1)
        len2 = len(lines2)
        if len1 != len2:
            return False
        for x in range(0, len1):
            if lines1[x].startswith('<environment:'):
                continue
            if lines1[x] != lines2[x]:
                return False
        return True

    log = tests in {'all', 'real', 'toy', 'test'}  # to tell if there is a main log compare needed
    master_log_file = path.join(master_path,
                                tests + '_log.txt')
    log_file = path.join(output_path, tests + '_log.txt')

    if DEBUG:
        print("DEBUG 7")

    def write_to_log(line):
        with open(log_file, 'a') as writefile:
            writefile.write(line + '\n')

    if log:
        if path.exists(log_file):
            remove(log_file)
        with open(log_file, 'w') as a:
            a.write('~~~~~ Regression log file ~~~~~\n')

    def get_id(l):
        return int(l[0])

    if DEBUG:
        print("DEBUG 8")
        print('args.print_command',  args.print_command)
        print('args.diff',  args.diff)
        print('args.debug', args.debug)
    
    # sort test list
    # new_list = []
    # new_dict = dict()
    test_id_list.sort(key=get_id)
    if not (args.print_command or args.diff or args.debug):
        for i in test_id_list:
            test_id = i[0]
            execute_test = i[1]
            test_model = i[2]
            test_errors = i[4]
            test_prefix = 'Test' + test_id + '_'
            if execute_test:
                output_prefixes = [test_prefix]
                if test_model:
                    output_prefixes.append(test_model)
                new_files = get_file_from_list_underscore(output_prefixes, files_in_output)
                master_files = get_file_from_list_underscore(output_prefixes, files_in_master)
                '''
                for d in range(0, len(double_tests) - 1):
                    double = double_tests[d]
                    t1 = double[0]
                    t2 = double[1]
                    if int(test_id) in [t1, t2]:
                        cross_tests.append(double_tests.pop(d))
                '''
                test_result = True
                test_files_check = []
                txt_index = -1
                # print(new_files)
                # while not smlp_txt_file(new_files[txt_index]):
                #     txt_index += 1
                for k in range(0, len(new_files)):
                    if smlp_txt_file(new_files[k]):
                        txt_index = k  # found the txt file
                if txt_index != -1:
                    new_files_tmp = new_files[:]
                    new_files = [new_files_tmp.pop(txt_index)]
                    new_files_tmp.sort()
                    new_files = new_files + new_files_tmp
                to_show = True
                answer = None
                for file in new_files:
                    new_file = path.join(output_path, file)
                    master_file = path.join(master_path, file)
                    if os.path.isdir(new_file):
                        if os.path.exists(master_file):
                            assert os.path.isdir(master_file)
                        if new_file.endswith('_plots'):
                            if os.path.exists(master_file):
                                assert master_file.endswith('_plots')
                                file_to_minitor =  'plotReport.html'
                                new_file = os.path.join(new_file,)
                                master_file = os.path.join(master_file, file_to_minitor)
                                #print('dropping from master_files',  file)
                                master_files.remove(file)
                                file =  os.path.join(file, file_to_minitor)
                                #print('appending to master files', file)
                                master_files.append(file)
                                #print('update new_file', new_file); print('updated master file', master_file);
                                             
                    file_name = file
                    config_file = 'config' in file_name
                    # model_file = 'model' in file_name  # if its a model file it needs to be replaced in data as well
                    model_file = file_name.startswith('test' + str(test_id) + '_model')
                    txt_file = False
                    if path.exists(master_file):
                        if new_file.endswith('.txt') and not config_file :
                            txt_file = True
                        # condition before, dropping from it h5 file checks because getting UnicodeDecodeError error on Sles 15, say on Test 13.
                        # (new_file.endswith('.csv') or new_file.endswith('.txt') or  new_file.endswith('.html') or new_file.endswith('.json') or new_file.endswith('.h5')) and not file_name in files_to_ignore_from_diff:
                        if (new_file.endswith('.csv') or new_file.endswith('.txt') or  new_file.endswith('.html') or new_file.endswith('.json')) and not (file_name in files_to_ignore_from_diff): # or file_name.endswith('_model_term.json')
                            print('comparing {file} to master'.format(file=file_name))
                            p = Popen(
                                '{diff} -B -I \'Feature selection.*file .*\' -I \'\\[-v-] Input.*\' -I \'usage:.*\' {k} {l}'.format(
                                    diff=diff,
                                    k=new_file,
                                    l=master_file),
                                shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                            output, error = p.communicate()
                            if p.returncode == 1:
                                if not comapre_files(new_file, master_file):
                                    if not args.no_graphical_compare and to_show:
                                        Popen('{diff} {l} {k}'.format(diff=DIFF, k=new_file, l=master_file), shell=True).wait()
                                    if args.default or (args.config_default and config_file):
                                        if args.config_default and config_file:
                                            user_input = args.config_default
                                        else:
                                            user_input = args.default
                                    elif not to_show:
                                        print('answer is: ' + answer)
                                        user_input = answer
                                    else:
                                        user_input = input(
                                            'Do you wish to switch the new file with the master?\n(yes/no|y/n): ').lower()
                                    while user_input not in {'yes', 'no', 'y', 'n'}:
                                        user_input = input('(yes/no|y/n):').lower()
                                    if user_input in {'yes', 'y'}:
                                        if model_file or config_file:
                                            copyfile(new_file, master_file)
                                            copyfile(new_file, path.join(models_path, file_name))
                                            print('Replacing Files both in master and data')

                                        else:
                                            copyfile(new_file, master_file)
                                            print('Replacing Files...')
                                            if path.exists(path.join(data_path, file_name)):
                                                if args.default:
                                                    user_input = args.default
                                                else:
                                                    user_input = input(
                                                        'File exists also in data, switch there as well?\n(yes/no|y/n): ').lower()
                                                while user_input not in {'yes', 'no', 'y', 'n'}:
                                                    user_input = input('(yes/no|y/n):').lower()
                                                if user_input in {'yes', 'y'}:
                                                    copyfile(new_file, path.join(data_path, file_name))
                                    test_result = False
                                    test_files_check.append((file_name, 'Failed -> content diff'))
                                    if txt_file and args.fail_txt:
                                        to_show = False
                                        answer = user_input
                                else:
                                    print("Passed!")
                                    test_files_check.append((file_name, 'Passed'))
                            else:
                                print("Passed!")
                                test_files_check.append((file_name, 'Passed'))
                        if model_file:
                            master_files.remove(file_name)
                            remove(new_file)
                            if file in master_files:
                                master_files.remove(file)
                        else:
                            if os.path.isfile(file):
                                master_files.remove(file)
                    else:
                        # not comparing directories; such as the range plots directory in mode subgroups 
                        if os.path.isdir(new_file):
                            continue
                        print('File master {file} does not exist'.format(file=file))
                        test_files_check.append((file, 'Failed -> master file does not exist'))
                        if file.endswith("smlp_error.txt"):
                            to_print = 'Test number ' + test_id + ' Crashed!'
                            print(to_print)
                            new_error_ids.append(test_id)
                            new_error_fns.append(file)
                        elif file.endswith("png"):
                            continue
                        else:
                            if not args.default:
                                user_input = input(
                                    'What to do with the new file?\n1 - Nothing\n2 - Copy to master only\n3 - Copy to master and models\n4 - Remove from master only\n5 - Remove from master and models\nOption number: ')
                                while user_input not in {'1', '2', '3', '4', '5'}:
                                    user_input = input('(1|2|3|4|5):')
                                if user_input == '1':
                                    pass
                                elif user_input == '2':
                                    if os.path.isdir(new_file):
                                        copytree(new_file, master_file, dirs_exist_ok=True)
                                    else:
                                        copyfile(new_file, master_file)
                                elif user_input == '3':
                                    copyfile(new_file, master_file)
                                    copyfile(new_file, path.join(models_path, file_name))
                                elif user_input == '4':
                                    os.remove(master_file)
                                elif user_input == '5':
                                    os.remove(master_file)
                                    os.remove(path.join(models_path, file_name))

                        """
                        diff_errors.append('File master {file} does not exist'.format(file=file))
                        user_input = input('Do you wish to copy the new file to master?\n(yes/no|y/n): ').lower()
                        while user_input not in {'yes', 'no', 'y', 'n'}:
                            user_input = input('(yes/no|y/n):').lower()
                        if user_input in {'yes', 'y'}:
                            copyfile(new_file, master_file)
                            print('Copying file...')
                        """
                for file in master_files:
                    new_file = path.join(output_path, file); #print('new_file', new_file)
                    master_file = path.join(master_path, file); #print(' master_file',  master_file)
                    file_name = file
                    print('File new {file} does not exist'.format(file=file))
                    test_files_check.append((file, 'Failed -> new file does not exist'))
                    test_result = False
                    #  diff_errors.append('File new {file} does not exist'.format(file=file))
                    if not args.default:
                         user_input = input(
                             'What to do with the master file?\n1 - Nothing\n2 - Remove from master only\n3 - Remove from master and models\nOption number: ')
                         while user_input not in {'1', '2', '3',}:
                             user_input = input('(1|2|3):')
                         if user_input == '1':
                            pass
                         elif user_input == '2':
                            os.remove(master_file)
                         elif user_input == '3':
                            os.remove(master_file)
                            if os.path.exists(path.join(models_path, file_name)):
                                os.remove(path.join(models_path, file_name))
                if log:
                    if test_result:
                        write_to_log('Test ' + test_id + ' Passed:')
                    else:
                        write_to_log('Test ' + test_id + ' Failed:')
                    for file_check in test_files_check:
                        write_to_log(file_check[0] + ' ' + file_check[1])
                    write_to_log('')
            else:
                print('Test {id} Failed:'.format(id=test_id))
                if log:
                    write_to_log('Test {id} Failed:'.format(id=test_id))
                for test_error in test_errors:
                    print('Error in {stage} stage:'.format(stage=test_error[0]))
                    print(test_error[1])
                    if log:
                        write_to_log('Error in {stage} stage:'.format(stage=test_error[0]))
                        write_to_log(test_error[1])

    if DEBUG:
        print('9')
        print('log and not args.diff', log and not args.diff)
        
    if log and not args.diff:
        if path.exists(master_log_file):
            print('Comparing regression logs:')
            if not path.exists(master_log_file):
                copyfile(log_file, master_log_file)
            p = Popen('diff {master_log} {new_log}'.format(new_log=log_file, master_log=master_log_file), shell=True,
                      stdin=PIPE,
                      stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            if p.returncode == 1:
                Popen('{diff} {master_log} {new_log}'.format(diff=DIFF, new_log=log_file, master_log=master_log_file),
                      shell=True).wait()
                user_input = input(
                    'Do you wish to switch the new log file with the master log file?\n(yes/no|y/n): ').lower()
                while user_input not in {'yes', 'no', 'y', 'n'}:
                    user_input = input('(yes/no|y/n):').lower()
                if user_input in {'yes', 'y'}:
                    copyfile(log_file, master_log_file)
                    print('Replacing Files...')
            else:
                print("Passed!")
        else:
            print("master log file does not exist!")
            user_input = input(
                'Do you wish to copy the new log file to master?\n(yes/no|y/n): ').lower()
            while user_input not in {'yes', 'no', 'y', 'n'}:
                user_input = input('(yes/no|y/n):').lower()
            if user_input in {'yes', 'y'}:
                copyfile(log_file, master_log_file)
                print('Replacing Files...')

    """
    for testid, errors in test_errors_dict.items():
        if len(errors) >= 1:
            print('Test {0} had the following errors: {1}\n'.format(testid, errors))

    if len(diff_errors) > 0:
        print("Diff errors:\n")
    for err in diff_errors:
        print(err + '\n')
    """
    if False: # not args.print_command:
        chdir(code_path)
        try:
            rmtree(temp_code_dir)
        except:
            print("Can't delete " + temp_code_dir + " dir.")
    
    # report tests that crashed -- based on TestXXX_error.txt files that do not exist in master
    if len(new_error_fns) > 0:
        print('Tests crashed (not in the masters):')
        for efn in new_error_fns:      
            print(efn)
    else:
        print('No new tests crashed (not in the masters)')
            
    print("Time: " + str((time() - start_time)/60.0) + " minutes")
    print('End of regression')


if __name__ == "__main__":
    main()
