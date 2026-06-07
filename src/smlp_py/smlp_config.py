# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import os, json
import argparse
import textwrap

from .smlp_utils import str_to_bool


class _ConciseHelpFormatter(argparse.HelpFormatter):
    """
    Hybrid formatter _ConciseHelpFormatter:
      - Help strings containing explicit newline characters (`\n`) are preserved verbatim
      - Help strings without explicit newlines are wrapped normally (HelpFormatter behavior)
      - Option invocations are concise (-s, --long ARG)
        E.g., instead of
          -f FILE, --file FILE    Some text describing this option.
        this class will print
          -f, --file FILE         Some text describing this option.
    """

    def _split_lines(self, text, width):
        # If the author provided explicit newlines, preserve them
        if '\n' in text:
            text = textwrap.dedent(text).strip('\n')
            return text.splitlines()

        # Otherwise, let argparse wrap normally
        return super()._split_lines(text, width)

    def _fill_text(self, text, width, indent):
        return '\n\n'.join(
            super()._fill_text(par, width, indent)
            for par in text.split('\n\n')
        )

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts = action.option_strings
            # if the Optional takes a value, format is:
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                parts = list(action.option_strings)
                parts[-1] += ' ' + args_string
            return ', '.join(parts)

class SmlpConfig:
    def __init__(self):
        self.report_file_prefix = None
        self.model_file_prefix = None
        self.model_rerun_config = None
        self.config = None
        
        self._DEF_LABELED_DATA = None
        self._DEF_ANALYTICS_MODE = None #'train'
        self._DEF_SAVE_CONFIGURATION = False
        self._DEF_LOG_FILE_PREFIX = None
        self._DEF_OUTPUT_DIRECTORY = None
        self._DEF_INTERACTIVE_PLOTS = True
        self._DEF_SEED = None
        self._DEF_LOAD_CONFIGURATION = None
        
        self.modes_data_dict = {
            'analytics_mode': {'abbr':'mode', 'default':self._DEF_ANALYTICS_MODE, 'type':str,
                'help': '''\
                      What kind of analysis should be performed. Supported modes [default: {}]:
                      train         Train a model specified using option -model
                      predict       Load or train a model and run prediction
                      optimize      Run optimization over model or system outputs
                      verify        Verify properties of trained models or systems
                      query         Answer logical or numerical queries over models
                      optsyn        Perform optimization that satisfies assertions
                      subgroups     Discover data subgroups as feature-range tuples
                      doe           Perform design-of-experiments analysis
                      discretize    Discretize continuous features or responses
                '''.format(str(self._DEF_ANALYTICS_MODE))},
            'labeled_data': {'abbr':'data', 'default':self._DEF_LABELED_DATA, 'type':str, 
                'help':'Path, possibly excluding the .csv, or including gz or bz2 suffix, to input ' +
                    ' training data file containing labels [default {}]'.format(str(self._DEF_LABELED_DATA))}
        }

        self.config_params_dict = {
            'interactive_plots': {'abbr':'plots', 'default':self._DEF_INTERACTIVE_PLOTS, 'type':str_to_bool, 
                'help':'Should plots be displayed interactively (or only saved)?'+
                    '[default: {}]'.format(str(self._DEF_INTERACTIVE_PLOTS))},
            'seed': {'abbr':'seed', 'default':self._DEF_SEED, 'type':int, 
                'help':'Initial random seed [default {}]'.format(str(self._DEF_SEED))},
            'log_files_prefix': {'abbr':'pref', 'default':self._DEF_LOG_FILE_PREFIX, 'type':str, 
                'help':'String to be used as prefix for the output files ' + 
                    '[default: {}]'.format(str(self._DEF_LOG_FILE_PREFIX))},
            'output_directory': {'abbr':'out_dir', 'default':self._DEF_OUTPUT_DIRECTORY, 'type':str, 
                'help':'Output directory where all reports and output files will be written '+
                    '[default: the same directory from which data is loaded]'},
            'save_configuration': {'abbr':'save_config', 'default':self._DEF_SAVE_CONFIGURATION, 'type':str_to_bool, 
                'help':'Should tool run parameters be saved into a a configuration file? ' +
                    '[default: {}]'.format(str(self._DEF_SAVE_CONFIGURATION))},
            'load_configuration': {'abbr':'config', 'default':self._DEF_LOAD_CONFIGURATION, 'type':str, 
                'help':'Json config file name, to load tool parameter values from, or None. ' +
                    'Paramters specified through command line will override the correponding '
                    'config file values if they are specified there as well ' +
                    '[default: {}]'.format(str(self._DEF_LOAD_CONFIGURATION))}
        }
    
    
    # Compute two prefixes:
    #   - report_name_prefix: used in all report and log file names for an SMLP run
    #   - model_name_prefix: used for all files involved in saving a trained model
    #     or loading a previously saved model.
    #
    # Using the same model_name_prefix is safe because the options -use_model
    # (load an existing model) and -save_model (save a newly trained model)
    # cannot both be True in the same run. An assertion ensures that this
    # prohibited combination is caught if it occurs.
    def args_get_report_name_prefix(self, data_file_prefix:str, run_prefix:str, output_directory:str=None, 
            new_data_file_prefix:str=None, model_name:str=None, save_model:bool=None, use_model:bool=None, 
            doe_spec_file_prefix=None):
        
        if use_model:
            assert model_name is not None
            # Prefix (path + model name) of a previously trained and saved model, used when loading an 
            # existing model.
            load_model_name_prefix = model_name
        
        if not data_file_prefix is None:
            data_dir, data_name_prefix = os.path.split(data_file_prefix)
        else:
            data_dir, data_name_prefix = None, None

        # Compute model_dir and save_model_name_prefix. save_model_name_prefix is required when saving a 
        # trained model and will be updated below so that its directory path matches the output directory.
        if not model_name is None:
            model_dir, save_model_name_prefix = os.path.split(model_name)
        else:
            model_dir, save_model_name_prefix = None, None

        if not doe_spec_file_prefix is None:
            doe_spec_dir, doe_spec_name_prefix = os.path.split(doe_spec_file_prefix)
        else:
            doe_spec_dir, doe_spec_name_prefix = None, None
        
        out_dir = output_directory
        if out_dir is None:
            if not data_dir is None:
                out_dir = data_dir
            elif not model_dir is None:
                out_dir = model_dir
            elif not doe_spec_dir is None:
                out_dir = doe_spec_dir
            else:
                raise Exception('A training data file, a model or doe spec file should be provided')   
        
        if data_name_prefix is not None:
            input_data_name_prefix = data_name_prefix.removesuffix('.bz2')
            input_data_name_prefix = input_data_name_prefix.removesuffix('.gz')
            input_data_name_prefix = input_data_name_prefix.removesuffix('.csv')
        elif doe_spec_name_prefix is not None:
            input_data_name_prefix = doe_spec_name_prefix.removesuffix('.csv')
        else:
            assert model_name is not None
            input_data_name_prefix = None

        # Update save_model_name_prefix so it becomes the prefix for all files generated when saving a trained model.
        if model_name is None:
            assert not (data_name_prefix is None and doe_spec_dir is None)
            save_model_name_prefix = os.path.join(out_dir, run_prefix + '_' + input_data_name_prefix)
        else:
            save_model_name_prefix = os.path.join(out_dir, model_name)

        # define _report_name_prefix to be used as a prefix in SMLP report filenames
        if input_data_name_prefix is None:
            assert not model_name is None
            _, model_name = os.path.split(model_name)
            report_name_prefix = os.path.join(out_dir, run_prefix + '_' + model_name)
        else:
            report_name_prefix = os.path.join(out_dir, run_prefix + '_' + input_data_name_prefix)

        # if new_data is not None, its name is added to self._filename_prefix
        if not new_data_file_prefix is None:
            new_data_file_prefix = new_data_file_prefix.removesuffix('.bz2')
            new_data_file_prefix = new_data_file_prefix.removesuffix('.gz')
            new_data_file_prefix = new_data_file_prefix.removesuffix('.csv')
            _, new_data_fname = os.path.split(new_data_file_prefix)
            report_name_prefix = report_name_prefix + '_' + new_data_fname
            
        if use_model:
            model_name_prefix = load_model_name_prefix
        else:
            model_name_prefix = save_model_name_prefix
        
        return report_name_prefix, model_name_prefix

    # args parser to which some of the arguments are added explicitly in a regular way
    # and in addition it adds additional arguments from args_dict defined elsewhere;
    # As of now args_dict includes model training hyperparameters from ML packages
    # sklearm caret, keras -- model_params_dict = keras_dict | sklearn_dict | caret_dict, 
    # as well as data and logger related parameters: data_params_dict and logger_params_dict
    def args_dict_parse(self, argv, args_dict):
        parser = argparse.ArgumentParser(prog=argv[0], formatter_class=_ConciseHelpFormatter)
        
        for p, v in args_dict.items():
            if 'default' in v:
                parser.add_argument('-'+v['abbr'], '--'+p, default=v['default'], 
                                    type=v['type'], help=v['help'])
            else:
                parser.add_argument('-'+v['abbr'], '--'+p, type=v['type'], help=v['help'])

        args = parser.parse_args(argv[1:])

        # support for loading parameters from configuration file
        if args.load_configuration is not None:
            with open(args.load_configuration, 'r') as f:
                parser.set_defaults(**json.load(f))
            # Re-parse: CLI args will override config defaults
            args = parser.parse_args(argv[1:])

        # Args sanity check:
        assert not (args.use_model and args.save_model), \
            "Saving model should be disabled when a saved model is used"
        
        if args.model == 'system':
            args.prediction_plots = False
            args.response_plots = False
            args.interactive_plots = False
                
        # compute and save report_file_prefix and model_file_prefix as part of self
        self.report_file_prefix, self.model_file_prefix = self.args_get_report_name_prefix(args.labeled_data,
            args.log_files_prefix, args.output_directory, args.new_data, args.model_name, args.save_model,
            args.use_model, args.doe_spec_file)
        
        # Save tool configuration and model rerun configuration
        if args.save_configuration:
            args_config_file = self.report_file_prefix + '_args_config.json'
            tmp_args = vars(args).copy()
            del tmp_args['save_configuration'] # Do not dump value of this option
            del tmp_args['load_configuration'] # Values already loaded
            self.config = tmp_args
            with open(args_config_file, 'w') as f:
                f.write(json.dumps(tmp_args, indent=4, sort_keys=True))

        # Save model rerun config, to be able to build model with same parameters for new data
        if args.save_model_rerun_configuration:
            if not vars(args)['save_model']:
                return args
            model_args = vars(args).copy()
            # assign false to save_model since we are using an already saved model
            model_args['save_model'] = 'false'
            # assign true to use_model since we want to use a saved model
            model_args['use_model'] = 'true'
            # new data set must be provided, we are not using new data from config file
            model_args['new_data'] = None
            # training (labeled) data set from which model was built is not required
            model_args['labeled_data'] = None
            # the log file prefix used to create model is not needed
            model_args['log_files_prefix'] = None
            # prefix model_name with the output directory
            model_args['model_name'] = None
            self.model_rerun_config = model_args
        
        return args
    
