#import textwrap
import os, argparse, json
from smlp_py.smlp_utils import str_to_bool

class SmlpConfig:
    def __init__(self):
        self.report_file_prefix = None
        self.model_file_prefix = None
        self.model_rerun_config = None
        self.config = None
    
    # Compute prefix report_name_prefix to be used in all report / log file names of an SMLP run; 
    # as well as prefix model_name_prefix to be used in the names of all output files that are required 
    # to save a trained model info and re-run the saved model on new data (without re-training).
    def args_get_report_name_prefix(self, data_file_prefix:str, run_prefix:str, output_directory:str=None, 
            new_data_file_prefix:str=None, model_name:str=None, doe_spec_file_prefix=None):
        
        if not data_file_prefix is None:
            data_dir, data_name_prefix = os.path.split(data_file_prefix)
        else:
            data_dir, data_name_prefix = None, None

        # Define _model_dir and _model_name_prefix. 
        if not model_name is None:
            model_dir, model_name_prefix = os.path.split(model_name)
        else:
            model_dir, model_name_prefix = None, None

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
            input_data_name_prefix = data_name_prefix
        else:
            input_data_name_prefix = doe_spec_name_prefix
        
        # record model_name 
        #model_name = model_name

        # define model_name_prefix to be used as suffix in names of all files used to save the model related info
        #run_prefix = run_prefix
        if model_name is None:
            assert not (data_name_prefix is None and doe_spec_dir is None)
            model_name_prefix = os.path.join(out_dir, run_prefix + '_' + input_data_name_prefix)
        else:
            model_name_prefix = os.path.join(out_dir, model_name)

        # define _report_name_prefix to be used as a prefix in SMLP report filenames
        if input_data_name_prefix is None: #data_name_prefix
            assert not model_name is None
            _, model_name = os.path.split(model_name)
            report_name_prefix = os.path.join(out_dir, run_prefix + '_' + model_name)
        else:
            report_name_prefix = os.path.join(out_dir, run_prefix + '_' + input_data_name_prefix)

        # if new_data is not None, its name is added to self._filename_prefix
        new_data_file_prefix = new_data_file_prefix
        if not new_data_file_prefix is None:
            _, new_data_fname = os.path.split(new_data_file_prefix)
            report_name_prefix = report_name_prefix + '_' + new_data_fname
        return report_name_prefix, model_name_prefix

    # args parser to which some of the arguments are added explicitly in a regular way
    # and in addition it adds additional arguments from args_dict defined elsewhere;
    # As of now args_dict includes model training hyperparameters from ML packages
    # sklearm caret, keras -- model_params_dict = keras_dict | sklearn_dict | caret_dict, 
    # as well as data and logger related parameters: data_params_dict and logger_params_dict
    def args_dict_parse(self, argv, args_dict):
        #print('argv', argv)
        parser = argparse.ArgumentParser(prog=argv[0])
        #print('parser', parser)
        parser.add_argument('-data', '--labeled_data', metavar='DATA', type=str,
                        help='Path excluding the .csv suffix to input training data ' +
                            'file containing labels')
        #parser.add_argument('-model', '--model', metavar='MODEL', type=str,
        #                    help='Type of model to train (NN, Poly, ... ')
        parser.add_argument('-mode', '--analytics_mode', type=str, default=None,
                        help='What kind of analysis should be performed '+
                            'the supported modes are: "train", "predict", "subgroups",  ' +
                            '"doe", "discretize", "optimize", "verify", "query", "optsyn" ' +
                            '[default: train]')
        parser.add_argument('-plots', '--interactive_plots', type=str_to_bool, default=True,
                        help='Should plots be displayed interactively (or only saved)?'+
                            '[default: True]')
        parser.add_argument('-seed', '--seed', type=int, default=None,
                        help='Initial random seed')
        parser.add_argument('-pref', '--log_files_prefix', type=str, default=None,
                        help='String to be used as prefix for the output files '+
                            '[default: empty string]')
        parser.add_argument('-out_dir', '--output_directory', type=str, default=None,
                        help='Output directory where all reports and output files will be written '+
                            '[default: the same directory from which data is loaded]')
        parser.add_argument('-save_config', '--save_configuration', type=str_to_bool, default=False,
                        help='Should tool run parameters be saved into a a configuration file? ' +
                            '[default: True]')
        parser.add_argument('-config', '--load_configuration', type=str, default=None,
                        help='Json config file name, to load tool parameter values from, or None. ' +
                            'Paramters specified through command line will override the correponding '
                            'config file values if they are specified there as well [default: True]')
        
        for p, v in args_dict.items():
            #print('p', p, 'v', v); print('type', v['type'])
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

        # Reload arguments to override config file values with command line values
        args = parser.parse_args()

        # compute and save report_file_prefix and model_file_prefix as part of self
        self.report_file_prefix, self.model_file_prefix = self.args_get_report_name_prefix(args.labeled_data, 
            args.log_files_prefix, args.output_directory, args.new_data, args.model_name, args.doe_spec_file) 

        # Save tool configuration and model rerun configuration
        # Adapted code from https://micha-feigin.medium.com/on-using-config-files-with-pythons-argparse-8af09d0bdfb9
        # TODO !!! this is not the right place to save configuration. This is better to do 
        # within function args_dict_parse called above, but in current implementation we are forced
        # to save configuration only after inst (paths definitions) has been instantiated, as we 
        # need to compute file name for the dumped json file and for this we need function 
        # inst.get_report_name_prefix() from inst to be available
        #print('args.save_configuration', args.save_configuration)
        if args.save_configuration:
            #args_config_file = inst.get_report_name_prefix() + '_args_config.json'
            args_config_file = self.report_file_prefix + '_args_config.json'
            tmp_args = vars(args).copy()
            del tmp_args['save_configuration']  # Do not dump value of conf_export flag
            del tmp_args['load_configuration']  # Values already loaded
            self.config = tmp_args
            with open(args_config_file, 'w') as f:
                f.write(json.dumps(tmp_args,  indent=4, sort_keys=True))
                f.close()
                #json.dump(args, f, indent='\t', cls=np_JSONEncoder)
        
        # save configuration to be able to build model with same parameters for new data
        if args.save_model_rerun_configuration:
            #model_rerum_config_file = inst.get_report_name_prefix() + '_rerun_model_config.json'
            if not vars(args)['save_model']:
                return args
            #model_rerum_config_file = self.report_file_prefix + '_rerun_model_config.json'
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
        #print('args', args)
        return args
    