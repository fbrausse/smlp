
import argparse
import json
import os
import pytest
import shlex
import subprocess
import sys
import warnings

import csv_comparator as csv_cmp

from pathlib import Path

files_to_ignore_from_diff = ['Test41_doe_two_levels_doe.csv', 'Test42_doe_two_levels_doe.csv']

def _cwd_rel(path):
	return Path(path).relative_to(Path('.').absolute(), walk_up=True)

def _filter_out(lst, item, n):
	i = 0
	while i < len(lst):
		it = lst[i]
		if it == item:
			i += n
		else:
			yield it
		i += 1

def _get_arg(lst, item):
	i = 0
	while i < len(lst):
		if lst[i] == item:
			return lst[i+1]
		i += 1

# --------------- adapted from regr_smlp.py ---------------

def extract_smlp_error(error_string):
	error_list = error_string.splitlines()
	for line in error_list:
		if line.startswith('Error:'):
			error_msg = line[7:]
			return [error_msg]
	return []

def get_file_from_list_underscore(prefix_list, list1):
	outs_list = []
	for file1 in list1:
		for pref in prefix_list:
			if file1.startswith(pref):
				outs_list.append(file1)
	return outs_list

def smlp_txt_file(fname):
	if 'config' in fname or 'error' in fname or 'mrmr_features_summary' in fname or '_formula' in fname:
		return False
	elif fname.endswith('.txt'):
		return True
	else:
		return False

def get_all_files_from_dir(dir_path):
	return os.listdir(dir_path)

def comapre_files(file1, file2):
	if file1.suffix == '.csv':
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

# --------------- end adapted from regr_smlp.py ---------------

class CmdTestCase:
	def _construct_args(self, regrdir, extdir, tmpdir):
		pre = []

		if self.data:
			pre += ['-data', self.data + '.csv']

		pre += ['-out_dir', str(tmpdir)]
		pre += ['-pref', f'Test{self.nr}']
		pre += self.args

		if self.new_data:
			pre += ['-new_dat', self.new_data + '.csv']

		# resolve paths in special locations
		special = {
			'-config'     : regrdir/'models',
			'-data'       : regrdir/'data',
			'-new_dat'    : regrdir/'data',
			'-solver_path': extdir,
			'-spec'       : regrdir/'specs',
		}
		i = 0
		args = []
		while i < len(pre):
			arg = pre[i]
			loc = special.get(arg)
			if loc:
				path = Path(pre[i+1])
				if path.is_absolute():
					warnings.warn(
						f'path given to {arg} in args is absolute: {path}')
				else:
					path = _cwd_rel(loc/path)
				args += [arg, str(path)]
				i += 2
			else:
				args.append(arg)
				i += 1

		args = self._apply_morespecial_logic(args, regrdir)

		for o in ('-data', '-new_dat'):
			datapath = _get_arg(args, o)
			if datapath is not None and not Path(datapath).exists():
				warnings.warn(f'path for option {o} does not exist: {datapath}')
				pytest.skip(f'path for option {o} does not exist: {datapath}')
				return None

		return args

	def _apply_morespecial_logic(self, pre, regrdir):
		use_model = False
		for o in ('-use_model', '--use_model'):
			if o in pre:
				use_model |= pre[pre.index(o) + 1].lower().startswith('t')
		try:
			cfgpath = pre[pre.index('-config') + 1]
			with open(cfgpath, 'r') as f:
				# TODO: get rid of double quotes in config file to make this a Boolean
				use_model |= json.load(f)['use_model'] == 'true'
		except ValueError:
			pass

		if use_model:
			assert self.data
			# TODO: cannot use a relative path, it would be appended to the
			# -out_dir path, see
			# <https://github.com/SMLP-Systems/smlp/pull/61#issuecomment-4090755246>
			args = ['-model_name', str(regrdir/'models'/self.data)]
			args += list(_filter_out(pre, '-data', 1))
			return args

		try:
			is_doe = pre[pre.index('-mode') + 1] == 'doe'
		except ValueError:
			is_doe = False

		if is_doe:
			assert self.data
			path = regrdir/'grids'/(self.data + '.csv')
			if not path.exists():
				warnings.warn(f'-doe_spec file {path} does not exist')
				pytest.skip(f'-doe_spec file {path} does not exist')
				return None
			args = ['-doe_spec', str(_cwd_rel(path))]
			args += list(_filter_out(pre, '-data', 1))
			return args

		return pre

	def more_subproc_run_args(self) -> dict:
		return {}

	def test(self, tmp_path, pytestconfig):
		assert self.nr > 0
		projdir = pytestconfig.rootdir
		regrdir = projdir/'regr_smlp'
		extdir  = projdir/'..'/'external'
		exepath = projdir/'src'/'run_smlp.py'
		args    = self._construct_args(regrdir, extdir, tmp_path)

		print(f'Test {self.nr} using tmp-path: {tmp_path}', file=sys.stderr)

		# write the meta-data for this test so results can be checked
		# independently of the execution
		meta = tmp_path/'.meta'
		meta.mkdir()
		with open(meta/'env', 'w') as f:
			for k, v in os.environ.items():
				print(f'{k}={v}', file=f)
		with open(meta/'test-id', 'w') as f:
			print(self.nr, file=f)
		with open(meta/'cwd', 'w') as f:
			print(Path().absolute(), file=f)
		with open(meta/'project-dir', 'w') as f:
			print(projdir, file=f)

		if args is None:
			print('', file=sys.stderr)
		else:
			cmd = [str(_cwd_rel(exepath))] + args
			with open(meta/'cmd', 'w') as f:
				print(shlex.join(cmd), file=f)

			print(shlex.join(cmd), file=sys.stderr)

			with (open(meta/'stdout', 'wb', buffering=0) as o,
			      open(meta/'stderr', 'wb', buffering=0) as e):
				pr = subprocess.run(cmd,
				                    stdin=subprocess.DEVNULL,
				                    stdout=o, stderr=e, check=True,
				                    **self.more_subproc_run_args())

			#check_outputs(str(self.nr), args, pr.stdout, pr.stderr, regrdir, tmp_path)
			assert check_outputs(tmp_path), (
				'checking against master failed, '
				f'for details see summary in {meta/"test_log.txt"}'
			)

def check_outputs(tmpdir):
	meta = tmpdir/'.meta'
	try:
		with open(meta/'cmd') as f:
			cmd = shlex.split(f.read()[:-1])
		args = cmd[1:]
	except FileNotFoundError:
		args = None
	with open(meta/'cwd') as f:
		os.chdir(f.read()[:-1])
	with open(meta/'test-id') as f:
		test_id = f.read()[:-1]
	with open(meta/'project-dir') as f:
		projdir = Path(f.read()[:-1])
	with open(meta/'stdout') as f:
		stdout = f.read()
	with open(meta/'stderr') as f:
		stderr = f.read()

	return _check_outputs(test_id, args, stdout, stderr, projdir/'regr_smlp', tmpdir)

def _check_outputs(test_id, smlp_args, stdout, stderr, regrdir, output_path):
	# adapted from smlp_regr.py
	diff = 'diff'

	args = argparse.Namespace()
	args.no_graphical_compare = True
	args.default = 'n'
	args.config_default = 'n'
	args.fail_txt = False

	#log = tests in {'all', 'real', 'toy', 'test'}  # to tell if there is a main log compare needed
	log = True
	log_file = output_path/'.meta'/'test_log.txt'
	def write_to_log(line):
		with open(log_file, 'a') as writefile:
			writefile.write(line + '\n')

	execute_test = True

	test_model = False
	for o in ('-save_model', '--save_model'):
		o = _get_arg(smlp_args, o)
		if o and o.lower().startswith('t'):
			test_model = True
			break
	if test_model:
		test_model = _get_arg(smlp_args, '-model_name')
		if not test_model:
			warnings.warn(f'-save_model {o} specified in args but no -model_name given')

	test_errors = extract_smlp_error(stderr)
	test_prefix = 'Test' + test_id + '_'

	data_path   = regrdir/'data'
	models_path = regrdir/'models'
	master_path = regrdir/'master'
	files_in_master = get_all_files_from_dir(str(master_path))
	files_in_output = get_all_files_from_dir(str(output_path))

	if execute_test:
		output_prefixes = [test_prefix]
		if test_model:
			output_prefixes.append(test_model)
		new_files = get_file_from_list_underscore(output_prefixes, files_in_output)
		master_files = get_file_from_list_underscore(output_prefixes, files_in_master)
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
			new_file = str(output_path/file)
			master_file = str(master_path/file)
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
			if os.path.exists(master_file):
				if Path(new_file).suffix == '.txt' and not config_file:
					txt_file = True
				# condition before, dropping from it h5 file checks because getting UnicodeDecodeError error on Sles 15, say on Test 13.
				# (new_file.endswith('.csv') or new_file.endswith('.txt') or  new_file.endswith('.html') or new_file.endswith('.json') or new_file.endswith('.h5')) and not file_name in files_to_ignore_from_diff:
				exclude_cond = file_name in files_to_ignore_from_diff
				exclude_cond = file_name in files_to_ignore_from_diff or file_name.endswith('_model_term.json')
				if (Path(new_file).suffix in ('.csv', '.txt', '.html', '.json')) and not exclude_cond:
					print('comparing {file} to master'.format(file=file_name))
					# XXX fb: Hack using sed to replace output_path in new log
					#         file.
					p = subprocess.Popen(
						f'sed \'s,{output_path},.,g\' {new_file} | {diff} -B '
						'-I \'Feature selection.*file .*\' '
						'-I \'\\[-v-] Input.*\' '
						'-I \'usage:.*\' '
						'-I \'Seving model rerun configuration in file'
						f'- {master_file}',
						shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
									copyfile(new_file, models_path/file_name)
									print('Replacing Files both in master and data')

								else:
									copyfile(new_file, master_file)
									print('Replacing Files...')
									if path.exists(data_path/file_name):
										if args.default:
											user_input = args.default
										else:
											user_input = input(
												'File exists also in data, switch there as well?\n(yes/no|y/n): ').lower()
										while user_input not in {'yes', 'no', 'y', 'n'}:
											user_input = input('(yes/no|y/n):').lower()
										if user_input in {'yes', 'y'}:
											copyfile(new_file, data_path/file_name)
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
					os.remove(new_file)
					if file in master_files:
						master_files.remove(file)
				else:
					if os.path.isfile(new_file):
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
							copyfile(new_file, models_path/file_name)
						elif user_input == '4':
							os.remove(master_file)
						elif user_input == '5':
							os.remove(master_file)
							os.remove(models_path/file_name)

		for file in master_files:
			new_file = str(output_path/file); #print('new_file', new_file)
			master_file = str(master_path/file); #print(' master_file',  master_file)
			file_name = file
			print(f'File new {file} does not exist')
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
					if os.path.exists(models_path/file_name):
						os.remove(models_path/file_name)
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

	return test_result

def main():
	return 0 if check_outputs(Path(sys.argv[1])) else 1

if __name__ == '__main__':
	sys.exit(main())
