
import json
import sys
import shlex
import warnings

from pathlib import Path

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
						f'Test #{self.nr}: '
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
			args = ['-model_name', str(_cwd_rel(regrdir/'models'/self.data))]
			args += list(_filter_out(pre, '-data', 1))
			return args

		try:
			is_doe = pre[pre.index('-mode') + 1] == 'doe'
		except ValueError:
			is_doe = False

		if is_doe:
			assert self.data
			args = ['-doe_spec', str(_cwd_rel(regrdir/'grids'/(self.data + '.csv')))]
			args += list(_filter_out(pre, '-data', 1))
			return args

		return pre

	def test(self, tmp_path, pytestconfig):
		assert self.nr > 0
		projdir = pytestconfig.rootdir
		regrdir = projdir/'regr_smlp'
		extdir  = projdir/'..'/'external'
		exepath = projdir/'src'/'run_smlp.py'
		args    = self._construct_args(regrdir, extdir, tmp_path)
		if args is None:
			print('', file=sys.stderr)
		else:
			cmd = [str(_cwd_rel(exepath))] + args
			print(shlex.join(cmd), file=sys.stderr)
		assert False
