#!/usr/bin/env python3

import csv
import sys
import shlex

from smlp_regr import (conf_identifier, get_conf_name)

HEADER = '''\
#!/usr/bin/env python3

import pytest
import subprocess

from lib import *
'''

FOOTER = '''\
'''

def process(row : list[str]):
	assert len(row) >= 5, repr(row)
	nr = int(row[0])
	data = row[1]
	new_data = row[2]
	switches = row[3]
	comment = row[4:]

	args = shlex.split(switches)

	ignored_tests = []
	is_toy = ((row[1].startswith('smlp_toy') or
	           row[1].startswith('mlbt_toy') or
	           row[2].startswith('smlp_toy') or
	           row[2].startswith('mlbt_toy') or
	           (conf_identifier(row[3]) and get_conf_name(row[3]).startswith('smlp_toy')) or
	           (not conf_identifier(row[3]) and row[1] == '' and row[2] == '')) and
	           (row[0] not in ignored_tests))

	is_real = ((not (row[1].startswith('smlp_toy') or
	                 row[1].startswith('mlbt_toy') or
	                 row[2].startswith('smlp_toy') or
	                 row[2].startswith('mlbt_toy') or
	                 (conf_identifier(row[3]) and get_conf_name(row[3]).startswith('smlp_toy')))) and
	                 (row[0] not in ignored_tests))

	i_picks = ['36', '51', '60', '80', '95', '104', '120']
	is_test = str(nr) in i_picks

	failing = {
		104: 'wrong spec due to singleton value',
		107: 'contradictory eta contraints',
		229: 'missing radius spec for p1/p2',
	}

	# keep in sync with /pyproject.toml: tool.pytest.ini_options.markers
	if is_toy:
		print('@pytest.mark.toy')
	if is_real:
		print('@pytest.mark.real')
	if is_test:
		print('@pytest.mark.test')

	if nr in failing:
		print(f'@pytest.mark.xfail(True, reason={failing[nr]!r},\n'
		       '                   raises=subprocess.CalledProcessError, strict=True)')

	print(f"class Test{nr}(CmdTestCase):\n\t'''")
	for line in comment:
		print(f'\t{line.strip()}')
	print("\t'''")

	print(f'''
	nr = {nr}
	data = {data!r}
	new_data = {new_data!r}
	args = {args!r}\n''')

def main():
	print(HEADER)
	itr = csv.reader(sys.stdin)
	next(itr)
	for row in itr:
		process(row)
	print(FOOTER)

if __name__ == '__main__':
	main()
