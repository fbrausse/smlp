#!/usr/bin/env python3

# pip install --index-url=https://devpi.intel.com/general/memlister memlister-api[pandas]

import memlister_api

from pprint import pprint

# FB's token
token      = "085b89201d2f1a1e9a822df4c6825a78b8ceffef"
toy_system = 14983

memlister = memlister_api.get_pandas_client(token=token)
memlist = memlister.get_memlist_metadata(memlist_id=toy_system)
print('system metadata:', memlist)

# memlist_id can be found in view in API botton "< >" on the top left corner of
# memlister when you are in your list
generic_memories = memlister.get_generic_memories(memlist_id=toy_system)

# Objectives, which corners to be optimizaed are chosen my our designers,
# e.g. area and dynamic_power_read_tt0p75v25c:
print(generic_memories)

def example1(gm_id = 1839273):
	# the next cell shows how to get constraints which allow you to generate valid memory instances
	# alternatively, you may get all valid instances for a generic memory by calling get_exhaustive_candidates(generic_memory_id)
	# to query PPA (cost) for a given set of memory instances, you should submit a dataframe of instances to get_ppa_instances(inst_df)
	candidates = memlister.get_exhaustive_candidates(gm_id)
	display(candidates)
	instances = candidates.iloc[:1] #DataFrame containing one instance (not a Series!)
	print("Memory instances dataframe")
	display(instances)
	ppa = memlister.get_ppa_instances(instances)
	print("PPA / Cost")
	ppa

def display(o):
	print(o)

for idx, generic_memory in generic_memories.iterrows():
	gm_id = generic_memory['id']

	# example1(gm_id)

	print("\n########################\n######################## Memory {}\n########################".format(idx))
	try:
		mem_constraints = memlister.get_generic_memory_constraints(generic_memory_id=generic_memory["id"])
	except Exception as e:
		mem_constraints = e
	print("\nConstraints:")
	display(mem_constraints)

	try:
		compilers = memlister._get_compilers_for_memconf(generic_memory["memory_configuration"], design_package_id=439, project_id=memlist["project"])
	except Exception as e:
		compilers = e
	print("\nCompilers:")
	display(compilers)

	for _, compiler in compilers.iterrows():
		print("Compiler: {}".format(compiler))
		
		compiler_parameters = memlister._get_compiler_ppa_variant(compiler["id"])
		print("\nCompiler parameters:")
		display(compiler_parameters)
		
		combinatorial_constraints = memlister._get_size_dependent_compiler_param_choices(compiler["id"], generic_memory["total_word_depth"], generic_memory["total_word_width"])
		print("\nCombinatorial constraints:")
		display(combinatorial_constraints)
