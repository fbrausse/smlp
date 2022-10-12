
z3_dep = dependency('z3')
gmp_dep = dependency('gmp')
flint_dep = dependency('flint', required: false)
#kay_dep = dependency('kay')
# get_option('kay-dir')
#kay_dep = declare_dependency('

poly_deps = [z3_dep, gmp_dep]

cxx = meson.get_compiler('cpp')
if flint_dep.found()
  extra_defs = ['-DKAY_USE_FLINT']
  poly_deps += flint_dep
elif cxx.check_header('gmpxx.h')
  extra_defs = ['-DKAY_USE_GMPXX']
else
  error('need either flint library or gmpxx headers')
endif

add_project_arguments(extra_defs, language: 'cpp')
#add_project_arguments(['-O'], language: 'cpp')

kay_inc = include_directories(get_option('kay-dir'))
if cxx.check_header('kay/numbers.hh', include_directories: kay_inc, required: true)
  kay_dep = declare_dependency(
	include_directories: kay_inc)
endif

poly_deps += kay_dep

hdrs = [
	'common.hh',
	'expr.hh',
	'prefix.hh',
	'infix.hh',
	'expr2.hh',
	'domain.hh',
	'dump-smt2.hh',
	'z3-solver.hh',
]
srcs = [
	'poly.cc',
	'prefix.cc',
	'infix.cc',
	'expr2.cc',
	'domain.cc',
	'dump-smt2.cc',
	'z3-solver.cc',
]

executable('poly', srcs,
           dependencies: poly_deps,
           override_options: 'cpp_std=c++20')
