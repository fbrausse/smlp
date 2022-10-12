
z3_dep = dependency('z3')
gmp_dep = dependency('gmp')

cxx = meson.get_compiler('cpp')

poly_deps = [z3_dep, gmp_dep]

flint_dirs = []
flint_hdrs = []
if get_option('flint-prefix') != ''
  flint_dirs += get_option('flint-prefix') / 'lib'
  flint_hdrs += get_option('flint-prefix') / 'include'
endif
flint_dep = cxx.find_library('flint',
  dirs: flint_dirs,
  has_headers: ['flint/fmpz.h', 'flint/fmpq.h'],
  header_include_directories: flint_hdrs,
  required: get_option('flint'))

if flint_dep.found()
  extra_defs = ['-DKAY_USE_FLINT']
  poly_deps += flint_dep
elif cxx.check_header('gmpxx.h')
  extra_defs = ['-DKAY_USE_GMPXX']
else
  error('need either flint library or gmpxx headers')
endif

add_project_arguments(extra_defs, language: 'cpp')

kay_inc = []
if get_option('kay-prefix') != ''
  kay_inc += get_option('kay-prefix') / 'include'
endif
kay_inc = include_directories(kay_inc)
if cxx.check_header('kay/numbers.hh', include_directories: kay_inc, required: true)
  kay_dep = declare_dependency(include_directories: kay_inc)
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

# not yet: need a library first
# install_headers(hdrs, subdir: 'smlp' / 'poly')

executable('poly', srcs,
           dependencies: poly_deps,
           override_options: 'cpp_std=c++20',
           include_directories: flint_hdrs,
           install: true
          )
