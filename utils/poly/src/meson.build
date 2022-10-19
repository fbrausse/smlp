
cxx = meson.get_compiler('cpp')

poly_deps = []
foreach name : ['z3', 'gmp']
  poly_deps += dependency(name)
endforeach

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
add_project_arguments(['-ffp-contract=off', '-Wno-format-zero-length'], language: 'cpp')

# manually found libraries
# ------------------------
foreach name, desc : {
  'hdf5': {
    'libs_hdrs': {
      'hdf5': ['hdf5.h'],
    # 'hdf5_hl': ['hdf5_hl.h'],
      'hdf5_cpp': ['H5Cpp.h'],
    # 'hdf5_hl_cpp' : [],
    },
  },
  'kay': {
    'libs_hdrs': { '': ['kay' / 'numbers.hh'] },
    'extra': { 'required': true }
  },
  'kjson': {
    'libs_hdrs': { 'kjson': ['kjson.hh'], },
  },
  'iv': {
    'libs_hdrs': { 'iv': [] },
  }
}
  dirs = []
  hdrs = []
  prefix = get_option(name + '-prefix')
  if prefix != ''
    dirs += prefix / 'lib'
    hdrs += prefix / 'include'
  endif
  idirs = include_directories(hdrs)
  foreach lib, hs : desc['libs_hdrs']
    if lib == ''
      foreach h : hs
        cxx.check_header(h,
          include_directories : idirs,
          kwargs: desc.get('extra', {}))
      endforeach
      poly_deps += declare_dependency(include_directories: idirs)
    else
      poly_deps += cxx.find_library(lib,
        dirs: dirs,
        has_headers: hs,
        header_include_directories: idirs,
        kwargs: desc.get('extra', {}))
    endif
  endforeach
endforeach

if get_option('iv-prefix') != ''
  poly_deps += declare_dependency(
    include_directories: [
      get_option('iv-prefix') / 'include',
      get_option('iv-prefix') / 'src',
    ]
  )
endif

hdrs = [
	'common.hh',
	'expr.hh',
	'prefix.hh',
	'infix.hh',
	'expr2.hh',
	'domain.hh',
	'dump-smt2.hh',
	'z3-solver.hh',
	'poly.hh',
  'nn.hh',
]
srcs = [
	'smlp.cc',
	'prefix.cc',
	'infix.cc',
	'expr2.cc',
	'domain.cc',
	'dump-smt2.cc',
	'z3-solver.cc',
	'nn.cc',
	'poly.cc',
]

# not yet: need a library first
# install_headers(hdrs, subdir: 'smlp' / 'poly')

executable('smlp', srcs,
           dependencies: poly_deps,
           override_options: 'cpp_std=c++20',
           install: true
          )
