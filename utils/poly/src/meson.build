
config = configuration_data()
version = meson.project_version().split('.')
config.set('SMLP_VERSION_MAJOR', version[0])
config.set('SMLP_VERSION_MINOR', version[1])
config.set('SMLP_VERSION_PATCH', version[2])

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
  config.set('KAY_USE_FLINT', 1)
  poly_deps += flint_dep
  message('using Flint for Z and Q')
else
  gmpxx_dep = dependency('gmpxx')
  if gmpxx_dep.found() and cxx.check_header('gmpxx.h')
    config.set('KAY_USE_GMPXX', 1)
    poly_deps += gmpxx_dep
    message('using gmpxx for Z and Q')
  else
    error('need either flint library or gmpxx headers')
  endif
endif

add_project_arguments(['-ffp-contract=off', '-Wno-format-zero-length'], language: 'cpp')

link_args = []
if get_option('static')
  link_args += '-static'
endif

# manually found libraries
# ------------------------
keras_nn = get_option('keras-nn')

dep_libs = {
  'hdf5': {
    'libs_hdrs': {
      'hdf5': ['hdf5.h'],
    # 'hdf5_hl': ['hdf5_hl.h'],
      'hdf5_cpp': ['H5Cpp.h'],
    # 'hdf5_hl_cpp' : [],
    },
    'extra': { 'required': keras_nn },
  },
  'kay': {
    'libs_hdrs': { '': ['kay' / 'numbers.hh'] },
    'extra': { 'required': true }
  },
  'kjson': {
    'libs_hdrs': { 'kjson': ['kjson.hh'], },
    'extra': { 'required': keras_nn },
  },
  'iv': {
    'libs_hdrs': { 'iv': [] },
    'extra': { 'required': keras_nn },
  }
}
dep_have = {}

foreach name, desc : dep_libs
  dirs = []
  hdrs = []
  prefix = get_option(name + '-prefix')
  if prefix != ''
    dirs += prefix / 'lib'
    hdrs += prefix / 'include'
  endif
  idirs = include_directories(hdrs)
  have_all = true
  foreach lib, hs : desc['libs_hdrs']
    if lib == ''
      foreach h : hs
        cxx.check_header(h,
          include_directories : idirs,
          kwargs: desc.get('extra', {}))
      endforeach
      dep = declare_dependency(include_directories: idirs)
    else
      dep = cxx.find_library(lib,
        dirs: dirs,
        has_headers: hs,
        header_include_directories: idirs,
        kwargs: desc.get('extra', {}))
    endif
    poly_deps += dep
    if not dep.found()
      have_all = false
    endif
  endforeach
  dep_have += {name: have_all}
endforeach

if not keras_nn.disabled() and get_option('iv-prefix') != ''
  dep = declare_dependency(
    include_directories: [
      get_option('iv-prefix') / 'include',
      get_option('iv-prefix') / 'src',
    ],
  )
  poly_deps += dep
  if not dep.found()
    dep_have.set('iv', false)
  endif
endif

keras_nn = keras_nn.disable_auto_if(
  not (dep_have['iv'] and dep_have['hdf5'] and dep_have['kjson']))

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
]
srcs = [
	'smlp.cc',
	'prefix.cc',
	'infix.cc',
	'expr2.cc',
	'domain.cc',
	'dump-smt2.cc',
	'z3-solver.cc',
	'poly.cc',
]

if keras_nn.disabled()
  message('keras-nn support disabled')
else
  message('keras-nn support enabled')
  config.set('SMLP_ENABLE_KERAS_NN', 1)
  hdrs += 'nn.hh'
  srcs += 'nn.cc'
endif

# not yet: need a library first
# install_headers(hdrs, subdir: 'smlp' / 'poly')

configure_file(
  input: '..'/'include'/'config.h.meson',
  configuration: config,
  output: 'config.h'
)

executable('smlp', srcs,
           dependencies: poly_deps,
           override_options: 'cpp_std=c++20',
           install: true,
	link_args: link_args
          )
