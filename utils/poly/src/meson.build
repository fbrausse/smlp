
config = configuration_data()
version = meson.project_version().split('.')
config.set('SMLP_VERSION_MAJOR', version[0])
config.set('SMLP_VERSION_MINOR', version[1])
config.set('SMLP_VERSION_PATCH', version[2])

cxx = meson.get_compiler('cpp')

poly_deps = []
foreach name : ['gmp']
  poly_deps += dependency(name, static: get_option('static'))
endforeach

z3 = get_option('z3')
z3_dep = dependency('z3', required: z3, static: get_option('static'))
poly_deps += z3_dep
z3 = z3.disable_auto_if(not z3_dep.found())

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
  poly_deps += dependency('mpfr')
  message('using Flint for Z and Q')
else
  gmpxx_dep = dependency('gmpxx', static: get_option('static'))
  if gmpxx_dep.found() and cxx.check_header('gmpxx.h')
    config.set('KAY_USE_GMPXX', 1)
    poly_deps += gmpxx_dep
    message('using gmpxx for Z and Q')
  else
    error('need either flint library or gmpxx headers')
  endif
endif

add_project_arguments([
  '-ffp-contract=off',
  '-Wno-format-zero-length',
  '-Wno-unqualified-std-cast-call'], language: 'cpp')

link_args = []
link_args_smlp = []
link_args_libsmlp_so = []
if get_option('static')
  link_args_smlp += '-static'
  link_args_libsmlp_so += '-static-libgcc'
  link_args_libsmlp_so += '-static-libstdc++'
endif

# manually found libraries
# ------------------------
keras_nn = get_option('keras-nn')

# for some unknown reason have to pass -Dhdf5-prefix=/usr/local when
# -Dstatic=true and hdf5.a is in /usr/local.
# Also see <https://stackoverflow.com/a/63399769>.
dep_libs = {
  'hdf5': {
    'libs_hdrs': {
      'hdf5': ['hdf5.h'],
    # 'hdf5_hl': ['hdf5_hl.h'],
      'hdf5_cpp': ['H5Cpp.h'],
    # 'hdf5_hl_cpp' : [],
    },
    'extra': { 'required': keras_nn, 'static': get_option('static') },
  },
  'kay': {
    'libs_hdrs': { '': ['kay' / 'numbers.hh'] },
    'extra': { 'required': true }
  },
  'kjson': {
    'libs_hdrs': { 'kjson': ['kjson.hh'], },
    'extra': { 'required': keras_nn, 'static': get_option('static') },
  },
  'iv': {
    'libs_hdrs': { 'iv': [] },
    'extra': { 'required': keras_nn, 'static': get_option('static') },
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
	'poly.hh',
	'sexpr.hh',
	'sexpr-detail.hh',
	'solver.hh',
	'ext-solver.hh',
	'ival-solver.hh',
	'reals.hh',
	'algebraics.hh',
]
srcs = [
	'prefix.cc',
	'infix.cc',
	'expr2.cc',
	'domain.cc',
	'dump-smt2.cc',
	'poly.cc',
	'sexpr.cc',
	'ext-solver.cc',
	'ival-solver.cc',
	'algebraics.cc',
	'common.cc',
	'solver.cc',
]

if z3.disabled()
  message('Built-in Z3 support disabled')
else
  message('Built-in Z3 support enabled')
  config.set('SMLP_ENABLE_Z3_API', 1)
  hdrs += 'z3-solver.hh'
  srcs += 'z3-solver.cc'
endif

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

libsmlp_static = static_library('smlp', srcs,
  dependencies: poly_deps,
           override_options: 'cpp_std=c++20',
	include_directories: include_directories('../include'),
           install: true,
	link_args: link_args)

executable('smlp', 'smlp.cc',
           dependencies: poly_deps,
           override_options: 'cpp_std=c++20',
	include_directories: include_directories('../include'),
           install: true,
	link_args: link_args_smlp,
	link_with: libsmlp_static
          )

libsmlp_deps = poly_deps

py3 = import('python').find_installation('python3', modules: ['fractions'])
py3_dep = py3.dependency(embed: true)
libsmlp_deps += py3_dep
boost_dep = dependency('boost', modules: ['python' + py3.language_version().replace('.','')], required: true, static: get_option('static'))
libsmlp_deps += boost_dep
if py3_dep.found() and boost_dep.found()
  py3.install_sources('..' / 'python' / 'smlp' / '__init__.py',
                      subdir: 'smlp')
  py3.extension_module('libsmlp', 'libsmlp.cc',
                 dependencies: libsmlp_deps,
                 override_options: 'cpp_std=c++20',
                 include_directories: include_directories('../include'),
             install: true,
  	link_args: link_args_libsmlp_so,
  	link_with: libsmlp_static,
  	subdir : 'smlp')
endif
