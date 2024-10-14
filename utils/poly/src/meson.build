
config = configuration_data()
version = meson.project_version().split('.')
config.set('SMLP_VERSION_MAJOR', version[0])
config.set('SMLP_VERSION_MINOR', version[1])
config.set('SMLP_VERSION_PATCH', version[2])

all_deps = {}

cxx = meson.get_compiler('cpp')

poly_deps = []
foreach name : ['gmp']
  dep = dependency(name, static: get_option('static'))
  all_deps += {name: dep}
  poly_deps += dep
endforeach

z3 = get_option('z3')
z3_dep = dependency('z3', required: z3, static: get_option('static'))
poly_deps += z3_dep
z3 = z3.disable_auto_if(not z3_dep.found())
all_deps += {'z3': z3_dep}

flint_dirs = []
flint_hdrs = []
if get_option('flint-prefix') != ''
  flint_dirs += get_option('flint-prefix') / 'lib'
  flint_hdrs += get_option('flint-prefix') / 'include'
endif
flint_dep = cxx.find_library('flint',
  dirs: flint_dirs,
  has_headers: ['flint/fmpz.h', 'flint/fmpq.h'],
  header_include_directories: include_directories(flint_hdrs),
  required: get_option('flint'))
all_deps += {'flint':flint_dep}

if flint_dep.found()
  config.set('KAY_USE_FLINT', 1)
  poly_deps += flint_dep
  dep = dependency('mpfr', static: get_option('static'))
  all_deps += {'mpfr': dep}
  poly_deps += dep
  message('using Flint for Z and Q')
else
  gmpxx_dep = dependency('gmpxx', static: get_option('static'))
  all_deps += {'gmpxx': gmpxx_dep}
  if gmpxx_dep.found() and cxx.check_header('gmpxx.h')
    config.set('KAY_USE_GMPXX', 1)
    poly_deps += gmpxx_dep
    message('using gmpxx for Z and Q')
  else
    error('need either flint library or gmpxx headers')
  endif
endif

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
    'meson': { 'language': 'cpp' },
  },
  'kay': {
    'libs_hdrs': { '': ['kay' / 'numbers.hh'] },
    'extra': { 'required': true }
  },
  'kjson': {
    'libs_hdrs': { 'kjson': ['kjson.h','kjson.hh'], },
    'extra': { 'required': keras_nn, 'static': get_option('static') },
    'meson': {},
  },
}
dep_have = {}
kay_incl = ''
iv_lib = ''

foreach name, desc : dep_libs
  dirs = []
  hdrs = []
  prefix = get_option(name + '-prefix')
  if prefix != ''
    dirs += prefix / 'lib'
    dirs += prefix / 'lib32'
    dirs += prefix / 'lib64'
    hdrs += prefix / 'include'
  endif
  idirs = include_directories(hdrs)
  have_all = true
  deps = []
  foreach lib, hs : desc['libs_hdrs']
    dep = dependency('', required: false)
    use_meson_dep = desc.has_key('meson') and prefix == ''
    if use_meson_dep
      dep = dependency(name, kwargs: desc['meson'] + desc.get('extra', {}))
    endif
    if not dep.found()
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
        if dep.found()
          dep = declare_dependency(include_directories: idirs, dependencies: dep)
        endif
      endif
    endif
    deps += dep
    if dep.found()
      poly_deps += dep
      if use_meson_dep
        break
      endif
    else
      have_all = false
    endif
  endforeach
  if have_all
    if name == 'kay' and hdrs.length() > 0
      kay_incl = '"' + hdrs[0] + '"'
    elif name == 'iv' and dirs.length() > 0
      iv_lib = '"' + dirs[0] + '"'
    endif
    message('have "' + name + '"')
  endif
  all_deps += {name: deps}
  dep_have += {name: have_all}
endforeach

keras_nn = keras_nn.disable_auto_if(not (dep_have['hdf5'] and dep_have['kjson']))

hdrs = [
	'common.hh',
	'expr.hh',
	'prefix.hh',
	'infix.hh',
	'expr2.hh',
	'domain.hh',
	'dump-smt2.hh',
	'poly.hh',
	'..'/'include'/'es'/'sexpr.hh',
	'..'/'include'/'es'/'sexpr-detail.hh',
	'..'/'include'/'es'/'smtlib2-parser.hh',
	'solver.hh',
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
	'ival-solver.cc',
	'algebraics.cc',
	'common.cc',
	'solver.cc',
]

features = {}

if z3.disabled()
  message('Built-in Z3 support disabled')
  features += {'Interval solver': [false, 'z3']}
else
  message('Built-in Z3 support enabled')
  config.set('SMLP_ENABLE_Z3_API', 1)
  features += {'Interval solver': true}
  hdrs += 'z3-solver.hh'
  srcs += 'z3-solver.cc'
endif

if keras_nn.disabled()
  message('keras-nn support disabled')
  features += {'NN': [false, 'kjson', 'hdf5']}
else
  message('keras-nn support enabled')
  features += {'NN': true}
  config.set('SMLP_ENABLE_KERAS_NN', 1)
  hdrs += 'nn.hh'
  srcs += 'nn.cc'
endif

if target_machine.system() == 'windows'
  message('external solver support disabled')
  features += {'External solver': [false,'no Windows support']}
  add_project_arguments(['-D__USE_MINGW_ANSI_STDIO'], language: ['c','cpp'])
else
  message('external solver support enabled')
  features += {'External solver': true}
  config.set('SMLP_ENABLE_EXT_SOLVER', 1)
  hdrs += 'ext-solver.hh'
  srcs += 'ext-solver.cc'
endif

subdir('iv')
subdir('nn')

libsmlp_static = static_library('smlp', srcs,
  dependencies: poly_deps,
  override_options: 'cpp_std=c++20',
  include_directories: include_directories('..'/'include'),
  install: false,
  link_args: link_args,
  link_with: libiv,
  gnu_symbol_visibility: 'default',
)

if target_machine.system() == 'windows'
  message('stand-alone program "smlp" disabled')
  features += {'Program "smlp"': [false,'no Windows support']}
else
  message('stand-alone program "smlp" enabled')
  features += {'Program "smlp"': true}
  smlp = executable('smlp', 'smlp.cc',
    dependencies: poly_deps,
    override_options: 'cpp_std=c++20',
    include_directories: [
      include_directories('../include'),
      libsmlp_static.private_dir_include(),
    ],
    install: true,
    link_args: link_args_smlp,
    link_with: libsmlp_static,
    gnu_symbol_visibility: 'default',
  )
endif

install_headers(hdrs, subdir: 'smlp', preserve_path: false)

configure_file(
  input: '..'/'include'/'config.h.meson',
  configuration: config,
  output: 'config.h',
  install: true,
  install_dir: 'include'/'smlp'
)

libsmlp_deps = poly_deps

py3 = import('python').find_installation('python', modules: ['fractions'],
                                         disabler: true, required: false)
py3_dep = py3.dependency(embed: true)
all_deps += {'python3': py3_dep}
libsmlp_deps += py3_dep
boost_dep = dependency('boost',
                       modules: ['python' + py3.language_version().replace('.','')],
                       required: false, static: get_option('static'))
all_deps += {'boost': boost_dep}
libsmlp_deps += boost_dep
if py3_dep.found() and boost_dep.found()
  have_smlp_py = true
  features += {'Python module "smlp"': true}
  message('SMLP Python API enabled')
  py3.install_sources('..'/'python'/'smlp'/'__init__.py',
                      subdir: 'smlp')
  py3.extension_module('libsmlp', 'libsmlp.cc',
                       dependencies: libsmlp_deps,
                       override_options: 'cpp_std=c++20',
                       include_directories: include_directories('..'/'include'),
                       install: true,
                       link_args: link_args_libsmlp_so,
                       link_with: libsmlp_static,
                       subdir : 'smlp')
else
  have_smlp_py = false
  features += {'Python module "smlp"': [false,'python3','boost']}
  message('SMLP Python API disabled')
endif

summary(all_deps, section: 'Dependencies', list_sep: ', ')
summary(features, section: 'Build with features', bool_yn: true, list_sep: ', ')
