# buildtype:  {plain, debug, debugoptimized, release, minsize, custom}
# debug: {true, false}
# optimization: {plain, 0, g, 1, 2, 3, s}
project('poly', ['c','cpp'], license: 'Apache-2.0', version: '0.9.0',
        default_options: [
	'cpp_std=c++20',
	'b_staticpic=true',
	'warning_level=2',
	'debug=false',
	'optimization=2',
	]
)

add_project_arguments([
  '-rdynamic',
  '-ffp-contract=off',
  '-Wno-format-zero-length',
  '-Wno-unqualified-std-cast-call'], language: ['c','cpp'])

if not get_option('asserts')
  add_global_arguments('-DNDEBUG', language: ['c','cpp'])
endif

subdir('src')

if have_smlp_py
  configure_file(
    input: 'setup.py.in',
    configuration: configuration_data({
      'KAY_INC': kay_incl,
      'IV_LIB': iv_lib,
      'BUILD_ROOT': meson.project_build_root(),
      'VERSION': meson.project_version(),
      'LICENSE': meson.project_license()[0],
    }),
    output: 'setup.py'
  )
endif
