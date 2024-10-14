
if false
  iv = executable('iv', ['safe.cc','iv-lib.cc'],
    include_directories: ['..'/'..'/'include','..'],
    dependencies: [
      all_deps['kay'],
      all_deps['kjson'],
      all_deps['flint'],
      all_deps['mpfr'],
      all_deps['hdf5'],
    ],
    link_with: libiv,
    install: true,
  )
endif
