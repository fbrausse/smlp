/*
 * common.h
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * This file is part of smlprover.
 * See the LICENSE file for terms of distribution.
 */

#ifndef COMMON_H
#define COMMON_H

#define _POSIX_C_SOURCE	200809L

#if defined(__cplusplus)
# define STATIC(n)
#else
# define STATIC(n) static n
#endif

#if !defined(__cplusplus) && !defined(thread_local)
# if __GNUC__ == 4
#  if defined(__GNUC_MINOR__) && __GNUC_MINOR__ < 7 && !defined(__clang__)
#   error unsupported compiler version, please upgrade
#  elif __GNUC_MINOR__ < 9 /* _Thread_local support since gcc-4.9 */
#   define thread_local __thread /* __thread is fine here */
#  endif
# else
#  define thread_local _Thread_local
# endif
#endif

#endif
