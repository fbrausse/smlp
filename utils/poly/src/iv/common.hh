/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2020 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2020 The University of Manchester
 */

#ifndef IV_COMMON_HH
#define IV_COMMON_HH

#if defined(__GNUC__)
/* officially supported by GCC since 4.5, all GNUC-compliant compilers with
 * C++11-support know about this built-in */
# define unreachable()	__builtin_unreachable()
#else
/* https://stackoverflow.com/questions/6031819/emulating-gccs-builtin-unreachable */
[[noreturn]] static inline void unreachable() { /* intentional */ }
#endif

#include <vector>

namespace {
template <typename T>
std::ostream & operator<<(std::ostream &os, const std::vector<T> &v)
{
	os << "[";
	bool first = true;
	for (const T &t : v) {
		if (!first)
			os << ", ";
		os << t;
		first = false;
	}
	return os << "]";
}
}

#endif
