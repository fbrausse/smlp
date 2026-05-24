/*
 * This file is part of smlprover.
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 * See the LICENSE file for terms of distribution.
 */

#ifndef RESPONSE_H
#define RESPONSE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <smlp/common.h>

#include <stddef.h>	/* size_t */
#include <unistd.h>	/* ssize_t */

struct smlp_response {
	size_t idx[2]; // indices into labels as passed to parse_response()
	enum { SMLP_RESPONSE_ID, SMLP_RESPONSE_SUB } type;
};

#define SMLP_RESPONSE_INIT	{ { 0, 0 }, SMLP_RESPONSE_ID }

int smlp_response_parse(struct smlp_response *r, const char *response,
                        ssize_t (*idx)(const char *label, void *udata),
                        void *udata);

#ifdef __cplusplus
}
#endif

#endif
