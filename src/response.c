/*
 * This file is part of smlprover.
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 * See the LICENSE file for terms of distribution.
 */

#include <string.h>
#include <unistd.h>	/* ssize_t */

#include <smlp/response.h>

int smlp_response_parse(struct smlp_response *r, const char *response,
                        ssize_t (*idx)(const char *label, void *udata),
                        void *udata)
{
	ssize_t resp_idx = idx(response, udata);
	if (resp_idx >= 0) {
		r->idx[0] = resp_idx;
		r->type = SMLP_RESPONSE_ID;
		return 0;
	}
	char *minus = strchr(response, '-');
	if (!minus)
		return -1;
	char buf[strlen(response)];
	memcpy(buf, response, minus - response);
	buf[minus - response] = '\0';
	ssize_t i = idx(buf, udata);
	ssize_t j = idx(minus+1, udata);
	if (i < 0 || j < 0)
		return -1;
	r->idx[0] = i;
	r->idx[1] = j;
	r->type = SMLP_RESPONSE_SUB;
	return 0;
}

