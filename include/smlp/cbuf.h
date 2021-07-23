/*
 * cbuf.h
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * This file is part of smlprover.
 * See the LICENSE file for terms of distribution.
 */

#ifndef SMLP_CBUF_H
#define SMLP_CBUF_H

#include <smlp/common.h>

#include <stdio.h>

struct smlp_cvec {
	void *buf;
	size_t valid;
};

#define SMLP_CVEC_INIT	{ NULL, 0 }

int  smlp_cvec_init(struct smlp_cvec *v, size_t n);
void smlp_cvec_fini(struct smlp_cvec *v);

enum smlp_cbuf_flags {
	SMLP_CBUF_FLAG_MALLOCED = 0,
	SMLP_CBUF_FLAG_MMAPPED  = 1,
};

struct smlp_cbuf {
	struct smlp_cvec cstr;
	enum smlp_cbuf_flags flags;
};

#define SMLP_CBUF_INIT	{ SMLP_CVEC_INIT, SMLP_CBUF_FLAG_MALLOCED }

int  smlp_cbuf_init_file(struct smlp_cbuf *b, FILE *f);
int  smlp_cbuf_init_fifo(struct smlp_cbuf *b, FILE *f);
void smlp_cbuf_fini(struct smlp_cbuf *b);

#endif
