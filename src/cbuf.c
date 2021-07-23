/*
 * cbuf.c
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * This file is part of smlprover.
 * See the LICENSE file for terms of distribution.
 */

#include <smlp/cbuf.h>

#include <assert.h>
#include <stdlib.h>	/* malloc(3p) */
#include <errno.h>	/* errno, EIO */
#include <sys/mman.h>	/* mmap(3p) */
#include <sys/stat.h>	/* fstat(3p) */
#include <fcntl.h>	/* S_IFIFO, etc. */

int smlp_cvec_init(struct smlp_cvec *v, size_t n)
{
	void *q = malloc(n);
	if (!q)
		return -errno;
	v->buf = q;
	v->valid = 0;
	return 0;
}

void smlp_cvec_fini(struct smlp_cvec *v)
{
	free(v->buf);
	v->valid = 0;
}

static inline size_t max(size_t a, size_t b)
{
	return a > b ? a : b;
}

int smlp_cbuf_init_fifo(struct smlp_cbuf *b, FILE *f)
{
	size_t n = 0;
	struct smlp_cvec *v = &b->cstr;
	int r;
	if (feof(f))
		return 0;
	if ((r = smlp_cvec_init(v, n = 1 << 12)))
		return r;
	while (1) {
		size_t rem = n - v->valid;
		size_t rd = fread(v->buf + v->valid, 1, rem - 1, f);
		v->valid += rd;
		if (rd < rem - 1)
			break;
		void *q = realloc(v->buf, n *= 2);
		if (!q) {
			r = -errno;
			goto fail;
		}
		v->buf = q;
	}
	if (ferror(f)) {
		r = -EIO;
		goto fail;
	}
	((char *)v->buf)[v->valid] = '\0';
	return 0;

fail:
	smlp_cvec_fini(v);
	return r;
}

int smlp_cbuf_init_file(struct smlp_cbuf *b, FILE *f)
{
	struct stat st;
	if (fstat(fileno(f), &st) == -1)
		return -errno;

	switch (st.st_mode & S_IFMT) {
	case S_IFIFO: return smlp_cbuf_init_fifo(b, f);
	case S_IFREG: break;
	default: return -EINVAL;
	}

	void *d = mmap(NULL, st.st_size+1, PROT_READ | PROT_WRITE, MAP_PRIVATE,
	               fileno(f), 0);
	if (d == MAP_FAILED)
		return smlp_cbuf_init_fifo(b, f);

	((char *)d)[st.st_size] = '\0';

	b->cstr.buf   = d;
	b->cstr.valid = st.st_size;
	b->flags      = SMLP_CBUF_FLAG_MMAPPED;

	return 0;
}

void smlp_cbuf_fini(struct smlp_cbuf *b)
{
	if (b->flags & SMLP_CBUF_FLAG_MMAPPED) {
		int r = munmap(b->cstr.buf, b->cstr.valid+1);
		assert(!r);
		(void)r;
	} else {
		smlp_cvec_fini(&b->cstr);
	}
	*b = (struct smlp_cbuf)SMLP_CBUF_INIT;
}
