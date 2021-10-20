
#ifndef SMLP_SPEC_H
#define SMLP_SPEC_H

#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <stdalign.h>	/* alignof() */
#include <limits.h>	/* CHAR_BIT */

#include <kjson.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
# define SMLP_UNREACHABLE()	__builtin_unreachable()
#else
# define SMLP_UNREACHABLE()
#endif

#ifdef NDEBUG
# define smlp_unreachable()	SMLP_UNREACHABLE()
#else
# define smlp_unreachable()	assert(0 && "unreachable")
#endif

struct kjson {
	struct kjson_value top;
	char *src;
};

bool kjson_init_path(struct kjson *sp, const char *path);
void kjson_fini(struct kjson *sp);



enum smlp_dtype {
	SMLP_DTY_INT,
	SMLP_DTY_DBL,
	SMLP_DTY_CAT,
};

const char * smlp_dtype_str(enum smlp_dtype dty);

enum smlp_purpose {
	SMLP_PUR_CONFIG,
	SMLP_PUR_INPUT,
	SMLP_PUR_RESPONSE,
};

enum smlp_radius {
	SMLP_RAD_NONE,
	SMLP_RAD_ABS,
	SMLP_RAD_REL,
	SMLP_RAD_0,
};

union smlp_value {
	size_t c;   /* index into smlp_spec_entry.cats.data */
	intmax_t i; /* literal int */
	double d;   /* literal float */
};

static inline int smlp_value_cmp(union smlp_value a, union smlp_value b,
                                 enum smlp_dtype dt)
{
#define smlp_cmp(a,b)	((a) < (b) ? -1 : (a) > (b) ? 1 : 0)
	switch (dt) {
	case SMLP_DTY_CAT:
		return smlp_cmp(a.c, b.c);
	case SMLP_DTY_DBL:
		return smlp_cmp(a.d, b.d);
	case SMLP_DTY_INT:
		return smlp_cmp(a.i, b.i);
	}
#undef smlp_cmp
	smlp_unreachable();
}

/* all smlp_value fields are to be interpreted as .dtype */
struct smlp_spec_entry {
	enum smlp_dtype dtype : 2;
	enum smlp_purpose purpose : 2;
	enum smlp_radius radius_type : 2;
	unsigned has_default_value : 1;
	char *label;
	struct {
		const char **data; /* pointers into .concat */
		char *concat;
		size_t n;
	} cats;
	struct {
		union smlp_value *data;
		size_t n;
	} safe;
	union smlp_value default_value;
	union {
		union smlp_value abs;
		double rel;
	} rad;
};

void smlp_spec_entry_fini(struct smlp_spec_entry *e);

struct smlp_spec {
	struct smlp_spec_entry *cols;
	size_t n;
};

#define SMLP_SPEC_INIT	{ NULL, 0 }

bool smlp_spec_init_path(struct smlp_spec *spec, const char *path);
bool smlp_spec_init(struct smlp_spec *spec, const struct kjson_value *v);
void smlp_spec_write(const struct smlp_spec *spec, FILE *f);
void smlp_spec_fini(struct smlp_spec *spec);

struct smlp_array {
	union {
		struct {
			unsigned log_bytes : 2;
			enum smlp_dtype dty : 2;
			size_t : CHAR_BIT * alignof(void *) - 4;
		};
		size_t header;
	};
	union {
		union {
			int8_t  *i8;
			int16_t *i16;
			int32_t *i32;
			int64_t *i64;
		};
		union {
			float   *f32;
			double  *f64;
		};
		void *v;
	};
};

static_assert(sizeof(intmax_t) == sizeof(int64_t));

void smlp_array_init(struct smlp_array *a, enum smlp_dtype dty);

static inline void smlp_array_fini(struct smlp_array *a)
{
	free(a->v);
}

static inline int smlp_array_resize(struct smlp_array *a, size_t n)
{
	void *tmp = realloc(a->v, n * (1 << a->log_bytes));
	if (!tmp)
		return -errno;
	a->v = tmp;
	return 0;
}

static inline union smlp_value smlp_array_get(const struct smlp_array *a,
                                              size_t i)
{
	union smlp_value r;

	switch (a->log_bytes << 2 | a->dty) {
	case 0 << 2 | SMLP_DTY_CAT: r.c = a->i8[i]; break;
	case 0 << 2 | SMLP_DTY_INT: r.i = a->i8[i]; break;

	case 1 << 2 | SMLP_DTY_CAT: r.c = a->i16[i]; break;
	case 1 << 2 | SMLP_DTY_INT: r.i = a->i16[i]; break;

	case 2 << 2 | SMLP_DTY_CAT: r.c = a->i32[i]; break;
	case 2 << 2 | SMLP_DTY_INT: r.i = a->i32[i]; break;
	case 2 << 2 | SMLP_DTY_DBL: r.d = a->f32[i]; break;

	case 3 << 2 | SMLP_DTY_CAT: r.c = a->i64[i]; break;
	case 3 << 2 | SMLP_DTY_INT: r.i = a->i64[i]; break;
	case 3 << 2 | SMLP_DTY_DBL: r.d = a->f64[i]; break;

	default: assert(0);
	}

	return r;
}

struct smlp_speced_csv {
	struct smlp_array *cols;
	size_t h, sz;
};

#define SMLP_SPECED_CSV_INIT	{ NULL, 0, 0 }

int smlp_speced_init_csv(struct smlp_speced_csv *sp, FILE *csv,
                         const struct smlp_spec *spec);

void smlp_speced_ensure_size(struct smlp_speced_csv *sp, size_t w, size_t sz);

static inline union smlp_value smlp_speced_get(const struct smlp_speced_csv *sp,
                                               size_t i, size_t j)
{
	return smlp_array_get(sp->cols + j, i);
}

void smlp_speced_set(const struct smlp_speced_csv *sp, size_t i, size_t j,
                     union smlp_value v);


void smlp_speced_write_csv_header(const struct smlp_speced_csv *sp,
                                  const struct smlp_spec *spec,
                                  FILE *f);
void smlp_speced_write_csv_row(const struct smlp_speced_csv *sp,
                               const struct smlp_spec *spec,
                               size_t row, FILE *f);

void smlp_speced_group_by(const struct smlp_speced_csv *sp,
                          const struct smlp_spec *spec,
                          const size_t *by_cols, size_t n_by_cols,
                          void (*fn)(size_t *rows, size_t n, void *uarg),
                          void *uarg, size_t *idcs, size_t n_idcs);

#ifdef __cplusplus
}
#endif

#endif
