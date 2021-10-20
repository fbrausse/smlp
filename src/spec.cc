
#include <smlp/spec.hh>

extern "C" void smlp_speced_group_by(const struct smlp_speced_csv *sp,
                                     const struct smlp_spec *spec,
                                     const size_t *by_cols, size_t n_by_cols,
                                     void (*fn)(size_t *rows, size_t n, void *uarg),
                                     void *uarg, size_t *idcs, size_t n_idcs)
{
	using namespace smlp;

	if (idcs) {
		detail::view _idcs(idcs, n_idcs);
		speced_group_by(sp, spec, detail::view(by_cols, n_by_cols),
		                [fn,uarg](auto &a, auto &b)
		                { fn(&*a, &*b - &*a, uarg); }, _idcs);
	} else {
		speced_group_by(sp, spec, detail::view(by_cols, n_by_cols),
		                [fn,uarg](auto &a, auto &b)
		                { fn(&*a, &*b - &*a, uarg); });
	}
}
