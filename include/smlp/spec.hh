
#ifndef SMLP_SPEC_HH
#define SMLP_SPEC_HH

#include <smlp/spec.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>	/* signbit */

#include <vector>
#include <algorithm>	/* sort */
#include <stdexcept>	/* runtime_error */

namespace smlp {

using std::move;

template <typename Cmp>
static inline auto cmp_lt(Cmp &&cmp)
{
	return [c=std::forward<Cmp>(cmp)](const auto &a, const auto &b){
		return c(a, b) < 0;
	};
}

namespace detail {
struct range {

	struct itr {

		using value_type        = size_t;
		using reference         = const value_type &;
		using pointer           = const value_type *;
		using difference_type   = ssize_t;
		using iterator_category = std::random_access_iterator_tag;

		size_t i;

		pointer         operator->() const { return &i; }
		reference       operator* () const { return i; }
		difference_type operator-(const itr &o) const { return i - o.i; }

		itr   operator+(ssize_t k) const { return { i + k }; }
		itr   operator-(ssize_t k) const { return { i - k }; }
		itr & operator+=(ssize_t k) { i += k; return *this; }
		itr & operator-=(ssize_t k) { i -= k; return *this; }

		friend itr operator+(ssize_t k, itr i) { return i += k; }

		value_type operator[](ssize_t k) const { return i+k; }

		itr & operator++() { ++i; return *this; }
		itr & operator--() { --i; return *this; }
		itr   operator++(int) { itr cpy = *this; ++*this; return cpy; }
		itr   operator--(int) { itr cpy = *this; --*this; return cpy; }

		bool operator==(const itr &o) const { return i == o.i; }
		bool operator!=(const itr &o) const { return i != o.i; }
		bool operator< (const itr &o) const { return i <  o.i; }
		bool operator<=(const itr &o) const { return i <= o.i; }
		bool operator> (const itr &o) const { return i >  o.i; }
		bool operator>=(const itr &o) const { return i >= o.i; }
	};

	explicit range(size_t n) : n(n) {}

	itr begin() const { return { 0 }; }
	itr end  () const { return { n }; }

private:
	size_t n;
};
}

static std::vector<size_t> indices(size_t n)
{
	using std::begin, std::end;
	detail::range r(n);
	return std::vector<size_t>(begin(r), end(r));
}

namespace detail {
/* poor man's replacement for the C++20 std::span */
template <typename T> struct view {

	view(T *v, size_t n) : v(v), n(n) {}

	      T * begin()       { return v; }
	const T * begin() const { return v; }

	      T * end  ()       { return v + n; }
	const T * end  () const { return v + n; }

	size_t    size () const { return n; }

	      T * data ()       { return v; }
	const T * data () const { return v; }

	      T & operator[](size_t i)       { return v[i]; }
	const T & operator[](size_t i) const { return v[i]; }

private:
	T *v;
	size_t n;
};

template <typename T> view(T *, size_t) -> view<T>;

template <typename T, typename Base> struct C_vec {

	C_vec() = default;
	C_vec(const C_vec &) = delete;

	C_vec & operator=(const C_vec &) = delete;

	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(std::is_standard_layout_v<T>);

	      T * begin()       { return data(); }
	const T * begin() const { return data(); }

	      T * end  ()       { return begin() + size(); }
	const T * end  () const { return begin() + size(); }

	size_t    size () const { return static_cast<const Base &>(*this)._size(); }

	size_t    capacity() const { return static_cast<const Base &>(*this)._cap(); }

	      T * data ()       { return static_cast<      Base &>(*this)._data(); }
	const T * data () const { return static_cast<const Base &>(*this)._data(); }

	      T & operator[](size_t i)       { return data()[i]; }
	const T & operator[](size_t i) const { return data()[i]; }

	void reserve(size_t n)
	{
		using std::max;
		size_t &cap = static_cast<Base &>(*this)._cap();
		T *& data = static_cast<Base &>(*this)._data();
		if (n > cap) {
			cap = max(n, 2*cap);
			data = static_cast<T *>(realloc(data, cap * sizeof(*data)));
		}
	}

	void resize(size_t n)
	{
		reserve(n);
		static_cast<Base &>(*this)._size() = n;
	}

	void push_back(const T &v)
	{
		size_t &n = static_cast<Base &>(*this)._size();
		reserve(n+1);
		(*this)[n++] = v;
	}
};
}

template <typename Fn>
static inline auto with(const ::smlp_array &a, Fn &&fn)
{
	switch (a.log_bytes << 2 | a.dty) {
	case 0 << 2 | SMLP_DTY_CAT: return fn(a.i8);
	case 0 << 2 | SMLP_DTY_INT: return fn(a.i8);

	case 1 << 2 | SMLP_DTY_CAT: return fn(a.i16);
	case 1 << 2 | SMLP_DTY_INT: return fn(a.i16);

	case 2 << 2 | SMLP_DTY_CAT: return fn(a.i32);
	case 2 << 2 | SMLP_DTY_INT: return fn(a.i32);
	case 2 << 2 | SMLP_DTY_DBL: return fn(a.f32);

	case 3 << 2 | SMLP_DTY_CAT: return fn(a.i64);
	case 3 << 2 | SMLP_DTY_INT: return fn(a.i64);
	case 3 << 2 | SMLP_DTY_DBL: return fn(a.f64);
	}
	smlp_unreachable();
}

template <typename T> static inline int sgn(const T &x)
{
	return x < 0 ? -1 : x > 0 ? +1 : 0;
}

template <> inline int sgn(const int8_t &x) { return x; }
template <> inline int sgn(const int16_t &x) { return x; }
template <> inline int sgn(const int32_t &x) { return x; }
template <> inline int sgn(const float &x) { return std::signbit(x); }
template <> inline int sgn(const double &x) { return std::signbit(x); }

template <typename T> static inline int cmp(const T &a, const T &b)
{
	return a < b ? -1 : a > b ? +1 : 0;
}

template <> inline int cmp(const int8_t &a, const int8_t &b) { return sgn(a-b); }
template <> inline int cmp(const int16_t &a, const int16_t &b) { return sgn(a-b); }
template <> inline int cmp(const int32_t &a, const int32_t &b) { return sgn(a-b); }
template <> inline int cmp(const int64_t &a, const int64_t &b) { return sgn(a-b); }

template <typename Cols, typename Fn, typename Idcs>
static inline void
speced_group_by(const struct smlp_speced_csv *sp,
                const struct smlp_spec *spec,
                const Cols &by_cols,
                Fn &&fn, Idcs &idcs)
{
	using std::begin, std::end;

	auto group_cmp = [&](const size_t &a, const size_t &b){
		for (size_t j : by_cols) {
			if (int c = with(sp->cols[j],
			                 [&](auto p){ return cmp(p[a], p[b]); }))
				return c;
		}
		return 0;
	};

	auto group_eq = [&](const size_t &a, const size_t &b){
		for (size_t j : by_cols)
			if (with(sp->cols[j], [&](auto p){ return p[a] != p[b]; }))
				return false;
		return true;
	};

	std::sort(begin(idcs), end(idcs), cmp_lt(group_cmp));

	for (auto it = begin(idcs), jt = it; it != end(idcs); it = jt) {
		do ++jt; while (jt != end(idcs) && group_eq(*it, *jt));
		fn(it, jt);
	}
}

template <typename Cols, typename Fn, typename Idcs>
static inline void
speced_group_by(const struct smlp_speced_csv *sp,
                const struct smlp_spec *spec,
                const Cols &by_cols,
                Fn &&fn, Idcs &&idcs)
{
	speced_group_by(sp, spec, by_cols, std::forward<Fn>(fn), idcs);
}

template <typename Cols, typename Fn>
static inline auto
speced_group_by(const struct smlp_speced_csv *sp,
                const struct smlp_spec *spec,
                const Cols &by_cols,
                Fn &&fn)
{
	auto idcs = indices(sp->h);
	speced_group_by(sp, spec, by_cols, std::forward<Fn>(fn), idcs);
	return idcs;
}

struct specification : ::smlp_spec, detail::C_vec<::smlp_spec_entry, specification> {

	specification()
	: ::smlp_spec SMLP_SPEC_INIT
	, C_vec {}
	{}

	explicit specification(const kjson_value *v)
	: specification()
	{
		char *error = NULL;
		if (int r = smlp_spec_init(this, v, &error))
			throw std::runtime_error(r < 0 ? strerror(-r) : error);
	}

	explicit specification(const char *path)
	: specification()
	{
		char *error = NULL;
		if (int r = smlp_spec_init_path(this, path, &error))
			throw std::runtime_error(std::string(path) + ": " +
			                         (r < 0 ? strerror(-r) : error));
	}

	~specification()
	{
		smlp_spec_fini(this);
	}

	specification(specification &&o)
	: ::smlp_spec(o)
	, C_vec {}
	{
		o._data() = nullptr;
		o._size() = 0;
		o._delayed_cap = 0;
	}

	specification(const specification &) = delete;

	friend void swap(specification &a, specification &b)
	{
		using std::swap;
		swap(a._data(), b._data());
		swap(a._size(), b._size());
		swap(a._cap() , b._cap());
	}

	specification & operator=(specification o)
	{
		swap(*this, o);
		return *this;
	}

private:
	friend struct C_vec<::smlp_spec_entry, specification>;

	      ::smlp_spec_entry *      & _data()       { return ::smlp_spec::cols; }
	const ::smlp_spec_entry *const & _data() const { return ::smlp_spec::cols; }

	      size_t & _size()       { return ::smlp_spec::n; }
	const size_t & _size() const { return ::smlp_spec::n; }

	      size_t & _cap()        { using std::max; _delayed_cap = max(_delayed_cap, _size()); return _delayed_cap; }
	const size_t & _cap()  const { using std::max; _delayed_cap = max(_delayed_cap, _size()); return _delayed_cap; }

	mutable size_t _delayed_cap = 0;
};

/* owning semantics */
struct array : ::smlp_array {

	explicit array(::smlp_dtype dty)
	{
		smlp_array_init(this, dty);
	}

	array(::smlp_array &&o)
	: ::smlp_array(o)
	{
		o.v = nullptr;
	}

	~array()
	{
		smlp_array_fini(this);
	}

	array(array &&o)
	: ::smlp_array(o)
	{
		o.v = nullptr;
	}

	array(const array &o) = delete;

	friend void swap(array &a, array &b)
	{
		using std::swap;
		swap(a.header, b.header);
		swap(a.v, b.v);
	}

	array & operator=(array o)
	{
		swap(*this, o);
		return *this;
	}

	smlp_value get(size_t i) const { return smlp_array_get(this, i); }
};

struct speced_csv : ::smlp_speced_csv, private detail::C_vec<::smlp_array, speced_csv> {

	speced_csv()
	: ::smlp_speced_csv SMLP_SPECED_CSV_INIT
	, C_vec {}
	{}

	explicit speced_csv(FILE *f, specification &&spec)
	: speced_csv {}
	{
		if (int r = smlp_speced_init_csv(this, f, &spec))
			throw std::runtime_error("invalid spec'ed CSV");
		this->_spec = move(spec);
	}

	~speced_csv()
	{
		for (::smlp_array &a : *this)
			smlp_array_fini(&a);
		free(::smlp_speced_csv::cols);
	}

	friend void swap(speced_csv &a, speced_csv &b)
	{
		using std::swap;
		swap(static_cast<::smlp_speced_csv &>(a), static_cast<::smlp_speced_csv &>(b));
		swap(a._spec       , b._spec  );
		swap(a._delayed_cap, b._delayed_cap);
	}

	speced_csv(speced_csv &&o)
	: ::smlp_speced_csv(o)
	, C_vec {}
	, _spec(move(o._spec))
	, _delayed_cap(move(o._delayed_cap))
	{
		o.cols = nullptr;
	}

	speced_csv & operator=(speced_csv o)
	{
		swap(*this, o);
		return *this;
	}

	speced_csv(const speced_csv &) = delete;

	size_t add_column(const ::smlp_spec_entry &e)
	{
		return add_column(e, array(e.dtype));
	}

	size_t add_column(const ::smlp_spec_entry &e, array &&c)
	{
		assert(e.dtype == c.dty);
		size_t n = _size();
		reserve(n+1);
		::smlp_array *a = &(*this)[n];
		*a = move(c);
		smlp_array_resize(a, height_capacity());
		_spec.push_back(e);
		return n;
	}

	array drop_column(size_t i)
	{
		::smlp_array *a = &(*this)[i];
		size_t &n = _size();
		array r = move(*a);
		memmove(a, a+1, (n - (i+1)) * sizeof(*a));
		::smlp_spec_entry *e = &_spec[i];
		smlp_spec_entry_fini(e);
		memmove(e, e+1, (n - (i+1)) * sizeof(*e));
		n--;
		return r;
	}

	::smlp_value get(size_t i, size_t j) const
	{
		return smlp_speced_get(this, i, j);
	}

	void set(size_t i, size_t j, ::smlp_value v) const
	{
		smlp_speced_set(this, i, j, v);
	}

	size_t width () const { return _size(); }
	size_t height() const { return ::smlp_speced_csv::h; }

	size_t height_capacity() const { return ::smlp_speced_csv::sz; }

	void reserve_rows(size_t n)
	{
		smlp_speced_ensure_size(this, width(), n);
	}

	void resize_rows(size_t n)
	{
		reserve_rows(n);
		::smlp_speced_csv::h = n;
	}

	const specification & spec() const { return _spec; }
	const ::smlp_spec_entry & spec(size_t j) const { return _spec[j]; }

	const C_vec<::smlp_array, speced_csv> & columns() const { return *this; }

	const ::smlp_array & column(size_t j) const { return columns()[j]; }

	std::pair<const ::smlp_array &,const ::smlp_spec_entry &> col(size_t j) const
	{
		return { column(j), spec(j) };
	}

	ssize_t column_idx(std::string_view label) const
	{
		for (size_t j=0; j<width(); j++)
			if (_spec[j].label == label)
				return j;
		return -1;
	}

	template <typename Cols, typename Fn, typename Idcs>
	void group_by(const Cols &by_cols, Fn &&fn, Idcs &idcs) const
	{
		speced_group_by(this, &_spec, by_cols, std::forward<Fn>(fn), idcs);
	}

	template <typename Cols, typename Fn>
	std::vector<size_t> group_by(const Cols &by_cols, Fn &&fn) const
	{
		return speced_group_by(this, &_spec, by_cols, std::forward<Fn>(fn));
	}

	template <typename BackInsertIterator, typename Cols>
	void unique_value_rows(BackInsertIterator &&out, const Cols &cols) const
	{
		group_by(cols, [&](auto &rows, auto &rend) { *out = *rows; });
	}

	template <typename BackInsertIterator, typename Idcs>
	void unique_col_values(size_t j, BackInsertIterator &&out, Idcs &idcs) const
	{
		group_by(std::array { j },
		         [&](auto &rows, auto &rend)
		         { with(column(j), [&](auto ptr){ *out = ptr[*rows]; }); },
		         idcs);
	}

	template <typename BackInsertIterator>
	void unique_rows(BackInsertIterator &&out) const
	{
		unique_value_rows(std::forward<BackInsertIterator>(out),
		                  detail::range(width()));
	}

	void write_csv_header(FILE *out) const
	{
		smlp_speced_write_csv_header(this, &_spec, out);
	}

	void write_csv_row(FILE *out, size_t row) const
	{
		smlp_speced_write_csv_row(this, &_spec, row, out);
	}

private:
	specification _spec;

	friend struct C_vec<::smlp_array, speced_csv>;

	      ::smlp_array *      & _data()       { return ::smlp_speced_csv::cols; }
	const ::smlp_array *const & _data() const { return ::smlp_speced_csv::cols; }

	      size_t & _size()       { return _spec.n; }
	const size_t & _size() const { return _spec.n; }

	      size_t & _cap()        { using std::max; _delayed_cap = max(_delayed_cap, _size()); return _delayed_cap; }
	const size_t & _cap()  const { using std::max; _delayed_cap = max(_delayed_cap, _size()); return _delayed_cap; }

	mutable size_t _delayed_cap = 0;
};

static inline const char * to_str(::smlp_dtype dty) { return smlp_dtype_str(dty); }

template <typename Expr> struct speced_expr;

template <typename Sp>
struct speced_concat_rows {

	Sp sp;
	size_t n;

	auto   get(size_t i, size_t j)
	{
		size_t h = 0;
		for (size_t k=0; k<n; k++) {
			auto s = sp(k);
			if (i - h < s.height())
				return s.get(i - h, j);
			h += s.height();
		}
		return sp(-1).get(0, j);
	}
	// auto   col(size_t j) const { return sp(0).col(j); }
	size_t width  () const { return sp(0).width(); }
	size_t height () const
	{
		size_t h = 0;
		for (size_t k=0; k<n; k++)
			h += sp(k).height();
		return h;
	}
};

template <typename Expr, typename Cols>
struct speced_select_cols {

	speced_expr<Expr> base;
	Cols cols;

	auto   get(size_t i, size_t j) const { return base.get(i, cols[j]); }
	auto   col(size_t j) const { return base.col(cols[j]); }
	size_t width  () const { using std::size; return size(cols); }
	size_t height () const { return base.height(); }
};

template <typename Expr, typename Idcs>
struct speced_select_rows {

	speced_expr<Expr> base;
	Idcs idcs;

	auto   get(size_t i, size_t j) const { return base.get(idcs[i], j); }
	// auto   col(size_t j) const { return base.col(j); }
	size_t width  () const { return base.width(); }
	size_t height () const { using std::size; return size(idcs); }
};

}

#endif
