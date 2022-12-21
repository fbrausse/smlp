
#ifndef IV_SAFE_HH
#define IV_SAFE_HH

#include <iv/safe.h>
#include <iv/ival.hh>
#include <smlp/table.h>

#include <stdexcept>	/* std::runtime_error */
#include <vector>
#include <sstream>
#include <iostream>	/* std::cerr */

namespace iv {

class File {

	FILE *f = nullptr;
public:
	explicit File(const char *path, const char *modes)
	: f(fopen(path, modes))
	{
		if (!f)
			throw std::system_error(errno, std::system_category(),
			                        path);
	}
	~File() { if (f) fclose(f); }
	File() = default;
	File(File &&o) : f(o.f) { o.f = nullptr; }

	File & operator=(File o) { using std::swap; swap(f, o.f); return *this; }

	operator FILE *() const noexcept { return f; }
};

struct table_exception : std::runtime_error {

	int table_ret;

	explicit table_exception(int r, std::string msg)
	: std::runtime_error(move(msg))
	, table_ret(r)
	{}
};

struct Table : ::smlp_table {

	Table() : ::smlp_table SMLP_TABLE_INIT {}

	Table(FILE *f, bool read_header)
	: Table()
	{
		int r = smlp_table_read(this, f, read_header);
		if (r) {
			smlp_table_fini(this);
			std::stringstream ss;
			ss << "read_table() returned error " << r;
			throw table_exception(r, ss.str());
		}
	}

	Table(Table &&t) : Table()
	{
		swap(*this, t);
	}

	~Table() { smlp_table_fini(this); }

	Table & operator=(Table o)
	{
		swap(*this, o);
		return *this;
	}

	friend void swap(Table &a, Table &b)
	{
		using std::swap;
		swap(static_cast<::smlp_table &>(a),
		     static_cast<::smlp_table &>(b));
	}

	struct row {
		const ::smlp_table &t;
		size_t i;

		row(const ::smlp_table &t, size_t i)
		: t(t)
		, i(i)
		{}

		float operator[](size_t j) const
		{
			return t.data[i*t.n_cols+j];
		}

		float operator[](const char *label) const
		{
			return t.data[i*t.n_cols+smlp_table_col_idx(&t, label)];
		}

		template <typename T>
		void get(const std::vector<size_t> &col_idcs, std::vector<T> &r) const
		{
			assert(size(col_idcs) == size(r));
			size_t k = 0;
			for (size_t j : col_idcs)
				r[k++] = T((*this)[j]);
		}
	};

	struct row_itr : row {
		using row::row;
		row_itr & operator++() { ++i; return *this; }
		const row & operator*() const { return *this; }
		const row * operator->() const { return this; }
		friend bool operator==(const row_itr &a, const row_itr &b)
		{
			return std::addressof(a.t) == std::addressof(b.t) &&
			       a.i == b.i;
		}
		friend bool operator!=(const row_itr &a, const row_itr &b)
		{
			return !(a == b);
		}
	};

	row_itr begin() const { return { *this, 0 }; }
	row_itr end  () const { return { *this, this->n_rows }; }
};

namespace detail {

struct forall_in_dataset_base {

	Table t;
	std::vector<size_t> dom_idcs;

	forall_in_dataset_base(const iv_model_fun &mf, FILE *dataset);
};

}

template <typename HRes>
struct forall_in_dataset : detail::forall_in_dataset_base {

	HRes handle_result;

	forall_in_dataset(const iv_model_fun &mf, FILE *dataset,
	                  HRes handle_result)
	: detail::forall_in_dataset_base(mf, dataset)
	, handle_result(handle_result)
	{}

	forall_in_dataset(const iv_model_fun &mf, const char *dataset_path,
	                  HRes handle_result)
	: forall_in_dataset(mf, dataset_path ? File(dataset_path, "r") : stdin,
	                    handle_result)
	{}

	template <typename G>
	void operator()(G g) const
	{
		std::vector<ival> v(size(dom_idcs));
		for (const Table::row &r : t) {
			r.get(dom_idcs, v);
			handle_result(g(v));
		}
	}
};

ival tf_eval_ival(const iv_target_function &tf, const std::vector<ival> &x);
ival tf_eval_ival(const iv_target_function &tf,       std::vector<ival> &&x);

}

#endif
