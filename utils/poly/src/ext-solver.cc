/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "ext-solver.hh"
#include "dump-smt2.hh"

#include <unistd.h>   /* pipe2(), dup2() */
#include <signal.h>   /* kill() */
#include <sys/wait.h> /* waitpid() */

using namespace smlp;

namespace {

struct Pipe {

	file rd, wr;

	explicit Pipe(int flags = 0)
	{
		int rw[2];
		if (pipe2(rw, flags))
			throw std::error_code(errno, std::system_category());
		rd = file(rw[0], "r");
		wr = file(rw[1], "w");
	}
};

}

process::process(const char *cmd)
{
	Pipe i, o/*, e*/;
	pid = fork();
	if (pid == -1)
		throw std::error_code(errno, std::system_category());
	if (!pid) {
		i.wr.close();
		o.rd.close();
		// e.rd.close();
		if (dup2(fileno(i.rd), STDIN_FILENO) == -1)
			throw std::error_code(errno, std::system_category());
		if (dup2(fileno(o.wr), STDOUT_FILENO) == -1)
			throw std::error_code(errno, std::system_category());
		/*
		if (dup2(fileno(e.wr), STDERR_FILENO) == -1)
			throw std::error_code(errno, std::system_category());*/
		const char *shell = getenv("SHELL");
		if (!shell)
			shell = "sh";
		execlp(shell, shell, "-c", cmd, NULL);
		throw std::error_code(errno, std::system_category());
	}
	i.rd.close();
	o.wr.close();
	// e.wr.close();
	in = move(i.wr);
	out = move(o.rd);
	// err = move(e.rd);
}

process::~process()
{
	if (pid == -1)
		return;
	in.close();
	out.close();
	// err.close();
	int status;
	waitpid(pid, &status, WNOHANG);
	if (!WIFEXITED(status)) {
		kill(pid, SIGTERM);
		waitpid(pid, &status, WUNTRACED);
	}
	log(mod_ext, WEXITSTATUS(status) ? WARN : NOTE,
	    "child %d exited with code %d\n", pid, WEXITSTATUS(status));
}

str ext_solver::get_info(const char *what)
{
	fprintf(in, "(get-info %s)\n", what);
	opt<es::sexpr> reply = out_s.next();
	assert(reply);
	assert(reply->size() == 2);
	assert(std::get<es::slit>((*reply)[0]) == what);
	const es::slit &s = std::get<es::slit>((*reply)[1]);
	assert(s.length() >= 2);
	assert(s[0] == '"');
	assert(s[s.length()-1] == '"');
	return s.substr(1, s.length() - 2);
}

ext_solver::ext_solver(const char *cmd, const char *logic)
: process(cmd)
, out_s((ungetc(' ', out), out))
{
	setvbuf(in, NULL, _IOLBF, 0);

	fprintf(in, "(set-option :print-success false)\n");

	name = get_info(":name");
	version = get_info(":version");
	std::string_view v = version;
	if (name == "MathSAT5" && v.starts_with("MathSAT5 version "))
		v = v.substr("MathSAT5 version "sv.length());
	else if (name == "ksmt" && v.starts_with("v"))
		v = v.substr(1);
	auto [_,ec] = from_chars(v.data(), v.data() + v.length(), parsed_version);
	assert(ec == std::errc {});
	note(mod_ext, "ext-solver pid %d: %s %s\n", pid, name.c_str(),
	     to_string(parsed_version).c_str());

	if (name != "ksmt")
		fprintf(in, "(set-option :produce-models true)\n");
	if (logic)
		fprintf(in, "(set-logic %s)\n", logic);
}

void ext_solver::declare(const domain &d)
{
	assert(!n_vars);
	dump_smt2(in, d);
	n_vars = size(d);
}

static bool matches(const es::arg &a, const std::string_view &v)
{
	using es::slit;
	if (const slit *s = std::get_if<slit>(&a))
		return *s == v;
	return false;
}

static kay::Q Q_from_smt2(const es::arg &s)
{
	using namespace kay;
	using es::slit;
	using es::sexpr;

	if (const slit *sls = std::get_if<slit>(&s))
		return Q_from_str(str(*sls).data());
	const sexpr &se = std::get<sexpr>(s);
	if (size(se) == 2 && matches(se[0], "to_real"))
		return Q_from_smt2(se[1]);
	if (size(se) == 2 && matches(se[0], "-"))
		return -Q_from_smt2(se[1]);
	if (size(se) == 2 && matches(se[0], "+"))
		return +Q_from_smt2(se[1]);
	assert(size(se) == 3);
	assert(matches(se[0], "/"));
	return Q_from_smt2(se[1]) / Q_from_smt2(se[2]);
}

static pair<str,sptr<term2>> parse_smt2_asgn(const es::sexpr &a)
{
	using es::slit;
	using es::arg;
	using es::sexpr;

	str v, t;
	arg b = slit("");
	if (size(a) == 2) {
		/* (name cnst) [MathSAT 5.6.3] */
		const auto &[var,s] = as_tuple_ex<slit,arg>(a);
		v = var;
		t = "Real"; /* dummy, unused */
		b = s;
	} else if (size(a) == 3) {
		/* (= name cnst) [Yices 2.6.1] */
		const auto &[eq,var,s] = as_tuple_ex<slit,slit,arg>(a);
		assert(eq == "=");
		v = var;
		t = "Real"; /* dummy, unused */
		b = s;
	} else if (size(a) == 4) {
		/* (define-const name type cnst) */
		const auto &[def,var,ty,s] = as_tuple_ex<slit,slit,slit,arg>(a);
		assert(def == "define-const");
		v = var;
		t = ty;
		b = s;
	} else {
		/* (define-fun name () type cnst) */
		assert(size(a) == 5);
		const auto &[def,var,none,ty,s] =
			as_tuple_ex<slit,slit,sexpr,slit,arg>(a);
		assert(def == "define-fun");
		assert(size(none) == 0);
		v = var;
		t = ty;
		b = s;
	}

	assert(t == "Real" || t == "Int");
	return { v, make2t(cnst2 { Q_from_smt2(b) }) };
}

result ext_solver::check()
{
	using es::slit;
	using es::arg;
	using es::sexpr;

	info(mod_ext,"solving...\n");

	fprintf(in, "(check-sat)\n");
	out_s.skip_space();
	if (out_s.c == '(') {
		opt<sexpr> e = out_s.compound();
		assert(e);
		es::formatter f;
		f.f = stderr;
		f.emit(*e);
		abort();
	}

	opt<slit> res = out_s.atom();
	assert(res);
	if (*res == "unsat")
		return unsat {};
	if (*res == "unknown")
		return unknown { get_info(":reason-unknown") };
	assert(*res == "sat");

	fprintf(in, "(get-model)\n");
	hmap<str,sptr<term2>> m;
	if (name == "Yices") {
		for (size_t i=0; i<n_vars; i++) {
			opt<sexpr> n = out_s.next();
			assert(n);
			assert(size(*n) == 3);
			auto [it,ins] = m.insert(parse_smt2_asgn(*n));
			assert(ins);
		}
		return sat { move(m) };
	}
	opt<sexpr> no = out_s.next();
	assert(no);
	const sexpr &n = *no;
	size_t off = 0;
	if (name == "cvc4" || name == "ksmt" ||
	    (name == "MathSAT5" && parsed_version >= split_version {5,6,8})) {
		assert(std::get<slit>(n[0]) == "model");
		off = 1;
	}
	assert(size(n) == off+n_vars);
	for (size_t i=0; i<n_vars; i++) {
		auto [it,ins] = m.insert(parse_smt2_asgn(std::get<sexpr>(n[off+i])));
		assert(ins);
	}
	return sat { move(m) };
}

void ext_solver::add(const sptr<form2> &f)
{
	fprintf(in, "(assert ");
	dump_smt2(in, *f);
	fprintf(in, ")\n");
}
