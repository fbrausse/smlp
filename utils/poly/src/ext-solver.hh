/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "solver.hh"

#include "../include/smtlib2-parser.hh"

namespace smlp {

struct process {

	file in, out, err;
	pid_t pid = -1;

	explicit process(const char *cmd);
	~process();
};

struct ext_solver : process, solver {

	explicit ext_solver(const char *cmd, const char *logic = nullptr);
	void declare(const domain &d) override;
	void add(const form2 &f) override;
	result check() override;

private:
	es::smtlib2::parser out_s;
	str name, version;
	size_t n_vars = 0;

	str get_info(const char *what);
};

}
