
#include "common.hh"
#include "solver.hh"

#define CSI		"\x1b["
#define SGR_DFL		CSI "m"
#define SGR_BOLD	CSI "1m"
#define COL_FG		"3"
#define COL_BG		"4"
#define COL_FG_B	"9"
#define COL_BG_B	"10"
#define COL_BLACK	"0"
#define COL_RED		"1"
#define COL_GREEN	"2"
#define COL_YELLOW	"3"
#define COL_BLUE	"4"
#define COL_MAGENTA	"5"
#define COL_CYAN	"6"
#define COL_WHITE	"7"

using namespace smlp;

hmap<strview,Module *> Module::modules;

bool Module::log_color = isatty(STDERR_FILENO);

Module::Module(const char *name, const char *color, loglvl lvl)
: name(name)
, color(color)
, lvl(lvl)
{
	auto [it,ins] = modules.emplace(name, this);
	assert(ins);
}

bool Module::vlog(loglvl l, const char *fmt, va_list ap) const
{
	if (!logs(l))
		return false;
	const char *lvl = nullptr;
	const char *col = "";
	switch (l) {
	case QUIET: break;
	case ERROR: lvl = "error"; col = CSI COL_FG_B COL_RED "m"; break;
	case WARN : lvl = "warn" ; col = CSI COL_FG_B COL_YELLOW "m"; break;
	case INFO : lvl = "info" ; col = CSI COL_FG_B COL_WHITE "m"; break;
	case NOTE : lvl = "note" ; break;
	case DEBUG: lvl = "debug"; col = CSI COL_FG   COL_GREEN "m"; break;
	}
	fprintf(stderr, "%s[%-4s]%s %s%-5s%s: ",
	        log_color ? color : "", name, log_color ? SGR_DFL : "",
	        log_color ? col : "", lvl, log_color ? SGR_DFL : "");
	vfprintf(stderr, fmt, ap);
	return true;
}

Module smlp::mod_cand { "cand",          CSI COL_FG   COL_GREEN   "m" };
Module smlp::mod_coex { "coex",          CSI COL_FG   COL_RED     "m" };
Module smlp::mod_smlp { "smlp",                                       };
Module smlp::mod_prob { "prob", SGR_BOLD CSI COL_FG_B COL_BLACK   "m" };
Module smlp::mod_ival { "ival",          CSI COL_FG   COL_YELLOW  "m" };
Module smlp::mod_crit { "crit",          CSI COL_FG   COL_MAGENTA "m" };
Module smlp::mod_z3   { "z3"  ,          CSI COL_FG_B COL_BLUE    "m" };
Module smlp::mod_ext  { "ext" ,          CSI COL_FG   COL_CYAN    "m" };
Module smlp::mod_nn   { "nn"  ,          CSI COL_FG   COL_BLUE    "m" };
Module smlp::mod_poly { "poly",          CSI COL_FG   COL_BLUE    "m" };

int solver::alg_dec_prec_approx = 10;
opt<str> smlp::ext_solver_cmd;
opt<str> smlp::inc_solver_cmd;
long  smlp::intervals = -1;