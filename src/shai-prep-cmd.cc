
#define _POSIX_C_SOURCE 200809L

#include <cstdio>
#include <cstdlib>

#include <unistd.h>	/* getopt */
#include <libgen.h>	/* basename */

#define DEFAULT_TIMING	"Timing"
#define DEFAULT_DELTA	"delta"

#include "shai-prep.cc"

static const char *progname = "shai-prep";

static void usage(void)
{
	printf("\
usage: %s [-OPTS] -s IN.spec\n\
\n\
Options [defaults]:\n\
  -1 {rx|tx}   prepare single-objective version for RX or TX [multi-objective]\n\
  -D DELTA     name of the 'delta' feature in IN.csv [" DEFAULT_DELTA "]\n\
  -h           display this help message\n\
  -i IN.csv    read dataset from IN.csv [stdin]\n" /*
  -j N         use N parallel threads to process the dataset [none]\n\
  -m           enable compatibility to gmake's jobserver for parallelism [false]\n" */ "\
  -o OUT.csv   write processed dataset to OUT.csv [stdout]\n\
  -q           quiet mode, suppress all non-error outputs [false]\n\
  -s IN.spec   read .spec from IN.spec\n\
  -t OUT.spec  write processed .spec to OUT.spec\n\
  -T TIMING    name of the 'time' feature in IN.csv [" DEFAULT_TIMING "]\n\
  -v           increase verbosity [1]\n\
", progname);
	exit(0);
}

#define DIE(code, ...) do { fprintf(stderr,__VA_ARGS__); exit(code); } while (0)

int main(int argc, char **argv)
{
	smlp_mrc_shai_params par = {
		.csv_in_path = NULL,
		.spec_in_path = NULL,
		.spec_out_path = NULL,
		.timing_lbl = DEFAULT_TIMING,
		.delta_lbl = DEFAULT_DELTA,
		.v1 = NULL,
	};
	const char *csv_out_path = NULL;/*
	bool use_jobserver = false;
	int jobs = 0;*/
	bool quiet = false;

	progname = basename(argv[0]);
	verbosity = 1;

	for (int opt; (opt = getopt(argc, argv, ":1:D:hi:"/*j:m"*/"o:qs:t:T:v")) != -1;)
		switch (opt) {
		case '1': par.v1 = optarg; break;
		case 'D': par.delta_lbl = optarg; break;
		case 'h': usage();
		case 'i': par.csv_in_path = optarg; break;/*
		case 'j': jobs = atoi(optarg); break;
		case 'm': use_jobserver = true; break;*/
		case 'o': csv_out_path = optarg; break;
		case 'q': quiet = true; break;
		case 's': par.spec_in_path = optarg; break;
		case 't': par.spec_out_path = optarg; break;
		case 'T': par.timing_lbl = optarg; break;
		case 'v': verbosity++; break;
		case ':': DIE(1,"error: option '-%c' requires a parameter",optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}

	if (quiet)
		verbosity = 0;

	if (optind < argc) {
		fprintf(stderr, "error: unknown trailing options:");
		for (; optind < argc; optind++)
			fprintf(stderr, " %s", argv[optind]);
		DIE(1,"\n");
	}

	smlp_mrc_shai sh;
	int r = smlp_mrc_shai_prep_init(&sh, &par, SMLP_MRC_SHAI_PREP_NO_PY);
	if (r < 0)
		DIE(-r,"error: %s\n", strerror(-r));
	if (r > 0)
		DIE(r,"error: %s\n", sh.error);

	FILE *out = csv_out_path ? fopen(csv_out_path, "w") : stdout;
	if (!out)
		DIE(1,"error: %s: %s\n",csv_out_path,strerror(errno));

	sh.shai->write_csv_header(out);
	for (auto diam : { 70, 80, 90, 100, 110, 120, 130 }) {
		smlp_mrc_shai_prep_rad(&sh, diam / 2);
		for (size_t i : sh.shai->index)
			sh.shai->write_csv_row(out, i);
	}
	fclose(out);

	smlp_mrc_shai_prep_fini(&sh);

	return 0;
}
