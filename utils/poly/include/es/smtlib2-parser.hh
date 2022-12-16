
#ifndef ES_SMTLIB2_PARSER_HH
#define ES_SMTLIB2_PARSER_HH

#include "sexpr.hh"

namespace es::smtlib2 {

struct parser : sexpr_parser {

	parser(FILE *fin = stdin, size_t line_no = 1, size_t col = 0)
	: sexpr_parser(fin, line_no, col)
	{ _get(); }

	void _get()
	{
		if (c == ';') {
			/* comment */
			do sexpr_parser::get(); while (c != '\n');
		}
	}

	void get() override
	{
		sexpr_parser::get();
		_get();
	}

	std::optional<slit> str()
	{
		std::string s;
		assert(c == '"');
		s.push_back(c);
		while (true) {
			sexpr_parser::get();
			s.push_back(c);
			if (c == EOF) {
				fail("unexpected EOF in string");
				return {};
			}
			if (c == '"') {
				get();
				if (c != '"')
					return { s };
				s.push_back(c);
			}
		}
	}

	std::optional<slit> quot()
	{
		std::string s;
		assert(c == '|');
		do {
			s.push_back(c);
			sexpr_parser::get();
			if (c == EOF) {
				fail("unexpected EOF in quoted symbol");
				return {};
			}
		} while (c != '|');
		s.push_back(c);
		sexpr_parser::get();
		return { s };
	}

	std::optional<slit> atom() override
	{
		switch (c) {
		case '"': return str();
		case '|': return quot();
		default: return sexpr_parser::atom();
		}
	}
};

}

#endif
