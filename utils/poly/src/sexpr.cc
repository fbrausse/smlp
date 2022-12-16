
#include <es/sexpr.hh>

using namespace es;

sexpr_parser::sexpr_parser(FILE *fin, size_t line_no, size_t col)
: fin(fin), p { line_no, col }
{ if (fin) get(); }

void sexpr_parser::get()
{
	c = getc(fin);
	switch (c) {
	case EOF:
		return;
	case '\n':
		p.line_no++;
		p.col = 0;
	}
	p.col++;
}

void sexpr_parser::skip_space()
{
	while (true)
		switch (c) {
		case '\t':
		case '\n':
		case '\r':
		case ' ':
			get();
			break;
		default:
			return;
		}
}

std::optional<slit> sexpr_parser::atom()
{
	std::string s;
	do {
		s.push_back(c);
		get();
	} while (c != EOF && !strchr("() \r\n\t", c));
	return { s };
}

std::optional<sexpr> sexpr_parser::compound()
{
	assert(c == '(');
	get();
	sexpr r;
	while (true) {
		skip_space();
		switch (c) {
		case EOF:
			fail("unexpected EOF in compound");
			return {};
		case '(':
			if (auto s = compound(); !s)
				return {};
			else
				r.emplace_back(*s);
			break;
		case ')':
			get();
			return { r };
		default:
			if (auto a = atom())
				r.emplace_back(*a);
			else
				return {};
			break;
		}
	}
}

std::optional<sexpr> sexpr_parser::next()
{
	assert(*this);
	skip_space();
	switch (c) {
	case EOF:
		return {};
	case '(':
		return compound();
	default:
		fail("expected '('");
		return {};
	}
}

static size_t len(const std::string &s, size_t tab_w = 8)
{
	size_t n = 0;
	for (char d : s)
		if (d == '\t')
			n = (n + tab_w) & -tab_w;
		else
			n++;
	return n;
}

static std::optional<size_t> single_line_width(const sexpr &a, size_t tab_w);

static std::optional<size_t> single_line_width(const arg &a, size_t tab_w)
{
	if (auto *s = std::get_if<slit>(&a)) {
		if (s->find("\n") != s->npos)
			return {};
		return { len(*s, tab_w) };
	} else
		return single_line_width(std::get<sexpr>(a), tab_w);
}

static std::optional<size_t> single_line_width(const sexpr &s, size_t tab_w)
{
	size_t w = 1; /* ( */
	for (const arg &a : s)
		if (auto s = single_line_width(a, tab_w))
			w += *s + 1;
		else
			return {};
	w += 1 - (empty(s) ? 0 : 1); /* ) */
	return { w };
}

static std::optional<size_t> first_line_width(const arg &a, size_t tab_w)
{
	if (auto *s = std::get_if<slit>(&a)) {
		if (s->find("\n") != s->npos)
			return {};
		return { len(*s, tab_w) };
	}
	const sexpr &e = std::get<sexpr>(a);
	if (empty(e))
		return { 2 };
	if (auto z = first_line_width(e[0], tab_w))
		return { 1 + *z };
	return {};
}

void formatter::putc(char c)
{
	::putc(c, f);
	if (c == '\t')
		pos = (pos + tab_w) & -tab_w;
	else if (c == '\n')
		pos = 0;
	else
		pos++;
}

bool formatter::fits(std::optional<size_t> w, ssize_t add) const
{
	return w && add + pos + *w <= width;
}

void formatter::emit(const arg &a, std::vector<size_t> &align)
{
	if (auto *s = std::get_if<slit>(&a)) {
		for (char c : *s)
			putc(c);
		if (!indent)
			indent = pos;
	} else
		emit(std::get<sexpr>(a), align);
}

void formatter::spaces(size_t n)
{
	while (n--)
		putc(' ');
}

void formatter::emit(const sexpr &e, std::vector<size_t> &align)
{
	bool f = fits(single_line_width(e, tab_w));
	putc('(');
	align.push_back(pos);
	if (f) {
		for (size_t i=0; i<size(e); i++) {
			if (i)
				putc(' ');
			emit(e[i], align);
		}
	} else {
		emit(e[0], align);
		align.back() = pos + 1;

		bool last_exception = false;
		for (size_t i=1; i<size(e); i++) {
			if ((fits((i == 1 && true) ? first_line_width(e[i], tab_w)
			                           : single_line_width(e[i], tab_w), 1) && !last_exception) ||
			    (!last_exception &&
			     (!fits(single_line_width(e[i], tab_w), align.back() - pos)
			      ? (last_exception = true), false : false))
			   ) {
				putc(' ');
				emit(e[i], align);
			} else {
				last_exception = true;
				putc('\n');
				spaces(align.back());
				emit(e[i], align);
			}
		}
	}
	align.pop_back();
	putc(')');
}

void formatter::emit(const arg &e, bool newline)
{
	std::vector<size_t> align = { pos };
	emit(e, align);
	if (newline) {
		putc('\n');
		indent = 0;
	}
}
