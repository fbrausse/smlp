
CXXSTD = -std=c++20
WARNS  = -Wall -Wextra
OPT    = -O0 -g
#OPT    = -O

poly: poly.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)
poly: override LDLIBS += -lz3 -lgmp

poly.o: poly.cc Makefile
poly.o: override CXXFLAGS += $(CXXSTD) $(WARNS) $(OPT) -MMD
poly.o: override CPPFLAGS += -I ~/stovokor/kay/include -DKAY_USE_GMPXX

clean:
	$(RM) poly poly.o poly.d

-include $(wildcard *.d)
