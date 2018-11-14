source = gen_lognormal.cpp
objects = 
executables = gen_lognormal
CXX = g++
LANGFLAGS = -std=c++14
CXXFLAGS = -Ofast $(LANGFLAGS) -pedantic -Wall -Wextra -Weffc++
LIBS =

all: $(executables)

gen_lognormal: gen_lognormal.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	-rm -f $(executables) *.o *~
