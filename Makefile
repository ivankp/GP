SHELL := bash
CXX := g++
CPPFLAGS := -std=c++14 -Iinclude
CXXFLAGS := -Wall -Ofast -flto -fmax-errors=3 $(CPPFLAGS)
# CXXFLAGS := -Wall -g -fmax-errors=3 $(CPPFLAGS)
LDFLAGS :=
LDLIBS :=

BLD := .build
EXT := .cc

VPATH = $(BLD)

.PHONY: all clean

ifeq (0, $(words $(findstring $(MAKECMDGOALS), clean)))

SRCS := $(shell find -L src -type f -name '*$(EXT)')
DEPS := $(patsubst src/%$(EXT),$(BLD)/%.d,$(SRCS))

GREP_EXES := grep -rl '^ *int \+main *(' src --include='*$(EXT)'
EXES := $(patsubst src%$(EXT),bin%, $(shell $(GREP_EXES)))

all: $(EXES)

-include $(DEPS)

# -------------------------------------------------------------------
bin/test_gp: linalg.o
bin/test_wls: linalg.o wls.o
bin/hgam_sandbox: linalg.o wls.o
bin/test_gp_opt: linalg.o

L_test_gp_opt := -lgsl -lgslcblas
L_gsl_min := -lgsl -lgslcblas
# -------------------------------------------------------------------

$(DEPS): $(BLD)/%.d: src/%$(EXT)
	@mkdir -pv $(dir $@)
	$(CXX) $(CPPFLAGS) $(C_$*) -MM -MT '$(@:.d=.o)' $< -MF $@

$(BLD)/%.o:
	@mkdir -pv $(dir $@)
	$(CXX) $(CXXFLAGS) $(C_$*) -c $(filter %$(EXT),$^) -o $@

bin/%: $(BLD)/%.o
	@mkdir -pv $(dir $@)
	$(CXX) $(LDFLAGS) $(filter %.o,$^) -o $@ $(LDLIBS) $(L_$*)

endif

clean:
	@rm -rfv $(BLD) bin

