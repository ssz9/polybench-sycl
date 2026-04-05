PLATFORM ?= A10
REPODIR ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SYCLDIR ?= $(HOME)/sycl_workspace/llvm/build/install

VALID_PLATFORMS := CPU DCU SW MT3K A10
PLATFORM_UPPER := $(shell printf '%s' '$(PLATFORM)' | tr '[:lower:]' '[:upper:]')
PLATFORM_LOWER := $(shell printf '%s' '$(PLATFORM_UPPER)' | tr '[:upper:]' '[:lower:]')
ifeq ($(filter $(PLATFORM_UPPER),$(VALID_PLATFORMS)),)
$(error Unsupported PLATFORM '$(PLATFORM)'. Valid choices: $(VALID_PLATFORMS))
endif

CC := $(SYCLDIR)/bin/clang
CXX := $(SYCLDIR)/bin/clang++
ifeq ($(PLATFORM_UPPER),SW)
CC := $(HOME)/online/shenzt/swclcc/archives/5575c717571bb16ccf6799d6a9ae892ac15b9d5e/bin/swsycl
CXX := $(HOME)/online/shenzt/swclcc/archives/5575c717571bb16ccf6799d6a9ae892ac15b9d5e/bin/swsycl
endif

COMMON_CXXFLAGS := -O3 -fsycl -ffp-contract=off
PLATFORM_CXXFLAGS_CPU ?= -fsycl-targets=spir64_x86_64
PLATFORM_CXXFLAGS_A10 ?= -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86
PLATFORM_CXXFLAGS_DCU ?=
PLATFORM_CXXFLAGS_SW ?=
PLATFORM_CXXFLAGS_MT3K ?=

CXXFLAGS := $(COMMON_CXXFLAGS) $(PLATFORM_CXXFLAGS_$(PLATFORM_UPPER))
CPPFLAGS := -I$(REPODIR)/include -DPLF_$(PLATFORM_UPPER)

BINDIR := bin
OBJDIR := obj
OBJDIR_PLF := $(OBJDIR)/$(PLATFORM_LOWER)

$(OBJDIR_PLF)/%.o: polybench/*/%.cpp
	@mkdir -p $(OBJDIR_PLF)
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

$(OBJDIR_PLF)/%.o: open-earth-benchmarks/*/%.cpp
	@mkdir -p $(OBJDIR_PLF)
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

define BENCH_OBJS
$(patsubst polybench/$(1)/%.cpp,$(OBJDIR_PLF)/%.o,$(wildcard polybench/$(1)/$(1)_sycl_$(PLATFORM_LOWER)*.cpp)) \
$(patsubst open-earth-benchmarks/$(1)/%.cpp,$(OBJDIR_PLF)/%.o,$(wildcard open-earth-benchmarks/$(1)/$(1)_sycl_$(PLATFORM_LOWER)*.cpp))
endef
.SECONDEXPANSION:
$(BINDIR)/test_%: $(OBJDIR_PLF)/test_%.o $(OBJDIR_PLF)/%_serial.o $(OBJDIR_PLF)/%_sycl.o $$(call BENCH_OBJS,$$*)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^


.PHONY: all clean

# polybench
ALL_TESTS := \
	$(BINDIR)/test_correlation \
	$(BINDIR)/test_covariance \
	$(BINDIR)/test_gemm \
	$(BINDIR)/test_heat-3d \
	$(BINDIR)/test_jacobi-2d
# open-earth-benchmarks
ALL_TESTS += \
	$(BINDIR)/test_fastwavesuv \
	$(BINDIR)/test_hadvuv
all: $(ALL_TESTS)

clean:
	rm -rf $(BINDIR)/* $(OBJDIR)/*
