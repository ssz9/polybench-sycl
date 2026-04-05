PLATFORM ?= A10
REPODIR := /home/shenzitao/repos/polybench-sycl
SYCLDIR := /home/shenzitao/sycl_workspace/llvm/build/install

CC := $(SYCLDIR)/bin/clang
CXX := $(SYCLDIR)/bin/clang++

VALID_PLATFORMS := CPU DCU SW MT3K A10
PLATFORM_UPPER := $(shell printf '%s' '$(PLATFORM)' | tr '[:lower:]' '[:upper:]')
PLATFORM_LOWER := $(shell printf '%s' '$(PLATFORM_UPPER)' | tr '[:upper:]' '[:lower:]')

ifeq ($(filter $(PLATFORM_UPPER),$(VALID_PLATFORMS)),)
$(error Unsupported PLATFORM '$(PLATFORM)'. Valid choices: $(VALID_PLATFORMS))
endif

COMMON_CXXFLAGS := -O3 -fsycl -fno-sycl-id-queries-fit-in-int -ffp-contract=off
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

define BENCH_OBJS
$(patsubst polybench/$(1)/%.cpp,$(OBJDIR_PLF)/%.o,$(wildcard polybench/$(1)/$(1)_sycl_$(PLATFORM_LOWER)*.cpp))
endef
.SECONDEXPANSION:
$(BINDIR)/test_%: $(OBJDIR_PLF)/test_%.o $(OBJDIR_PLF)/%_serial.o $(OBJDIR_PLF)/%_sycl.o $$(call BENCH_OBJS,$$*)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^


.PHONY: all clean

ALL_TESTS := \
	$(BINDIR)/test_correlation \
	$(BINDIR)/test_covariance \
	$(BINDIR)/test_gemm \
	$(BINDIR)/test_heat-3d \
	$(BINDIR)/test_jacobi-2d
all: $(ALL_TESTS)

clean:
	rm -rf $(BINDIR)/* $(OBJDIR)/*
