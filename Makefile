PLATFORM = CUDA
REPODIR=/home/shenzitao/repos/polybench-sycl
SYCLDIR=/home/shenzitao/sycl_workspace/llvm/build/install

CC = ${SYCLDIR}/bin/clang
CXX= ${SYCLDIR}/bin/clang++

# CXXFLAGS = -O3 -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-targets=spir64_x86_64 		 		# for x86
CXXFLAGS = -O3 -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda  -Xsycl-target-backend=nvptx64-nvidia-cuda '-O3 --cuda-gpu-arch=sm_70'

CXXFLAGS += -ffp-contract=off # don't allow fp optimization e.g., FMA instr generation

CXXDEFS=
CXXFLAGS+= $(CXXDEFS) 

BINDIR = bin
OBJDIR = obj
INCLUDES = -I${REPODIR}/include

# polybench ======================================================================================================================================
${OBJDIR}/%_serial.o: polybench/*/%_serial.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<
${OBJDIR}/%_sycl.o: polybench/*/%_sycl.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<
${OBJDIR}/%_sycl_sw.o: polybench/*/%_sycl_sw.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<
${OBJDIR}/%_sycl_dcu.o: polybench/*/%_sycl_dcu.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<
${OBJDIR}/%_sycl_mt3k.o: polybench/*/%_sycl_mt3k.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<
${OBJDIR}/%_hip.o: polybench/*/%_hip.cpp
	echo "Not supported yet"
${OBJDIR}/%_athread.o: polybench/*/%_athread.cpp
	echo "Not supported yet"
${OBJDIR}/%_hthread.o: polybench/*/%_hthread.cpp
	echo "Not supported yet"
${OBJDIR}/test_%.o: polybench/*/test_%.cpp                                
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

${BINDIR}/test_%: ${OBJDIR}/test_%.o ${OBJDIR}/%_serial.o ${OBJDIR}/%_sycl.o #${OBJDIR}/%_sycl_sw.o ${OBJDIR}/%_sycl_dcu.o ${OBJDIR}/%_sycl_mt3k.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# open-earth-benchmarks ==========================================================================================================================

mkdir:
	mkdir -p $(BINDIR) $(OBJDIR)
.PHONY : mkdir

clean:
	rm -rf ./$(BINDIR)/* ./$(OBJDIR)/*
.PHONY : clean