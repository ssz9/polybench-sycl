#include "hadvuv.h"
#include "sw_sycl_utils.h"

static void kernel_hadvuv_sycl_sw(
    buffer<DATA_TYPE, 3> &uout_buf, buffer<DATA_TYPE, 3> &vout_buf, buffer<DATA_TYPE, 3> &uin_buf, buffer<DATA_TYPE, 3> &vin_buf,
    buffer<DATA_TYPE, 1> &acrlat0_buf, buffer<DATA_TYPE, 1> &acrlat1_buf, buffer<DATA_TYPE, 1> &tgrlatda0_buf, buffer<DATA_TYPE, 1> &tgrlatda1_buf,
    buffer<DATA_TYPE, 3> &uatupos_buf, buffer<DATA_TYPE, 3> &vatupos_buf, buffer<DATA_TYPE, 3> &uatvpos_buf, buffer<DATA_TYPE, 3> &vatvpos_buf,
    buffer<DATA_TYPE, 3> &uavg_buf, buffer<DATA_TYPE, 3> &vavg_buf, buffer<DATA_TYPE, 3> &ures_buf, buffer<DATA_TYPE, 3> &vres_buf,
    DATA_TYPE eddlat, DATA_TYPE eddlon, queue &Q)
{
  Q.submit([&](handler &h) {
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto uatupos = uatupos_buf.get_access<access::mode::write>(h);
    auto vatupos = vatupos_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwUatuposKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          uatupos[ii][jj][kk] = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (uin[ii - 1][jj][kk] + uin[ii][jj][kk] + uin[ii + 1][jj][kk]);
          vatupos[ii][jj][kk] = DATA_TYPE(0.25) * (vin[ii + 1][jj][kk] + vin[ii + 1][jj - 1][kk] + vin[ii][jj][kk] + vin[ii][jj - 1][kk]);
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto uatupos = uatupos_buf.get_access<access::mode::read>(h);
    auto vatupos = vatupos_buf.get_access<access::mode::read>(h);
    auto acrlat0 = acrlat0_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::write>(h);
    auto vavg = vavg_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwUavg0Kernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          uavg[ii][jj][kk] = acrlat0[jj] * uatupos[ii][jj][kk];
          vavg[ii][jj][kk] = EARTH_RADIUS_RECIP * vatupos[ii][jj][kk];
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::read>(h);
    auto vavg = vavg_buf.get_access<access::mode::read>(h);
    auto ures = ures_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwUresKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          ures[ii][jj][kk] = advection_driver_sycl(uin, i, j, k, uavg[ii][jj][kk], vavg[ii][jj][kk], eddlat, eddlon);
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto ures = ures_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto vatupos = vatupos_buf.get_access<access::mode::read>(h);
    auto tgrlatda0 = tgrlatda0_buf.get_access<access::mode::read>(h);
    auto uout = uout_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwUoutKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          uout[ii][jj][kk] = ures[ii][jj][kk] + tgrlatda0[jj] * uin[ii][jj][kk] * vatupos[ii][jj][kk];
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto uatvpos = uatvpos_buf.get_access<access::mode::write>(h);
    auto vatvpos = vatvpos_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwUatvposKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          uatvpos[ii][jj][kk] = DATA_TYPE(0.25) * (uin[ii - 1][jj][kk] + uin[ii][jj][kk] + uin[ii][jj + 1][kk] + uin[ii - 1][jj + 1][kk]);
          vatvpos[ii][jj][kk] = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (vin[ii][jj - 1][kk] + vin[ii][jj][kk] + vin[ii][jj + 1][kk]);
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto uatvpos = uatvpos_buf.get_access<access::mode::read>(h);
    auto vatvpos = vatvpos_buf.get_access<access::mode::read>(h);
    auto acrlat1 = acrlat1_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::write>(h);
    auto vavg = vavg_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwUavg1Kernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          uavg[ii][jj][kk] = acrlat1[jj] * uatvpos[ii][jj][kk];
          vavg[ii][jj][kk] = EARTH_RADIUS_RECIP * vatvpos[ii][jj][kk];
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::read>(h);
    auto vavg = vavg_buf.get_access<access::mode::read>(h);
    auto vres = vres_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwVresKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          vres[ii][jj][kk] = advection_driver_sycl(vin, i, j, k, uavg[ii][jj][kk], vavg[ii][jj][kk], eddlat, eddlon);
        }
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto vres = vres_buf.get_access<access::mode::read>(h);
    auto uatvpos = uatvpos_buf.get_access<access::mode::read>(h);
    auto tgrlatda1 = tgrlatda1_buf.get_access<access::mode::read>(h);
    auto vout = vout_buf.get_access<access::mode::write>(h);
    h.parallel_for<class HadvuvSwVoutKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          vout[ii][jj][kk] = vres[ii][jj][kk] - tgrlatda1[jj] * uatvpos[ii][jj][kk] * uatvpos[ii][jj][kk];
        }
      }
    });
  }).wait();
}

void hadvuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  queue Q;
  {
    range<3> r3(DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_HEIGHT + 2 * HALO_WIDTH);
    range<1> r1(DOMAIN_SIZE + 2 * HALO_WIDTH);
    buffer<DATA_TYPE, 3> uout_buf(uout, r3);
    buffer<DATA_TYPE, 3> vout_buf(vout, r3);
    buffer<DATA_TYPE, 3> uin_buf(uin, r3);
    buffer<DATA_TYPE, 3> vin_buf(vin, r3);
    buffer<DATA_TYPE, 1> acrlat0_buf(acrlat0, r1);
    buffer<DATA_TYPE, 1> acrlat1_buf(acrlat1, r1);
    buffer<DATA_TYPE, 1> tgrlatda0_buf(tgrlatda0, r1);
    buffer<DATA_TYPE, 1> tgrlatda1_buf(tgrlatda1, r1);
    buffer<DATA_TYPE, 3> uatupos_buf(uatupos, r3);
    buffer<DATA_TYPE, 3> vatupos_buf(vatupos, r3);
    buffer<DATA_TYPE, 3> uatvpos_buf(uatvpos, r3);
    buffer<DATA_TYPE, 3> vatvpos_buf(vatvpos, r3);
    buffer<DATA_TYPE, 3> uavg_buf(uavg, r3);
    buffer<DATA_TYPE, 3> vavg_buf(vavg, r3);
    buffer<DATA_TYPE, 3> ures_buf(ures, r3);
    buffer<DATA_TYPE, 3> vres_buf(vres, r3);
    kernel_hadvuv_sycl_sw(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                          uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);
  }
}

void bench_hadvuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  queue Q;
  {
    range<3> r3(DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_HEIGHT + 2 * HALO_WIDTH);
    range<1> r1(DOMAIN_SIZE + 2 * HALO_WIDTH);
    buffer<DATA_TYPE, 3> uout_buf(uout, r3);
    buffer<DATA_TYPE, 3> vout_buf(vout, r3);
    buffer<DATA_TYPE, 3> uin_buf(uin, r3);
    buffer<DATA_TYPE, 3> vin_buf(vin, r3);
    buffer<DATA_TYPE, 1> acrlat0_buf(acrlat0, r1);
    buffer<DATA_TYPE, 1> acrlat1_buf(acrlat1, r1);
    buffer<DATA_TYPE, 1> tgrlatda0_buf(tgrlatda0, r1);
    buffer<DATA_TYPE, 1> tgrlatda1_buf(tgrlatda1, r1);
    buffer<DATA_TYPE, 3> uatupos_buf(uatupos, r3);
    buffer<DATA_TYPE, 3> vatupos_buf(vatupos, r3);
    buffer<DATA_TYPE, 3> uatvpos_buf(uatvpos, r3);
    buffer<DATA_TYPE, 3> vatvpos_buf(vatvpos, r3);
    buffer<DATA_TYPE, 3> uavg_buf(uavg, r3);
    buffer<DATA_TYPE, 3> vavg_buf(vavg, r3);
    buffer<DATA_TYPE, 3> ures_buf(ures, r3);
    buffer<DATA_TYPE, 3> vres_buf(vres, r3);

    for (int i = 0; i < WARMUP_REPS; ++i)
      kernel_hadvuv_sycl_sw(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                            uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);

    TIMEIT({
      kernel_hadvuv_sycl_sw(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                            uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);
    }, BENCH_REPS, "\n", "sycl-sw");
  }
}
