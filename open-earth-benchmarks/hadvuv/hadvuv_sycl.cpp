#include "hadvuv.h"

void kernel_hadvuv_sycl(
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

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      uatupos[IDXSY(i, j, k)] = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (uin[IDXSY(i - 1, j, k)] + uin[IDXSY(i, j, k)] + uin[IDXSY(i + 1, j, k)]);
      vatupos[IDXSY(i, j, k)] = DATA_TYPE(0.25) * (vin[IDXSY(i + 1, j, k)] + vin[IDXSY(i + 1, j - 1, k)] + vin[IDXSY(i, j, k)] + vin[IDXSY(i, j - 1, k)]);
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto uatupos = uatupos_buf.get_access<access::mode::read>(h);
    auto vatupos = vatupos_buf.get_access<access::mode::read>(h);
    auto acrlat0 = acrlat0_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::write>(h);
    auto vavg = vavg_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      uavg[IDXSY(i, j, k)] = acrlat0[IDX_1D(j)] * uatupos[IDXSY(i, j, k)];
      vavg[IDXSY(i, j, k)] = EARTH_RADIUS_RECIP * vatupos[IDXSY(i, j, k)];
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::read>(h);
    auto vavg = vavg_buf.get_access<access::mode::read>(h);
    auto ures = ures_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      ures[IDXSY(i, j, k)] = advection_driver_sycl(uin, i, j, k, uavg[IDXSY(i, j, k)], vavg[IDXSY(i, j, k)], eddlat, eddlon);
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto ures = ures_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto vatupos = vatupos_buf.get_access<access::mode::read>(h);
    auto tgrlatda0 = tgrlatda0_buf.get_access<access::mode::read>(h);
    auto uout = uout_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      uout[IDXSY(i, j, k)] = ures[IDXSY(i, j, k)] + tgrlatda0[IDX_1D(j)] * uin[IDXSY(i, j, k)] * vatupos[IDXSY(i, j, k)];
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto uatvpos = uatvpos_buf.get_access<access::mode::write>(h);
    auto vatvpos = vatvpos_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      uatvpos[IDXSY(i, j, k)] = DATA_TYPE(0.25) * (uin[IDXSY(i - 1, j, k)] + uin[IDXSY(i, j, k)] + uin[IDXSY(i, j + 1, k)] + uin[IDXSY(i - 1, j + 1, k)]);
      vatvpos[IDXSY(i, j, k)] = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (vin[IDXSY(i, j - 1, k)] + vin[IDXSY(i, j, k)] + vin[IDXSY(i, j + 1, k)]);
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto uatvpos = uatvpos_buf.get_access<access::mode::read>(h);
    auto vatvpos = vatvpos_buf.get_access<access::mode::read>(h);
    auto acrlat1 = acrlat1_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::write>(h);
    auto vavg = vavg_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      uavg[IDXSY(i, j, k)] = acrlat1[IDX_1D(j)] * uatvpos[IDXSY(i, j, k)];
      vavg[IDXSY(i, j, k)] = EARTH_RADIUS_RECIP * vatvpos[IDXSY(i, j, k)];
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto uavg = uavg_buf.get_access<access::mode::read>(h);
    auto vavg = vavg_buf.get_access<access::mode::read>(h);
    auto vres = vres_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      vres[IDXSY(i, j, k)] = advection_driver_sycl(vin, i, j, k, uavg[IDXSY(i, j, k)], vavg[IDXSY(i, j, k)], eddlat, eddlon);
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto vres = vres_buf.get_access<access::mode::read>(h);
    auto uatvpos = uatvpos_buf.get_access<access::mode::read>(h);
    auto tgrlatda1 = tgrlatda1_buf.get_access<access::mode::read>(h);
    auto vout = vout_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<3>(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_HEIGHT), [=](id<3> idx) {
      ssize_t i = idx[0], j = idx[1], k = idx[2];
      vout[IDXSY(i, j, k)] = vres[IDXSY(i, j, k)] - tgrlatda1[IDX_1D(j)] * uatvpos[IDXSY(i, j, k)] * uatvpos[IDXSY(i, j, k)];
    });
  }).wait();
}

void hadvuv_sycl(
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
    kernel_hadvuv_sycl(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                       uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);
  }
}

void bench_hadvuv_sycl(
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
      kernel_hadvuv_sycl(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                         uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);

    TIMEIT({
      kernel_hadvuv_sycl(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                         uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);
    }, BENCH_REPS, "\n", "sycl-naive");
  }
}
