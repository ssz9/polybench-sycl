#include "hadvuv.h"

namespace {

constexpr size_t WG_SIZE = 256;
constexpr size_t CELLS = size_t(DOMAIN_SIZE) * size_t(DOMAIN_SIZE) * size_t(DOMAIN_HEIGHT);

size_t round_up(size_t value, size_t factor)
{
  return ((value + factor - 1) / factor) * factor;
}

inline void decode_index(size_t linear, size_t &i, size_t &j, size_t &k)
{
  i = linear / (size_t(DOMAIN_SIZE) * size_t(DOMAIN_HEIGHT));
  const size_t rem = linear - i * size_t(DOMAIN_SIZE) * size_t(DOMAIN_HEIGHT);
  j = rem / size_t(DOMAIN_HEIGHT);
  k = rem - j * size_t(DOMAIN_HEIGHT);
}

void kernel_hadvuv_u_sycl_dcu(
    buffer<DATA_TYPE, 3> &uout_buf, buffer<DATA_TYPE, 3> &uin_buf, buffer<DATA_TYPE, 3> &vin_buf,
    buffer<DATA_TYPE, 1> &acrlat0_buf, buffer<DATA_TYPE, 1> &tgrlatda0_buf,
    DATA_TYPE eddlat, DATA_TYPE eddlon, queue &Q)
{
  const range<1> local_range{WG_SIZE};
  const range<1> global_range{round_up(CELLS, WG_SIZE)};
  Q.submit([&](handler &h) {
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto acrlat0 = acrlat0_buf.get_access<access::mode::read>(h);
    auto tgrlatda0 = tgrlatda0_buf.get_access<access::mode::read>(h);
    auto uout = uout_buf.get_access<access::mode::write>(h);
    h.parallel_for(nd_range<1>(global_range, local_range), [=](nd_item<1> item) {
      const size_t linear = item.get_global_id(0);
      if (linear < CELLS) {
        size_t i, j, k;
        decode_index(linear, i, j, k);
        const DATA_TYPE uat = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (uin[IDXSY(i - 1, j, k)] + uin[IDXSY(i, j, k)] + uin[IDXSY(i + 1, j, k)]);
        const DATA_TYPE vat = DATA_TYPE(0.25) * (vin[IDXSY(i + 1, j, k)] + vin[IDXSY(i + 1, j - 1, k)] + vin[IDXSY(i, j, k)] + vin[IDXSY(i, j - 1, k)]);
        const DATA_TYPE uavg_v = acrlat0[IDX_1D(j)] * uat;
        const DATA_TYPE vavg_v = EARTH_RADIUS_RECIP * vat;
        const DATA_TYPE ures_v = advection_driver_sycl(uin, i, j, k, uavg_v, vavg_v, eddlat, eddlon);
        uout[IDXSY(i, j, k)] = ures_v + tgrlatda0[IDX_1D(j)] * uin[IDXSY(i, j, k)] * vat;
      }
    });
  }).wait();
}

void kernel_hadvuv_v_sycl_dcu(
    buffer<DATA_TYPE, 3> &vout_buf, buffer<DATA_TYPE, 3> &uin_buf, buffer<DATA_TYPE, 3> &vin_buf,
    buffer<DATA_TYPE, 1> &acrlat1_buf, buffer<DATA_TYPE, 1> &tgrlatda1_buf,
    DATA_TYPE eddlat, DATA_TYPE eddlon, queue &Q)
{
  const range<1> local_range{WG_SIZE};
  const range<1> global_range{round_up(CELLS, WG_SIZE)};
  Q.submit([&](handler &h) {
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto acrlat1 = acrlat1_buf.get_access<access::mode::read>(h);
    auto tgrlatda1 = tgrlatda1_buf.get_access<access::mode::read>(h);
    auto vout = vout_buf.get_access<access::mode::write>(h);
    h.parallel_for(nd_range<1>(global_range, local_range), [=](nd_item<1> item) {
      const size_t linear = item.get_global_id(0);
      if (linear < CELLS) {
        size_t i, j, k;
        decode_index(linear, i, j, k);
        const DATA_TYPE uat = DATA_TYPE(0.25) * (uin[IDXSY(i - 1, j, k)] + uin[IDXSY(i, j, k)] + uin[IDXSY(i, j + 1, k)] + uin[IDXSY(i - 1, j + 1, k)]);
        const DATA_TYPE vat = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (vin[IDXSY(i, j - 1, k)] + vin[IDXSY(i, j, k)] + vin[IDXSY(i, j + 1, k)]);
        const DATA_TYPE uavg_v = acrlat1[IDX_1D(j)] * uat;
        const DATA_TYPE vavg_v = EARTH_RADIUS_RECIP * vat;
        const DATA_TYPE vres_v = advection_driver_sycl(vin, i, j, k, uavg_v, vavg_v, eddlat, eddlon);
        vout[IDXSY(i, j, k)] = vres_v - tgrlatda1[IDX_1D(j)] * uat * uat;
      }
    });
  }).wait();
}

void kernel_hadvuv_sycl_dcu(
    buffer<DATA_TYPE, 3> &uout_buf, buffer<DATA_TYPE, 3> &vout_buf, buffer<DATA_TYPE, 3> &uin_buf, buffer<DATA_TYPE, 3> &vin_buf,
    buffer<DATA_TYPE, 1> &acrlat0_buf, buffer<DATA_TYPE, 1> &acrlat1_buf, buffer<DATA_TYPE, 1> &tgrlatda0_buf, buffer<DATA_TYPE, 1> &tgrlatda1_buf,
    buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &,
    buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &, DATA_TYPE eddlat, DATA_TYPE eddlon, queue &Q)
{
  kernel_hadvuv_u_sycl_dcu(uout_buf, uin_buf, vin_buf, acrlat0_buf, tgrlatda0_buf, eddlat, eddlon, Q);
  kernel_hadvuv_v_sycl_dcu(vout_buf, uin_buf, vin_buf, acrlat1_buf, tgrlatda1_buf, eddlat, eddlon, Q);
}

} // namespace

void hadvuv_sycl_dcu(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  queue Q{gpu_selector_v};
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
    kernel_hadvuv_sycl_dcu(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                           uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);
  }
}

void bench_hadvuv_sycl_dcu(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  queue Q{gpu_selector_v};
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
      kernel_hadvuv_sycl_dcu(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                             uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);

    TIMEIT({
      kernel_hadvuv_sycl_dcu(uout_buf, vout_buf, uin_buf, vin_buf, acrlat0_buf, acrlat1_buf, tgrlatda0_buf, tgrlatda1_buf, uatupos_buf, vatupos_buf,
                             uatvpos_buf, vatvpos_buf, uavg_buf, vavg_buf, ures_buf, vres_buf, eddlat, eddlon, Q);
    }, BENCH_REPS, "\n", "sycl-dcu");
  }
}
