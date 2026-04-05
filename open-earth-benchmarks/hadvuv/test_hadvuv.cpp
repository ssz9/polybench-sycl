#include "hadvuv.h"

extern void hadvuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *acrlat0, const DATA_TYPE *acrlat1,
    const DATA_TYPE *tgrlatda0, const DATA_TYPE *tgrlatda1, DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos,
    DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres, DATA_TYPE eddlat, DATA_TYPE eddlon);
extern void bench_hadvuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *acrlat0, const DATA_TYPE *acrlat1,
    const DATA_TYPE *tgrlatda0, const DATA_TYPE *tgrlatda1, DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos,
    DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres, DATA_TYPE eddlat, DATA_TYPE eddlon);
extern void hadvuv_sycl(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon);
extern void bench_hadvuv_sycl(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon);
#ifdef PLF_SW
extern void hadvuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon);
extern void bench_hadvuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *acrlat0, DATA_TYPE *acrlat1, DATA_TYPE *tgrlatda0, DATA_TYPE *tgrlatda1,
    DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos, DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres,
    DATA_TYPE eddlat, DATA_TYPE eddlon);
#endif

namespace {

struct HadvuvData {
  DATA_TYPE *uin;
  DATA_TYPE *vin;
  DATA_TYPE *acrlat0;
  DATA_TYPE *acrlat1;
  DATA_TYPE *tgrlatda0;
  DATA_TYPE *tgrlatda1;
  DATA_TYPE *uatupos;
  DATA_TYPE *vatupos;
  DATA_TYPE *uatvpos;
  DATA_TYPE *vatvpos;
  DATA_TYPE *uavg;
  DATA_TYPE *vavg;
  DATA_TYPE *ures;
  DATA_TYPE *vres;
  DATA_TYPE *uout;
  DATA_TYPE *vout;
};

HadvuvData alloc_data()
{
  return {alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc1D(DOMAIN_SIZE), alloc1D(DOMAIN_SIZE),
          alloc1D(DOMAIN_SIZE), alloc1D(DOMAIN_SIZE), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT)};
}

void free_data(HadvuvData &d)
{
  std::free(d.uin);
  std::free(d.vin);
  std::free(d.acrlat0);
  std::free(d.acrlat1);
  std::free(d.tgrlatda0);
  std::free(d.tgrlatda1);
  std::free(d.uatupos);
  std::free(d.vatupos);
  std::free(d.uatvpos);
  std::free(d.vatvpos);
  std::free(d.uavg);
  std::free(d.vavg);
  std::free(d.ures);
  std::free(d.vres);
  std::free(d.uout);
  std::free(d.vout);
}

void init_hadvuv(HadvuvData &d)
{
  fillMath3D(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, d.uin);
  fillMath3D(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, d.vin);
  fillMath1D(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, d.acrlat0);
  fillMath1D(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, d.acrlat1);
  fillMath1D(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, d.tgrlatda0);
  fillMath1D(3.2, 7.0, 2.5, 6.1, 3.0, 2.3, d.tgrlatda1);
  initValue(d.uatupos, DATA_TYPE{0});
  initValue(d.vatupos, DATA_TYPE{0});
  initValue(d.uatvpos, DATA_TYPE{0});
  initValue(d.vatvpos, DATA_TYPE{0});
  initValue(d.uavg, DATA_TYPE{0});
  initValue(d.vavg, DATA_TYPE{0});
  initValue(d.ures, DATA_TYPE{0});
  initValue(d.vres, DATA_TYPE{0});
  initValue(d.uout, DATA_TYPE{0});
  initValue(d.vout, DATA_TYPE{0});
}

bool check_hadvuv()
{
  const DATA_TYPE eddlat = std::ldexp(DATA_TYPE(1.0), -11);
  const DATA_TYPE eddlon = std::ldexp(DATA_TYPE(1.5), -11);

  HadvuvData gold = alloc_data();
  HadvuvData out = alloc_data();
  init_hadvuv(gold);
  init_hadvuv(out);

  hadvuv_serial(gold.uout, gold.vout, gold.uin, gold.vin, gold.acrlat0, gold.acrlat1, gold.tgrlatda0, gold.tgrlatda1,
                gold.uatupos, gold.vatupos, gold.uatvpos, gold.vatvpos, gold.uavg, gold.vavg, gold.ures, gold.vres, eddlat, eddlon);

  hadvuv_sycl(out.uout, out.vout, out.uin, out.vin, out.acrlat0, out.acrlat1, out.tgrlatda0, out.tgrlatda1, out.uatupos, out.vatupos,
              out.uatvpos, out.vatvpos, out.uavg, out.vavg, out.ures, out.vres, eddlat, eddlon);

  const size_t size_3d = (size_t)(DOMAIN_SIZE + 2 * HALO_WIDTH) * (DOMAIN_SIZE + 2 * HALO_WIDTH) * (DOMAIN_HEIGHT + 2 * HALO_WIDTH);
  bool u_ok = compare_array(gold.uout, out.uout, size_3d);
  bool v_ok = compare_array(gold.vout, out.vout, size_3d);
  std::printf("compare uout (sycl-naive): %s\n", u_ok ? "PASS" : "FAIL");
  std::printf("compare vout (sycl-naive): %s\n", v_ok ? "PASS" : "FAIL");

#ifdef PLF_SW
  init_hadvuv(out);
  hadvuv_sycl_sw(out.uout, out.vout, out.uin, out.vin, out.acrlat0, out.acrlat1, out.tgrlatda0, out.tgrlatda1, out.uatupos, out.vatupos,
                 out.uatvpos, out.vatvpos, out.uavg, out.vavg, out.ures, out.vres, eddlat, eddlon);
  bool u_sw_ok = compare_array(gold.uout, out.uout, size_3d);
  bool v_sw_ok = compare_array(gold.vout, out.vout, size_3d);
  std::printf("compare uout (sycl-sw): %s\n", u_sw_ok ? "PASS" : "FAIL");
  std::printf("compare vout (sycl-sw): %s\n", v_sw_ok ? "PASS" : "FAIL");
  u_ok = u_ok && u_sw_ok;
  v_ok = v_ok && v_sw_ok;
#endif

  free_data(gold);
  free_data(out);
  return u_ok && v_ok;
}

void bench_hadvuv()
{
  const DATA_TYPE eddlat = std::ldexp(DATA_TYPE(1.0), -11);
  const DATA_TYPE eddlon = std::ldexp(DATA_TYPE(1.5), -11);

  HadvuvData d = alloc_data();
  init_hadvuv(d);
  bench_hadvuv_serial(d.uout, d.vout, d.uin, d.vin, d.acrlat0, d.acrlat1, d.tgrlatda0, d.tgrlatda1,
                      d.uatupos, d.vatupos, d.uatvpos, d.vatvpos, d.uavg, d.vavg, d.ures, d.vres, eddlat, eddlon);

  init_hadvuv(d);
  bench_hadvuv_sycl(d.uout, d.vout, d.uin, d.vin, d.acrlat0, d.acrlat1, d.tgrlatda0, d.tgrlatda1, d.uatupos, d.vatupos, d.uatvpos, d.vatvpos, d.uavg,
                    d.vavg, d.ures, d.vres, eddlat, eddlon);
#ifdef PLF_SW
  init_hadvuv(d);
  bench_hadvuv_sycl_sw(d.uout, d.vout, d.uin, d.vin, d.acrlat0, d.acrlat1, d.tgrlatda0, d.tgrlatda1, d.uatupos, d.vatupos, d.uatvpos, d.vatvpos, d.uavg,
                       d.vavg, d.ures, d.vres, eddlat, eddlon);
#endif

  free_data(d);
}

} // namespace

int main()
{
  bool ok = check_hadvuv();
  bench_hadvuv();
  return ok ? 0 : 1;
}
