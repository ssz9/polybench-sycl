#include "fastwavesuv.h"

extern void fastwavesuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *utens, const DATA_TYPE *vtens,
    const DATA_TYPE *wgtfac, const DATA_TYPE *ppuv, const DATA_TYPE *hhl, const DATA_TYPE *rho, const DATA_TYPE *fx,
    DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, const DATA_TYPE edadlat, const DATA_TYPE dt);
extern void bench_fastwavesuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *utens, const DATA_TYPE *vtens,
    const DATA_TYPE *wgtfac, const DATA_TYPE *ppuv, const DATA_TYPE *hhl, const DATA_TYPE *rho, const DATA_TYPE *fx,
    DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, const DATA_TYPE edadlat, const DATA_TYPE dt);
extern void fastwavesuv_sycl(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt);
extern void bench_fastwavesuv_sycl(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt);
#ifdef PLF_SW
extern void fastwavesuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt);
extern void bench_fastwavesuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt);
#endif
#ifdef PLF_DCU
extern void fastwavesuv_sycl_dcu(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt);
extern void bench_fastwavesuv_sycl_dcu(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt);
#endif

namespace {

struct FastwavesuvData {
  DATA_TYPE *uin;
  DATA_TYPE *utens;
  DATA_TYPE *vin;
  DATA_TYPE *vtens;
  DATA_TYPE *wgtfac;
  DATA_TYPE *ppuv;
  DATA_TYPE *hhl;
  DATA_TYPE *rho;
  DATA_TYPE *uout;
  DATA_TYPE *vout;
  DATA_TYPE *fx;
  DATA_TYPE *ppgk;
  DATA_TYPE *ppgc;
  DATA_TYPE *ppgu;
  DATA_TYPE *ppgv;
};

FastwavesuvData alloc_data()
{
  return {alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc1D(DOMAIN_SIZE), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT),
          alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT), alloc3D(DOMAIN_SIZE, DOMAIN_HEIGHT)};
}

void free_data(FastwavesuvData &d)
{
  std::free(d.uin);
  std::free(d.utens);
  std::free(d.vin);
  std::free(d.vtens);
  std::free(d.wgtfac);
  std::free(d.ppuv);
  std::free(d.hhl);
  std::free(d.rho);
  std::free(d.uout);
  std::free(d.vout);
  std::free(d.fx);
  std::free(d.ppgk);
  std::free(d.ppgc);
  std::free(d.ppgu);
  std::free(d.ppgv);
}

void init_fastwavesuv(FastwavesuvData &d)
{
  fillMath3D(1.0, 3.3, 1.5, 1.5, 2.0, 4.0, d.uin);
  fillMath3D(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, d.utens);
  fillMath3D(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, d.vin);
  fillMath3D(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, d.vtens);
  fillMath3D(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, d.ppuv);
  fillMath3D(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, d.wgtfac);
  fillMath3D(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, d.hhl);
  fillMath3D(3.2, 7.0, 2.5, 6.1, 3.0, 2.3, d.rho);
  fillMath1D(4.5, 5.0, 2.5, 2.1, 3.0, 2.3, d.fx);
  initValue(d.uout, DATA_TYPE{0});
  initValue(d.vout, DATA_TYPE{0});
  initValue(d.ppgk, DATA_TYPE{0});
  initValue(d.ppgc, DATA_TYPE{0});
  initValue(d.ppgu, DATA_TYPE{0});
  initValue(d.ppgv, DATA_TYPE{0});
}

bool check_fastwavesuv()
{
  constexpr DATA_TYPE dt = DATA_TYPE(10.0);
  constexpr DATA_TYPE edadlat = DATA_TYPE(10.0);

  FastwavesuvData gold = alloc_data();
  FastwavesuvData out = alloc_data();
  init_fastwavesuv(gold);
  init_fastwavesuv(out);

  fastwavesuv_serial(gold.uout, gold.vout, gold.uin, gold.vin, gold.utens, gold.vtens, gold.wgtfac, gold.ppuv, gold.hhl, gold.rho, gold.fx,
                     gold.ppgk, gold.ppgc, gold.ppgu, gold.ppgv, edadlat, dt);

  fastwavesuv_sycl(out.uout, out.vout, out.uin, out.vin, out.utens, out.vtens, out.wgtfac, out.ppuv, out.hhl, out.rho, out.fx, out.ppgk, out.ppgc,
                   out.ppgu, out.ppgv, edadlat, dt);

  const size_t size_3d = (size_t)(DOMAIN_SIZE + 2 * HALO_WIDTH) * (DOMAIN_SIZE + 2 * HALO_WIDTH) * (DOMAIN_HEIGHT + 2 * HALO_WIDTH);
  bool u_ok = compare_array(gold.uout, out.uout, size_3d);
  bool v_ok = compare_array(gold.vout, out.vout, size_3d);
  std::printf("compare uout (sycl-naive): %s\n", u_ok ? "PASS" : "FAIL");
  std::printf("compare vout (sycl-naive): %s\n", v_ok ? "PASS" : "FAIL");

#ifdef PLF_SW
  init_fastwavesuv(out);
  fastwavesuv_sycl_sw(out.uout, out.vout, out.uin, out.vin, out.utens, out.vtens, out.wgtfac, out.ppuv, out.hhl, out.rho, out.fx, out.ppgk, out.ppgc,
                      out.ppgu, out.ppgv, edadlat, dt);
  bool u_sw_ok = compare_array(gold.uout, out.uout, size_3d);
  bool v_sw_ok = compare_array(gold.vout, out.vout, size_3d);
  std::printf("compare uout (sycl-sw): %s\n", u_sw_ok ? "PASS" : "FAIL");
  std::printf("compare vout (sycl-sw): %s\n", v_sw_ok ? "PASS" : "FAIL");
  u_ok = u_ok && u_sw_ok;
  v_ok = v_ok && v_sw_ok;
#endif
#ifdef PLF_DCU
  init_fastwavesuv(out);
  fastwavesuv_sycl_dcu(out.uout, out.vout, out.uin, out.vin, out.utens, out.vtens, out.wgtfac, out.ppuv, out.hhl, out.rho, out.fx, out.ppgk, out.ppgc,
                       out.ppgu, out.ppgv, edadlat, dt);
  bool u_dcu_ok = compare_array(gold.uout, out.uout, size_3d);
  bool v_dcu_ok = compare_array(gold.vout, out.vout, size_3d);
  std::printf("compare uout (sycl-dcu): %s\n", u_dcu_ok ? "PASS" : "FAIL");
  std::printf("compare vout (sycl-dcu): %s\n", v_dcu_ok ? "PASS" : "FAIL");
  u_ok = u_ok && u_dcu_ok;
  v_ok = v_ok && v_dcu_ok;
#endif

  free_data(gold);
  free_data(out);
  return u_ok && v_ok;
}

void bench_fastwavesuv()
{
  constexpr DATA_TYPE dt = DATA_TYPE(10.0);
  constexpr DATA_TYPE edadlat = DATA_TYPE(10.0);

  FastwavesuvData d = alloc_data();
  init_fastwavesuv(d);
  bench_fastwavesuv_serial(d.uout, d.vout, d.uin, d.vin, d.utens, d.vtens, d.wgtfac, d.ppuv, d.hhl, d.rho, d.fx,
                           d.ppgk, d.ppgc, d.ppgu, d.ppgv, edadlat, dt);

  init_fastwavesuv(d);
  bench_fastwavesuv_sycl(d.uout, d.vout, d.uin, d.vin, d.utens, d.vtens, d.wgtfac, d.ppuv, d.hhl, d.rho, d.fx, d.ppgk, d.ppgc, d.ppgu, d.ppgv,
                         edadlat, dt);
#ifdef PLF_SW
  init_fastwavesuv(d);
  bench_fastwavesuv_sycl_sw(d.uout, d.vout, d.uin, d.vin, d.utens, d.vtens, d.wgtfac, d.ppuv, d.hhl, d.rho, d.fx, d.ppgk, d.ppgc, d.ppgu, d.ppgv,
                            edadlat, dt);
#endif
#ifdef PLF_DCU
  init_fastwavesuv(d);
  bench_fastwavesuv_sycl_dcu(d.uout, d.vout, d.uin, d.vin, d.utens, d.vtens, d.wgtfac, d.ppuv, d.hhl, d.rho, d.fx, d.ppgk, d.ppgc, d.ppgu, d.ppgv,
                             edadlat, dt);
#endif

  free_data(d);
}

} // namespace

int main()
{
  bool ok = check_fastwavesuv();
  bench_fastwavesuv();
  return ok ? 0 : 1;
}
