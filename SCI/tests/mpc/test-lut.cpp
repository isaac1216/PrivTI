/*
Authors: Deevashwer Rathee, Mayank Rathee
Copyright (c) 2021 Microsoft Research
(MIT License)
*/

#include "BuildingBlocks/aux-protocols.h"
#include "FloatingPoint/fp-math.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

using namespace sci;
using namespace std;

#define MAX_THREADS 16

int party, port = 32000;
int num_threads = 16;
string address = "127.0.0.1";

static const int32_t fixed_dim = 128*128;

// we will use bw_y for LUT output width (fits uint64_t)
int32_t bw_y = 37;
int32_t s_y  = 12;

// (only for compatibility printing; not used by OT-LUT/SCAN here)
int32_t bw_x = 37;
int32_t s_x  = 12;

uint64_t mask_y = (bw_y == 64 ? ~0ULL : ((1ULL << bw_y) - 1));

IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];

struct BenchResult {
  double time_us;
  uint64_t comm_bytes;
};

enum class Mode : int { OT = 0, SCAN = 1 };
static Mode g_mode = Mode::OT;
static int  g_N = 10;
static int  warmup = 3;
static int  repeat = 10;

// ---------- helpers ----------
static inline int ceil_log2(int n) {
  int b = 0;
  int v = 1;
  while (v < n) { v <<= 1; b++; }
  return b;
}
static inline int next_pow2(int n) {
  int v = 1;
  while (v < n) v <<= 1;
  return v;
}
static inline uint64_t fp_exp_neg_i(int i, int s) {
  double val = std::exp(-(double)i) * std::ldexp(1.0, s);
  long long r = llround(val);
  if (r < 0) r = 0;
  return (uint64_t)r;
}

// Build LUT of length L (L is power-of-two). Only first N entries are "real", rest padded 0.
static inline void build_lut_pow2(vector<uint64_t> &T, int N, int L, int s) {
  T.assign(L, 0ULL);
  for (int i = 0; i < N; i++) T[i] = fp_exp_neg_i(i, s) & mask_y;
}

// Create XOR-shared selection bits sel[] for a given predicate (idx==target).
// - BOB knows idx_plain[] and computes bit = (idx==target).
// - ALICE samples random mask r[] and sends to BOB.
// - ALICE uses sel = r, BOB uses sel = r xor bit.
// This hides idx from ALICE.
static inline void make_sel_xor_shares(int tid,
                                      const uint16_t *idx_plain, // BOB only meaningful
                                      int n,
                                      int target,
                                      uint8_t *sel_out) {
  if (party == ALICE) {
    PRG128 prg;
    prg.random_bool((bool*)sel_out, n);            // sel_out = r
    iopackArr[tid]->io->send_data(sel_out, n);     // send r to BOB
  } else { // BOB
    uint8_t *r = new uint8_t[n];
    iopackArr[tid]->io->recv_data(r, n);           // receive r
    for (int i = 0; i < n; i++) {
      uint8_t bit = (idx_plain[i] == (uint16_t)target) ? 1 : 0;
      sel_out[i] = r[i] ^ bit;
    }
    delete[] r;
  }
  iopackArr[tid]->io->flush();
}

// ------------- SCAN baseline -------------
// We compute y = Σ sel_i * T[i] using AuxProtocols::multiplexer(sel, x, y)
// multiplexer(sel, x, y) computes y = (sel0 xor sel1) * (x0 + x1) in arithmetic shares.
// So if ALICE sets x = T[i] and BOB sets x = 0, output is sel * T[i] (secret-shared).
static inline void scan_lookup(int tid, int this_party,
                               const uint16_t *idx_plain, // BOB has real idx; ALICE dummy
                               uint64_t *y_out,           // arithmetic shares
                               int n,
                               const vector<uint64_t> &T_pow2,
                               int L) {
  AuxProtocols aux(this_party, iopackArr[tid], otpackArr[tid]);

  vector<uint64_t> acc(n, 0ULL);
  vector<uint64_t> x(n, 0ULL);
  vector<uint64_t> term(n, 0ULL);
  vector<uint8_t>  sel(n, 0);

  for (int i = 0; i < L; i++) {
    // secret-share sel for predicate (idx==i)
    make_sel_xor_shares(tid, idx_plain, n, i, sel.data());

    // x holds shares of public T[i]: ALICE has T[i], BOB has 0
    if (party == ALICE) {
      uint64_t ti = T_pow2[i] & mask_y;
      for (int j = 0; j < n; j++) x[j] = ti;
    } else {
      std::fill(x.begin(), x.end(), 0ULL);
    }

    // term = sel * T[i] (in arithmetic shares)
    aux.multiplexer(sel.data(), x.data(), term.data(), n, bw_y, bw_y);

    // acc += term
    for (int j = 0; j < n; j++) acc[j] = (acc[j] + term[j]) & mask_y;
  }

  memcpy(y_out, acc.data(), n * sizeof(uint64_t));
}

// ------------- OT LUT -------------
// Use AuxProtocols::lookup_table<uint64_t> which requires bw_x <= 8.
// For N<=256: single lookup with L=2^bw (L<=256).
// For N padded to 1024: do 4 blocks of 256 using low8, then oblivious mux by high2 bits.
static inline void ot_lookup_pow2_L(int tid, int this_party,
                                   const uint16_t *idx_plain, // BOB real idx
                                   uint64_t *y_out, int n,
                                   const vector<uint64_t> &T_pow2,
                                   int L) {
  AuxProtocols aux(this_party, iopackArr[tid], otpackArr[tid]);

  if (L <= 256) {
    int bw_idx = ceil_log2(L);
    assert(bw_idx >= 1 && bw_idx <= 8);

    // Build spec pointers: size x L (same table for all samples)
    uint64_t **spec = nullptr;
    uint64_t *x_in  = nullptr;
    uint64_t *y_tmp = nullptr;

    if (party == ALICE) {
      spec = new uint64_t*[n];
      for (int i = 0; i < n; i++) spec[i] = const_cast<uint64_t*>(T_pow2.data());
    } else {
      x_in = new uint64_t[n];
      y_tmp = new uint64_t[n];
      for (int i = 0; i < n; i++) x_in[i] = (uint64_t)(idx_plain[i] & (L - 1));
    }

    aux.lookup_table<uint64_t>(spec, x_in, (party==BOB? y_tmp: nullptr), n, bw_idx, bw_y);

    if (party == BOB) {
      memcpy(y_out, y_tmp, n * sizeof(uint64_t));
      delete[] x_in;
      delete[] y_tmp;
    } else {
      // ALICE does not receive output in this API; output share is implicitly 0 on ALICE side.
      // To keep additive sharing consistent, set ALICE output share to 0.
      memset(y_out, 0, n * sizeof(uint64_t));
      delete[] spec;
    }
    return;
  }

  // L == 1024 case: split into 4 blocks of 256, using bw_low=8 and hi2 bits for mux
  assert(L == 1024);

  // Prepare 4 block tables of 256
  const int BW_LOW = 8;
  const int BLK = 256;
  const int NB = 4;

  // Block pointers into T_pow2
  const uint64_t *T0 = T_pow2.data() + 0 * BLK;
  const uint64_t *T1 = T_pow2.data() + 1 * BLK;
  const uint64_t *T2 = T_pow2.data() + 2 * BLK;
  const uint64_t *T3 = T_pow2.data() + 3 * BLK;

  // BOB computes low and high
  vector<uint64_t> y0(n,0), y1(n,0), y2(n,0), y3(n,0);
  uint64_t **spec = nullptr;
  uint64_t *x_low = nullptr;
  uint64_t *tmp   = nullptr;

  if (party == ALICE) {
    // run 4 sends (each is a KKOT send of 256 entries per sample)
    // We'll reuse spec pointers per block.
    spec = new uint64_t*[n];
  } else {
    x_low = new uint64_t[n];
    tmp = new uint64_t[n];
    for (int i = 0; i < n; i++) x_low[i] = (uint64_t)(idx_plain[i] & 0xFF);
  }

  auto do_block = [&](const uint64_t *Tb, vector<uint64_t> &yb) {
    if (party == ALICE) {
      for (int i = 0; i < n; i++) spec[i] = const_cast<uint64_t*>(Tb);
      aux.lookup_table<uint64_t>(spec, nullptr, nullptr, n, BW_LOW, bw_y);
      // ALICE share is 0
      std::fill(yb.begin(), yb.end(), 0ULL);
    } else {
      aux.lookup_table<uint64_t>(nullptr, x_low, tmp, n, BW_LOW, bw_y);
      memcpy(yb.data(), tmp, n * sizeof(uint64_t));
    }
  };

  do_block(T0, y0);
  do_block(T1, y1);
  do_block(T2, y2);
  do_block(T3, y3);

  if (party == ALICE) {
    delete[] spec;
  } else {
    delete[] x_low;
    delete[] tmp;
  }

  // Now obliviously mux according to hi2 bits (b1 b0):
  // m01 = mux(b0, y0, y1), m23 = mux(b0, y2, y3), out = mux(b1, m01, m23)
  // Using formula: out = base + sel*(other-base), implemented via aux.multiplexer(sel, diff)
  AuxProtocols aux2(this_party, iopackArr[tid], otpackArr[tid]);

  vector<uint8_t> sel_b0(n), sel_b1(n);
  vector<uint64_t> diff(n), masked(n), m01(n), m23(n), out(n);

  // create XOR shares for b0 and b1 (BOB knows bits)
  // we reuse make_sel_xor_shares but target is bit value not eq, so make small helper:
  auto share_bit = [&](int bitpos, vector<uint8_t>& sel_out){
    if (party == ALICE) {
      PRG128 prg;
      prg.random_bool((bool*)sel_out.data(), n);
      iopackArr[tid]->io->send_data(sel_out.data(), n);
    } else {

      uint8_t *r = new uint8_t[n];
      iopackArr[tid]->io->recv_data(r, n);
      for (int i = 0; i < n; i++) {
        uint8_t bit = (uint8_t)((idx_plain[i] >> bitpos) & 1);
        sel_out[i] = r[i] ^ bit;
      }
      delete[] r;
    }
    iopackArr[tid]->io->flush();
  };

  // b0 is bit 8, b1 is bit 9 (since low8 used)
  share_bit(8, sel_b0);
  share_bit(9, sel_b1);

  auto mux2 = [&](const vector<uint8_t>& sel,
                  const vector<uint64_t>& a,
                  const vector<uint64_t>& b,
                  vector<uint64_t>& outv) {
    // out = b + sel*(a-b)
    for (int i = 0; i < n; i++) diff[i] = (a[i] - b[i]) & mask_y;
    aux2.multiplexer(const_cast<uint8_t*>(sel.data()),
                     diff.data(), masked.data(), n, bw_y, bw_y);
    for (int i = 0; i < n; i++) outv[i] = (b[i] + masked[i]) & mask_y;
  };

  mux2(sel_b0, y0, y1, m01);
  mux2(sel_b0, y2, y3, m23);
  mux2(sel_b1, m01, m23, out);

  memcpy(y_out, out.data(), n * sizeof(uint64_t));
}

// -------------------- thread entry --------------------
static void operation_thread(int tid,
                             const uint16_t *idx_plain, // BOB has idx; ALICE dummy
                             uint64_t *y_share,
                             int n,
                             const vector<uint64_t> *T_pow2,
                             int L) {
  int this_party = (tid & 1) ? (3 - party) : party;

  if (g_mode == Mode::OT) {
    ot_lookup_pow2_L(tid, this_party, idx_plain, y_share, n, *T_pow2, L);
  } else {
    scan_lookup(tid, this_party, idx_plain, y_share, n, *T_pow2, L);
  }
}

// Run one benchmark
static BenchResult run_one_dim(int32_t dim, const vector<uint64_t> &T_pow2, int L) {
  // BOB holds plaintext indices; ALICE doesn't need them
  uint16_t *idx = new uint16_t[dim];
  uint64_t *y   = new uint64_t[dim];

  PRG128 prg;
  if (party == BOB) {
    vector<uint64_t> tmp(dim);
    prg.random_data(tmp.data(), dim * sizeof(uint64_t));
    for (int i = 0; i < dim; i++) idx[i] = (uint16_t)(tmp[i] % (uint64_t)g_N); // in [0..N-1]
  } else {
    memset(idx, 0, dim * sizeof(uint16_t));
  }

  int active_threads = min(num_threads, (int)dim);
  if (active_threads <= 0) active_threads = 1;

  vector<uint64_t> comm_base(active_threads);
  for (int i = 0; i < active_threads; i++) comm_base[i] = iopackArr[i]->get_comm();

  auto start = clock_start();
  vector<thread> threads;
  threads.reserve(active_threads);

  int chunk = (dim + active_threads - 1) / active_threads;
  for (int tid = 0; tid < active_threads; tid++) {
    int off = tid * chunk;
    int len = min(chunk, dim - off);
    if (len <= 0) break;
    threads.emplace_back(operation_thread, tid, idx + off, y + off, len, &T_pow2, L);
  }
  for (auto &th : threads) th.join();
  long long t_us = time_from(start);

  uint64_t total_comm = 0;
  for (int i = 0; i < active_threads; i++) {
    total_comm += (iopackArr[i]->get_comm() - comm_base[i]);
  }

  delete[] idx;
  delete[] y;

  return BenchResult{(double)t_us, total_comm};
}

static inline const char* mode_name(Mode m){ return (m==Mode::OT) ? "OT" : "SCAN"; }

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "ALICE=1; BOB=2");
  amap.arg("p", port, "port");
  amap.arg("nt", num_threads, "threads");
  amap.arg("ip", address, "ip");

  int mode_int = 0;
  amap.arg("mode", mode_int, "0=OT(KKOT LUT), 1=SCAN(linear mux)");
  amap.arg("N", g_N, "N candidates (10/100/1000)");
  amap.arg("warmup", warmup, "warmup");
  amap.arg("repeat", repeat, "repeat");

  // allow override output bits
  amap.arg("bw_y", bw_y, "LUT output bitwidth");
  amap.arg("s_y",  s_y,  "LUT output frac bits");

  amap.parse(argc, argv);

  g_mode = (mode_int==1) ? Mode::SCAN : Mode::OT;
  assert(num_threads <= MAX_THREADS);

  mask_y = (bw_y == 64 ? ~0ULL : ((1ULL << bw_y) - 1));

  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    else       otpackArr[i] = new OTPack(iopackArr[i], party);
  }
  if (party == ALICE) cout << "All Base OTs Done\n";

  // Pad N to power-of-two length L
  int L = next_pow2(g_N);

  // For N=1000, L=1024, which requires hierarchical OT because lookup_table asserts bw_x<=8.
  // Our OT path supports that special case via 4x256 + mux.
  // SCAN path will scan full L.
  vector<uint64_t> T_pow2;
  build_lut_pow2(T_pow2, g_N, L, s_y);

  if (party == ALICE) {
    cout << "[" << mode_name(g_mode) << "] Benchmark fixed_dim=" << fixed_dim
         << ", N=" << g_N << ", padded_L=" << L
         << ", warmup=" << warmup
         << ", repeat=" << repeat
         << ", threads=" << num_threads << "\n";
    cout << "bw_y=" << bw_y << " s_y=" << s_y << "\n";
  }

  for (int i = 0; i < warmup; i++) (void)run_one_dim(fixed_dim, T_pow2, L);

  double sum_t=0, sum_t2=0, sum_c=0, sum_c2=0;
  for (int i = 0; i < repeat; i++) {
    auto r = run_one_dim(fixed_dim, T_pow2, L);
    sum_t += r.time_us; sum_t2 += r.time_us*r.time_us;
    sum_c += (double)r.comm_bytes; sum_c2 += (double)r.comm_bytes*r.comm_bytes;
  }

  double mean_t = sum_t/repeat;
  double std_t  = sqrt(max(0.0, sum_t2/repeat - mean_t*mean_t));
  double mean_c = sum_c/repeat;
  double std_c  = sqrt(max(0.0, sum_c2/repeat - mean_c*mean_c));

  if (party == ALICE) {
    cout << "fixed_dim=" << fixed_dim
         << "  time_mean=" << (mean_t/1000.0) << " ms"
         << "  time_std="  << (std_t/1000.0) << " ms"
         << "  comm_mean=" << (uint64_t)mean_c << " bytes"
         << "  comm_std="  << (uint64_t)std_c  << " bytes"
         << "  per_x_time=" << (mean_t/fixed_dim) << " us/x"
         << "  per_x_comm=" << (mean_c/fixed_dim) << " bytes/x\n";
  }

  for (int i = 0; i < num_threads; i++) { delete iopackArr[i]; delete otpackArr[i]; }
  return 0;
}

