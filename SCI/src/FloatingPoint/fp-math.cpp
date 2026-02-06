/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "FloatingPoint/fp-math.h"
#include "FloatingPoint/fp-math-coeffs.h"

using namespace std;
using namespace sci;

#define FRAC_RANGE 9
#define FP_INTMD_M_BITS 27
#define FP_INTMD_E_BITS 8
#define PI_DOUBLE                                                              \
  3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define LOG2E                                                                  \
  1.44269504088896340735992468100189213742664595415298593413544940693110921918118507988552662289350634449699
#define LOGE2                                                                  \
  0.693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418687
#define TWO_INV_SQRT_PI                                                        \
  1.128379167095512573896158903121545171688101258657997713688171443421284936882
#define NEG_LOGE2_INV                                                          \
  1.442695040888963423535598661526235116567603930130965898132921686199121361438241594804331289503955470328942932380383923264

FixArray get_idx_from_input(FixOp *fix, const FixArray &delta_m,
                            const FixArray &delta_e, int idx_m_bits,
                            int idx_e_bits, int e_offset) {
  assert(delta_m.party != PUBLIC && delta_e.party != PUBLIC);
  assert(delta_m.size == delta_e.size);
  assert(idx_m_bits + idx_e_bits <= delta_e.ell);
  FixArray idx_hi =
      fix->reduce(fix->add(delta_e, e_offset), idx_m_bits + idx_e_bits);
  idx_hi.signed_ = false;
  if (idx_m_bits == 0) {
    return idx_hi;
  }
  idx_hi = fix->mul(idx_hi, 1 << idx_m_bits, idx_m_bits + idx_e_bits);
  FixArray idx_lo = fix->truncate_reduce(delta_m, delta_m.ell - 1 - idx_m_bits);
  idx_lo = fix->sub(idx_lo, 1 << idx_m_bits);
  if (idx_m_bits + idx_e_bits < idx_m_bits + 1) {
    idx_lo = fix->reduce(idx_lo, idx_m_bits + idx_e_bits);
  }
  idx_lo.s = 0;
  BoolArray all_0 = fix->bool_op->input(ALICE, delta_m.size, uint8_t(0));
  FixArray idx = fix->add(
      idx_hi, fix->extend(idx_lo, idx_m_bits + idx_e_bits, all_0.data));
  return idx;
}

FPArray FPMath::tanpi(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);
  FPArray pos_x = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m.data, x_e.data, x.m_bits, x.e_bits);

  // Range Reduction:
  // Input: [2^-14, 2^23)
  // Output: [2^(-m_bits-REDUCED_RANGE_UB, 2^(-REDUCED_RANGE_UB))]

  // if x >= 2^-FRAC_RANGE
  BoolArray f0 = fix->GE(x_e, -FRAC_RANGE + x.e_bias());
  FixArray shift_amt = fix->add(x_e, FRAC_RANGE - x.e_bias());
  shift_amt.signed_ = false;
  FixArray N__ = fix->left_shift(x_m, shift_amt, x.m_bits + FRAC_RANGE + 1,
              22 + FRAC_RANGE, all_1.data);
  N__.s += FRAC_RANGE;
  FixArray N_ = fix->reduce(N__, FRAC_RANGE + x.m_bits);
  // f1 = N_ >= 2^(FRAC_RANGE + x.m_bits - 1)
  // Using f1 = N < 0 (signed) instead as the ring does not have 1-bit slack
  BoolArray f1 = fix->LT(N_, 0);
  N_ = fix->if_else(f1, fix->sub(1ULL << (FRAC_RANGE + x.m_bits), N_), N_);

  BoolArray f2 = fix->GE(N_, 1ULL << (FRAC_RANGE + x.m_bits - 2));
  N_ = fix->if_else(f2, fix->sub(1ULL << (FRAC_RANGE + x.m_bits - 1), N_), N_);

  FixArray f_m_ = fix->reduce(N_, x.m_bits);
  f_m_.s = x.m_bits;
  BoolArray msb_f_m_ = fix->LT(f_m_, 0);
  uint8_t *wrap_f_m_ = new uint8_t[x.size];
  fix->aux->MSB_to_Wrap(f_m_.data, msb_f_m_.data, wrap_f_m_, x.size, f_m_.ell);
  N_ = fix->truncate_reduce(N_, x.m_bits, wrap_f_m_);
  f_m_ = fix->extend(f_m_, x.m_bits + 1, msb_f_m_.data);

  BoolArray f_m__eq_zero = fix->EQ(f_m_, 0);
  BoolArray N__eq_zero = fix->EQ(N_, 0);

  FixArray f_e__if_if = fix->input(PUBLIC, x.size, uint64_t(0), x_e.signed_, x_e.ell, x_e.s);
  FixArray f_e__if =
      fix->if_else(N__eq_zero, f_e__if_if, -1 * FRAC_RANGE + x.e_bias());
  FixArray N__if = fix->if_else(N__eq_zero, N_, fix->sub(N_, 1));
  FixArray f_m__if = fix->if_else(N__eq_zero, f_m_, 1ULL << x.m_bits);
  BoolArray f_z_ = bool_op->AND(f_m__eq_zero, N__eq_zero);

  FixArray f_m__else = f_m_, N__else = N_;
  FixArray f_e__else = fix->input(PUBLIC, x.size, uint64_t(0), x_e.signed_, x_e.ell, x_e.s);
  fp_op->normalize(f_m__else, f_e__else, x.m_bits + FRAC_RANGE - x.e_bias());

  f_m_ = fix->if_else(f_m__eq_zero, f_m__if, f_m__else);
  FixArray f_e_ = fix->if_else(f_m__eq_zero, f_e__if, f_e__else);
  N_ = fix->if_else(f_m__eq_zero, N__if, N__else);

  // else
  FixArray N = fix->if_else(f0, N_, 0);
  N = fix->reduce(N, FRAC_RANGE - 2);
  FixArray f_m = fix->if_else(f0, f_m_, x_m);
  FixArray f_e = fix->if_else(f0, f_e_, x_e);
  BoolArray f_z = bool_op->if_else(f0, f_z_, x_z);
  BoolArray f_s = bool_op->input(ALICE, x.size, uint8_t(0));
  f1 = bool_op->if_else(f0, f1, 0);
  f2 = bool_op->if_else(f0, f2, 0);
  // increasing precision of f
  f_m = fix->scale_up(f_m, FP_INTMD_M_BITS + 1, FP_INTMD_M_BITS);
  FPArray f = fp_op->input(x.party, x.size, f_s.data, f_z.data,
          f_m.data, f_e.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  // Spline Evaluation on f
  BoolArray cond1 = fix->LT(f_e, -14 + f.e_bias());

  // if f < 2^-14, f_tan = pi * f
  FPArray pi = fp_op->input<double>(PUBLIC, x.size, PI_DOUBLE,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  FPArray f_tan_if = fp_op->mul(f, pi, true);

  // else spline evaluation on f
  FixArray idx = get_idx_from_input(fix, f_m, f_e, 2, 3, 14 - x.e_bias());
  vector<FPArray> theta = fp_op->GetCoeffs(tan_coeffs, trig_knots_bits, idx, 20,
                                           FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  FPArray f_tan_else = fp_op->mul(theta[1], f, false);
  f_tan_else = fp_op->mul(f_tan_else, f, false);
  f_tan_else = fp_op->add(f_tan_else, theta[0], true, true, false);
  f_tan_else = fp_op->mul(f_tan_else, f, false);

  FPArray f_tan = fp_op->if_else(cond1, f_tan_if, f_tan_else);

  // Range Propagation
  FPArray N_tan = fp_op->LUT(tan_N, N, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  FPArray sum = fp_op->add(f_tan, N_tan, true, true, false);
  FPArray prod = fp_op->mul(f_tan, N_tan, false);
  // secret-sharing one as fp_op add does not support PUBLIC FPArrays
  FPArray one = fp_op->input<float>(ALICE, x.size, 1.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  prod = fp_op->sub(one, prod, true, false, false);

  FPArray num = fp_op->if_else(f2, prod, sum);
  FPArray den = fp_op->if_else(f2, sum, prod);
  FPArray z = fp_op->div(num, den, true, false);

  // Special Cases
  // Case I
  BoolArray cond2 = fix->GE(x_e, 23 + x.e_bias());
  // secret-sharing zero as fp_op if_else does not support PUBLIC FPArrays
  FPArray zero = fp_op->input<double>(ALICE, x.size, 0.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  z = fp_op->if_else(cond2, zero, z);

  // Case II
  BoolArray cond3 = fix->LT(x_e, x.e_bias() - 14);
  // increasing precision of x
  FixArray x_m_prec = fix->scale_up(x_m, FP_INTMD_M_BITS + 1, FP_INTMD_M_BITS);
  FixArray x_e_prec = x_e;
  FPArray pos_x_prec = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m_prec.data, x_e_prec.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  FPArray pos_x_prec_pi = fp_op->mul(pos_x_prec, pi, true);
  z = fp_op->if_else(cond3, pos_x_prec_pi, z);

  // reduce precision to normal
  BoolArray z_s, z_z;
  FixArray z_m, z_e;
  tie(z_s, z_z, z_m, z_e) = fp_op->get_components(z);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(z_m, oflow_threshold);
  FixArray ret_m_if = fix->round_ties_to_even(z_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray ret_m = fix->if_else(rnd_no_oflow, ret_m_if, (1ULL << x.m_bits));
  FixArray ret_e = fix->if_else(rnd_no_oflow, z_e, fix->add(z_e, 1));

  BoolArray ret_s = bool_op->XOR(f1, x_s);
  FPArray ret = fp_op->input(x.party, x.size, ret_s.data, z_z.data,
          ret_m.data, ret_e.data, x.m_bits, x.e_bits);

  delete[] wrap_f_m_;

  return ret;
}

FPArray FPMath::exp2(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);
  FixArray shift_amt = fix->add(x_e, 24 - x.e_bias());
  shift_amt.signed_ = false;
  FixArray m = fix->left_shift(x_m, shift_amt, x.m_bits + 32, 31, all_1.data);
  m.s += 24;
  FixArray f = fix->reduce(m, x.m_bits + 24);
  BoolArray msb_f = fix->LT(f, 0);
  uint8_t *wrap_f = new uint8_t[x.size];
  fix->aux->MSB_to_Wrap(f.data, msb_f.data, wrap_f, x.size, f.ell);
  FixArray N = fix->truncate_reduce(m, x.m_bits + 24, wrap_f);
  N.signed_ = true;
  N = fix->if_else(x_s, fix->mul(N, -1), N);
  f = fix->extend(f, x.m_bits + 25, msb_f.data);

  BoolArray f_eq_0 = fix->EQ(f, 0);
  BoolArray neg_f = bool_op->AND(x_s, bool_op->NOT(f_eq_0));
  f = fix->if_else(neg_f, fix->sub(1ULL << (x.m_bits + 24), f), f);
  N = fix->if_else(neg_f, fix->sub(N, 1), N);
  FixArray delta_m = f;
  FixArray delta_e = fix->input(PUBLIC, x.size, uint64_t(0), x_e.signed_, x_e.ell, x_e.s);
  fp_op->normalize(delta_m, delta_e, x.m_bits + 24 - x.e_bias());
  delta_m = fix->truncate_reduce(delta_m, 24 + x.m_bits - FP_INTMD_M_BITS);
  delta_e = fix->if_else(f_eq_0, 0, delta_e);
  FPArray delta = fp_op->input(x.party, x.size, all_0.data, f_eq_0.data,
          delta_m.data, delta_e.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  // if delta < 2^-14, mu = 1.0
  BoolArray cond = fix->LT(delta_e, -24 + delta.e_bias());
  // secret-sharing one as fp_op else_if does not support PUBLIC FPArrays
  FPArray one = fp_op->input<float>(ALICE, x.size, 1.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  FPArray mu_if = one;

  // else evaluate spline
  BoolArray e_lt_neg_6 = fix->LT(delta_e, delta.e_bias() - 6);
  FixArray delta_e_prime =
      fix->if_else(e_lt_neg_6, delta.e_bias() - 7, delta_e);
  FixArray idx =
      get_idx_from_input(fix, delta_m, delta_e_prime, 5, 3, 7 - delta.e_bias());
  vector<FPArray> theta = fp_op->GetCoeffs(
      exp2_coeffs, exp2_knots_bits, idx, 64, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  FPArray mu_else = fp_op->mul(theta[2], delta, false);
  mu_else = fp_op->add(mu_else, theta[1], true, true, false);
  mu_else = fp_op->mul(mu_else, delta, false);
  mu_else = fp_op->add(mu_else, theta[0], true, true, false);
  FPArray mu = fp_op->if_else(cond, mu_if, mu_else);

  BoolArray mu_s, mu_z;
  FixArray mu_m, mu_e;
  tie(mu_s, mu_z, mu_m, mu_e) = fp_op->get_components(mu);
  N = fix->extend(N, x.e_bits + 2);
  FixArray gamma_e = fix->add(mu_e, N);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(mu_m, oflow_threshold);
  FixArray gamma_m_if =
      fix->round_ties_to_even(mu_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray gamma_m = fix->if_else(rnd_no_oflow, gamma_m_if, (1ULL << x.m_bits));
  gamma_e = fix->if_else(rnd_no_oflow, gamma_e, fix->add(gamma_e, 1));
  FPArray gamma = fp_op->input(x.party, x.size, all_0.data, all_0.data,
          gamma_m.data, gamma_e.data, x.m_bits, x.e_bits);

  // Special Cases
  // Case I
  BoolArray cond1 =
      bool_op->AND(bool_op->NOT(x_s), fix->GE(x_e, 7 + x.e_bias()));
  // secret-sharing inf as fp_op if_else does not support PUBLIC FPArrays
  FPArray inf = fp_op->input<double>(ALICE, x.size, INFINITY, x.m_bits, x.e_bits, false);
  gamma = fp_op->if_else(cond1, inf, gamma);

  // Case II
  BoolArray cond2 = fp_op->LT<double>(x, -126.0);
  // secret-sharing zero as fp_op if_else does not support PUBLIC FPArrays
  FPArray zero = fp_op->input<double>(ALICE, x.size, 0.0, x.m_bits, x.e_bits, false);
  gamma = fp_op->if_else(cond2, zero, gamma);

  // Case III
  BoolArray cond3 = fix->LE(x_e, -25 + x.e_bias());
  // secret-sharing one32 as fp_op if_else does not support PUBLIC FPArrays
  FPArray one32 = fp_op->input<double>(ALICE, x.size, 1.0, x.m_bits, x.e_bits, false);
  gamma = fp_op->if_else(cond3, one32, gamma);

  delete[] wrap_f;

  return gamma;
}

FPArray log_core(FPOp *fp_op, FixOp *fix, BoolOp *bool_op, const FPArray &x,
                 bool base_2 = false) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);

  BoolArray a = fix->EQ(x_e, -1 + x.e_bias());
  FixArray f_if = fix->sub(0, x_m);
  FixArray f_else = fix->sub(x_m, 1ULL << x.m_bits);
  FixArray f = fix->if_else(a, f_if, f_else);
  BoolArray f_eq_0 = fix->EQ(f, 0);
  FixArray delta_m = f;
  FixArray delta_e = fix->input(PUBLIC, x.size, uint64_t(0), x_e.signed_, x_e.ell, x_e.s);
  fp_op->normalize(delta_m, delta_e, x.m_bits - x.e_bias());
  FixArray delta_m_low_prec = delta_m;
  delta_m = fix->scale_up(delta_m, FP_INTMD_M_BITS + 1, FP_INTMD_M_BITS);
  delta_e = fix->if_else(a, fix->sub(delta_e, 1), delta_e);
  delta_e = fix->if_else(f_eq_0, 0, delta_e);
  FPArray delta = fp_op->input(x.party, x.size, all_0.data, f_eq_0.data,
          delta_m.data, delta_e.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  // if delta < 2^-5
  BoolArray cond1 = fix->LT(delta_e, -5 + delta.e_bias());
  FixArray idx_1 = get_idx_from_input(fix, delta_m_low_prec, delta_e, 0, 5,
                                      24 - delta.e_bias());
  FixArray idx_2 = get_idx_from_input(fix, delta_m_low_prec, delta_e, 4, 3,
                                      5 - delta.e_bias());
  vector<FPArray> theta_1, theta_2;
  if (base_2) {
    theta_1 = fp_op->GetCoeffs(log2_coeffs_1, log_knots_bits_1, idx_1, 19,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
    theta_2 = fp_op->GetCoeffs(log2_coeffs_2, log_knots_bits_2, idx_2, 35,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  } else {
    theta_1 = fp_op->GetCoeffs(ln_coeffs_1, log_knots_bits_1, idx_1, 19,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
    theta_2 = fp_op->GetCoeffs(ln_coeffs_2, log_knots_bits_2, idx_2, 35,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  }

  assert(theta_1.size() == theta_2.size());
  assert(theta_1.size() == 8);
  vector<FPArray> theta(theta_1.size());
  for (int i = 0; i < theta_1.size(); i++) {
    theta[i] = fp_op->if_else(cond1, theta_1[i], theta_2[i]);
  }
  for (int i = 0; i < 4; i++) {
    theta[i] = fp_op->if_else(a, theta[i], theta[i + 4]);
  }
  FPArray mu = fp_op->mul(theta[3], delta, false);
  mu = fp_op->add(mu, theta[2], true, true, false);
  mu = fp_op->mul(mu, delta, false);
  mu = fp_op->add(mu, theta[1], true, true, false);
  mu = fp_op->mul(mu, delta, false);
  mu = fp_op->add(mu, theta[0], true, true, false);

  // else (delta == 0, mu = 0.0)
  // secret-sharing zero as fp_op if_else does not support PUBLIC FPArrays
  FPArray zero = fp_op->input<double>(ALICE, x.size, 0.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  mu = fp_op->if_else(f_eq_0, zero, mu);

  FixArray N = fix->reduce(x_e, 8);
  FPArray beta;
  if (base_2) {
    beta = fp_op->LUT(log2_int_to_float, N, 6, FP_INTMD_E_BITS);
    for (int i = 0; i < x.size; i++) {
      beta.m[i] <<= (FP_INTMD_M_BITS - 6);
    }
    beta.m_bits = FP_INTMD_M_BITS;
  } else {
    beta = fp_op->LUT(ln_int_to_float, N, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  }
  FPArray neg_mu = fp_op->input(mu.party, mu.size, all_1.data, mu.z,
          mu.m, mu.e, mu.m_bits, mu.e_bits);

  // reduce precision to normal
  FPArray gamma =
      fp_op->if_else(a, neg_mu, fp_op->add(beta, mu, true, true, false));
  BoolArray gamma_s, gamma_z;
  FixArray gamma_m, gamma_e;
  tie(gamma_s, gamma_z, gamma_m, gamma_e) = fp_op->get_components(gamma);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(gamma_m, oflow_threshold);
  FixArray ret_m_if =
      fix->round_ties_to_even(gamma_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray ret_m = fix->if_else(rnd_no_oflow, ret_m_if, (1ULL << x.m_bits));
  FixArray ret_e = fix->if_else(rnd_no_oflow, gamma_e, fix->add(gamma_e, 1));

  FPArray ret = fp_op->input(x.party, x.size, gamma_s.data, gamma_z.data,
          ret_m.data, ret_e.data, x.m_bits, x.e_bits);

  return ret;
}

FPArray FPMath::log2(const FPArray &x) {
  return log_core(fp_op, fix, bool_op, x, true);
}

FPArray FPMath::ln(const FPArray &x) {
  return log_core(fp_op, fix, bool_op, x, false);
}

// returns mu
FPArray sinpi_core(FPOp *fp_op, FixOp *fix, BoolOp *bool_op, const FPArray &x,
                   const FixArray &m, FPArray &delta, BoolArray &a,
                   bool cos_invocation = false) {
  assert(x.party != PUBLIC && m.party != PUBLIC);
  assert(x.size == m.size);
  BoolArray all_0 = bool_op->input(ALICE, m.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, m.size, 1);

  FixArray f = fix->reduce(m, x.m_bits + 14);
  BoolArray msb_f = fix->LT(f, 0);
  uint8_t *wrap_f = new uint8_t[x.size];
  fix->aux->MSB_to_Wrap(f.data, msb_f.data, wrap_f, x.size, f.ell);
  FixArray a_fix = fix->truncate_reduce(m, x.m_bits + 14, wrap_f);
  a = fix->LSB(a_fix);
  f = fix->extend(f, x.m_bits + 15, msb_f.data);

  f = fix->if_else(msb_f, fix->sub(1ULL << (x.m_bits + 14), f), f);

  BoolArray f_eq_0 = fix->EQ(f, 0);
  FixArray delta_m = f;
  FixArray delta_e = fix->input(PUBLIC, x.size, uint64_t(0), true, x.e_bits + 2, 0);
  fp_op->normalize(delta_m, delta_e, x.m_bits + 14 - x.e_bias());
  delta_m = fix->truncate_reduce(delta_m, x.m_bits + 14 - FP_INTMD_M_BITS);
  delta_e = fix->if_else(f_eq_0, 0, delta_e);
  delta = fp_op->input(x.party, x.size, all_0.data, f_eq_0.data,
          delta_m.data, delta_e.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  // if delta < 2^-5
  BoolArray cond1 = fix->LT(delta_e, -5 + delta.e_bias());
  FixArray idx_1 =
      get_idx_from_input(fix, delta_m, delta_e, 0, 4, 14 - delta.e_bias());
  vector<FPArray> theta_1;
  if (cos_invocation) {
    theta_1 = fp_op->GetCoeffs(cos_coeffs_1, sin_knots_bits_1, idx_1, 9,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  } else {
    theta_1 = fp_op->GetCoeffs(sin_coeffs_1, sin_knots_bits_1, idx_1, 9,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  }

  // else if delta >= 2^-5
  BoolArray e_eq_neg_1 = fix->EQ(delta_e, delta.e_bias() - 1);
  FixArray idx_2 =
      get_idx_from_input(fix, delta_m, delta_e, 5, 2, 5 - delta.e_bias());
  idx_2 = fix->if_else(e_eq_neg_1, (1ULL << 7) - 1, idx_2);
  vector<FPArray> theta_2;
  if (cos_invocation) {
    theta_2 = fp_op->GetCoeffs(cos_coeffs_2, sin_knots_bits_2, idx_2, 34,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  } else {
    theta_2 = fp_op->GetCoeffs(sin_coeffs_2, sin_knots_bits_2, idx_2, 34,
                               FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  }
  assert(theta_1.size() == theta_2.size());
  assert(theta_1.size() == 3);
  vector<FPArray> theta(theta_1.size());
  for (int i = 0; i < theta_1.size(); i++) {
    theta[i] = fp_op->if_else(cond1, theta_1[i], theta_2[i]);
  }
  FPArray delta_sq = fp_op->mul(delta, delta, false);
  FPArray mu = fp_op->mul(theta[2], delta_sq, false);
  mu = fp_op->add(mu, theta[1], true, true, false);
  mu = fp_op->mul(mu, delta_sq, false);
  mu = fp_op->add(mu, theta[0], true, true, false);
  mu = fp_op->mul(mu, delta, false);

  delete[] wrap_f;

  return mu;
}

FPArray FPMath::sinpi(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);
  FixArray shift_amt = fix->add(x_e, 14 - x.e_bias());
  shift_amt.signed_ = false;
  FixArray m = fix->left_shift(x_m, shift_amt, x.m_bits + 14 + 1, 14 + 23, all_1.data);
  m.s += 14;

  BoolArray a;
  FPArray delta;
  FPArray mu = sinpi_core(fp_op, fix, bool_op, x, m, delta, a);

  BoolArray delta_s, delta_z;
  FixArray delta_m, delta_e;
  tie(delta_s, delta_z, delta_m, delta_e) = fp_op->get_components(delta);

  // else (delta < 2^-14, mu = pi * delta) and Special Case (x < 2^-14)
  BoolArray cond2 = fix->LT(delta_e, -14 + delta.e_bias());
  FPArray pi = fp_op->input<double>(PUBLIC, x.size, PI_DOUBLE,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  BoolArray cond3 = fix->LT(x_e, x.e_bias() - 14);
  // higher precision x
  FixArray x_m_prec = fix->scale_up(x_m, FP_INTMD_M_BITS + 1, FP_INTMD_M_BITS);
  FixArray x_e_prec = x_e;
  FPArray pos_x_prec = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m_prec.data, x_e_prec.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  FPArray y = fp_op->if_else(cond3, pos_x_prec, delta);
  BoolArray cond_23 = bool_op->if_else(cond3, cond3, cond2);
  mu = fp_op->if_else(cond_23, fp_op->mul(y, pi, true), mu);
  a = bool_op->if_else(cond3, 0, a);

  // Special Cases
  // Case I (Case II already done above)
  BoolArray cond4 = fix->GE(x_e, 23 + x.e_bias());
  // secret-sharing zero as fp_op if_else does not support PUBLIC FPArrays
  FPArray zero = fp_op->input<double>(ALICE, x.size, 0.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  FPArray gamma = fp_op->if_else(cond4, zero, mu);

  // reduce precision to normal
  BoolArray gamma_s, gamma_z;
  FixArray gamma_m, gamma_e;
  tie(gamma_s, gamma_z, gamma_m, gamma_e) = fp_op->get_components(gamma);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(gamma_m, oflow_threshold);
  FixArray ret_m_if =
      fix->round_ties_to_even(gamma_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray ret_m = fix->if_else(rnd_no_oflow, ret_m_if, (1ULL << x.m_bits));
  FixArray ret_e = fix->if_else(rnd_no_oflow, gamma_e, fix->add(gamma_e, 1));
  BoolArray ret_s = bool_op->XOR(a, x_s);

  FPArray ret = fp_op->input(x.party, x.size, ret_s.data, gamma_z.data,
          ret_m.data, ret_e.data, x.m_bits, x.e_bits);

  return ret;
}

FPArray FPMath::cospi(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);
  FixArray shift_amt = fix->add(x_e, 14 - x.e_bias());
  shift_amt.signed_ = false;
  FixArray m = fix->left_shift(x_m, shift_amt, x.m_bits + 14 + 1, 14 + 23, all_1.data);
  m = fix->add(m, 1ULL << (x.m_bits + 14 - 1));
  m.s += 14;

  BoolArray a;
  FPArray delta;
  FPArray mu = sinpi_core(fp_op, fix, bool_op, x, m, delta, a, true);

  BoolArray delta_s, delta_z;
  FixArray delta_m, delta_e;
  tie(delta_s, delta_z, delta_m, delta_e) = fp_op->get_components(delta);

  // else (delta < 2^-14, mu = pi * delta)
  BoolArray cond2 = fix->LT(delta_e, -14 + delta.e_bias());
  FPArray pi = fp_op->input<double>(PUBLIC, x.size, PI_DOUBLE,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  mu = fp_op->if_else(cond2, fp_op->mul(delta, pi, true), mu);
  for (int i = 0; i < x.size; i++) {
    mu.s[i] = a.data[i];
  }

  // Special Cases
  // secret-sharing one as fp_op add does not support PUBLIC FPArrays
  FPArray one = fp_op->input<float>(ALICE, x.size, 1.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);

  // Case I (x >= 2^23)
  BoolArray x_e_eq_23, x_e_gt_23;
  tie(x_e_gt_23, x_e_eq_23) = fix->MSB_and_zero_test(fix->sub(x.e_bias() + 23, x_e));
  BoolArray cond3 = bool_op->XOR(x_e_gt_23, x_e_eq_23);
  FPArray gamma = fp_op->if_else(cond3, one, mu);
  // If x_e == 23 and x_m & 1 == 1: output -1.0
  BoolArray lsb_x_m = fix->LSB(x_m);
  BoolArray case_1_s = bool_op->AND(lsb_x_m, x_e_eq_23);

  // Case II
  BoolArray cond4 = fix->LT(x_e, x.e_bias() - 14);
  gamma = fp_op->if_else(cond4, one, gamma);

  // reduce precision to normal
  BoolArray gamma_s, gamma_z;
  FixArray gamma_m, gamma_e;
  tie(gamma_s, gamma_z, gamma_m, gamma_e) = fp_op->get_components(gamma);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(gamma_m, oflow_threshold);
  FixArray ret_m_if =
      fix->round_ties_to_even(gamma_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray ret_m = fix->if_else(rnd_no_oflow, ret_m_if, (1ULL << x.m_bits));
  FixArray ret_e = fix->if_else(rnd_no_oflow, gamma_e, fix->add(gamma_e, 1));
  BoolArray ret_s = bool_op->XOR(case_1_s, gamma_s);

  FPArray ret = fp_op->input(x.party, x.size, ret_s.data, gamma_z.data,
          ret_m.data, ret_e.data, x.m_bits, x.e_bits);

  return ret;
}


FixArray FPMath::exp_lut(const FixArray &x) {
    int ell   = x.ell;     // 总位宽
    int scale = x.s;       // 小数精度 (定点缩放因子)
    int N     = x.size;    // batch 大小

    // Step1: 拆分整数/小数
    FixArray x_int = fix->truncate_reduce(x, scale);        // 去掉小数部分 (向下取整)
    x_int          = fix->scale_up(x_int, ell, scale);      // 补齐小数位宽
    FixArray x_dec = fix->sub(x, x_int);                    // 小数部分

    // Step2: 提取明文整数部分并裁剪范围 [-10, 0]
    std::vector<int> x_int_plain(N);
    for (int i = 0; i < N; i++) {
        // 原始定点数存储在 data[] 中，右移 scale 得到整数
        int64_t raw_val = (int64_t)x.data[i];  
        int int_val     = (int)(raw_val >> scale);
        if (int_val > 0) int_val = 0;
        if (int_val < -10) int_val = -10;
        x_int_plain[i] = int_val;
    }

    // Step3: 明文查表 e^int
    // LUT: e^0, e^-1, ..., e^-10
    std::vector<double> lut_value = {
        1.0,
        0.3678794412,  // e^-1
        0.1353352832,  // e^-2
        0.0497870684,  // e^-3
        0.0183156389,  // e^-4
        0.0067379470,  // e^-5
        0.0024787522,  // e^-6
        0.00091188197, // e^-7
        0.00033546263, // e^-8
        0.00012340981, // e^-9
        0.00004539993  // e^-10
    };

    // 转成定点数形式 (等比例缩放)
    std::vector<uint64_t> exp_int_data(N);
    for (int i = 0; i < N; i++) {
        double v = lut_value[-x_int_plain[i]];                // index = -int_val
        exp_int_data[i] = (uint64_t)std::llround(v * (1LL << scale));
    }

    // 封装成公共 FixArray (PUBLIC 注入，不走 MPC)
    FixArray exp_int = fix->input(PUBLIC, N, exp_int_data.data(),
                                  true, ell, scale);

    // Step4: 小数部分安全多项式近似 (比如泰勒二阶)
    FixArray exp_dec = fix->poly2(x_dec);

    // Step5: 合并整数/小数部分
    FixArray result = fix->mul(exp_int, exp_dec, ell + scale);
    result = fix->truncate_reduce(result, scale);

    // 返回结果，以及整数部分 (便于调试/复用)
    return result;
}


FixArray FPMath::exp_lut2(const FixArray &x) {
    int ell   = x.ell;     // 总位宽
    int scale = x.s;       // 小数精度 (定点缩放因子)
    int N     = x.size;    // batch 大小

    // Step1: 拆分整数/小数
    FixArray x_int = fix->truncate_reduce(x, scale);        // 去掉小数部分 (向下取整)
    x_int          = fix->scale_up(x_int, ell, scale);      // 补齐小数位宽
    FixArray x_dec = fix->sub(x, x_int);                    // 小数部分

    // Step2: 提取明文整数部分并裁剪范围 [-10, 0]
    std::vector<int> x_int_plain(N);
    for (int i = 0; i < N; i++) {
        // 原始定点数存储在 data[] 中，右移 scale 得到整数
        int64_t raw_val = (int64_t)x.data[i];  
        int int_val     = (int)(raw_val >> scale);
        if (int_val > 10) int_val = 10;
        if (int_val < -8) int_val = -8;
        x_int_plain[i] = int_val;
    }

    // Step3: 明文查表 e^int
    // LUT: e^0, e^-1, ..., e^-10
    std::vector<double> lut_value = {
	0.00033546263,  // e^-8
    0.00091188197,  // e^-7
    0.00247875218,  // e^-6
    0.00673794700,  // e^-5
    0.01831563889,  // e^-4
    0.04978706837,  // e^-3
    0.13533528324,  // e^-2
    0.36787944117,  // e^-1
    1.00000000000,  // e^0
    2.71828182846,  // e^1
    7.38905609893,  // e^2
    20.08553692319, // e^3
    54.59815003314, // e^4
    148.413159041,  // e^5
    403.428793493,  // e^6
    1096.63315843,  // e^7
    2980.95798704,  // e^8
    8103.08392758,  // e^9
    22026.4657948  // e^10
    };

    // 转成定点数形式 (等比例缩放)
    std::vector<uint64_t> exp_int_data(N);
    for (int i = 0; i < N; i++) {
        double v = lut_value[-x_int_plain[i]];                // index = -int_val
        exp_int_data[i] = (uint64_t)std::llround(v * (1LL << scale));
    }

    // 封装成公共 FixArray (PUBLIC 注入，不走 MPC)
    FixArray exp_int = fix->input(PUBLIC, N, exp_int_data.data(),
                                  true, ell, scale);

    // Step4: 小数部分安全多项式近似 (比如泰勒二阶)
    FixArray exp_dec = fix->poly2(x_dec);

    // Step5: 合并整数/小数部分
    FixArray result = fix->mul(exp_int, exp_dec, ell + scale);
    result = fix->truncate_reduce(result, scale);

    // 返回结果，以及整数部分 (便于调试/复用)
    return result;
}
std::tuple<FixArray, FixArray> FPMath::exp4(const FixArray &x){

  /*
  l = np.floor((x / -math.log(2)))
  p = x + l*math.log(2)
  fp = poly(p)
  return fp / (2**l)
  */

  // print_fix(x);

  int ell = x.ell;
  int scale = x.s;

  // All 0 and all 1 array for msb arg
  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);
  

  // ln2
  FixArray ln2 = fix->input(PUBLIC, x.size, uint64_t(2839), true, ell, scale);
  // print_fix(ln2);

  // inverse of negative ln2
  FixArray inl = fix->input(PUBLIC, x.size, uint64_t(-5909), true, ell, scale);
  // print_fix(inl);

  // x / -math.log(2)
  // Truncate to original scale and bitlength
  FixArray x_inl = fix->mul(x, inl, ell + scale);
  // Optimization: local truncation
  x_inl =  fix->truncate_reduce(x_inl, scale);
  // x_inl =  fix->reduce(x_inl, ell);
  // print_fix(x_inl);

  // Get the integer part and scale back
  FixArray l_short = fix->truncate_reduce(x_inl, scale);
  FixArray l_short_raw = l_short;
  FixArray l = fix->scale_up(l_short, ell, scale);

  // l*math.log(2)
  FixArray l_ln2 = fix->mul(l, ln2, ell+scale, all_0.data, all_0.data);
  l_ln2 =  fix->truncate_reduce(l_ln2, scale);
  // l_ln2 =  fix->reduce(l_ln2, ell);

  // Get the decimal part  
  FixArray p = fix->add(x, l_ln2);
  // Optimization: We don't need that much bit as p \in (-ln2, 0])
  p = fix->reduce(p, scale + 2);

  // Polynomial fit
  FixArray poly_p = fix->poly1(p);
  poly_p = fix->extend(poly_p, ell, all_0.data);

  l_short.signed_ = false;
  // Optimization: The polynomial result is within [0, ~0.7)
  // Thus the upper bound of shift is scale + 1

  FixArray bound = fix->input(PUBLIC, l_short.size, 13, false, l_short.ell, 0);
  BoolArray gt_bound = fix->GT(l_short, bound);
  l_short = fix->if_else(gt_bound, bound, l_short);

  FixArray ret = fix->right_shift(poly_p, l_short, scale + 1, all_1.data);

  return make_tuple(ret, l_short_raw);
}

FPArray FPMath::exp3(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));

  BoolArray x_s, x_z;

  FixArray x_m, x_e;

  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  FPArray x_copy = fp_op->input(x.party, x.size, x.s, x.z, x.m, x.e, x.m_bits, x.e_bits);

  FPArray nli = fp_op->input<float>(PUBLIC, x.size, NEG_LOGE2_INV, x.m_bits, x.e_bits);

  FPArray x_nli = fp_op->mul(x_copy, nli);

  FixArray shift_amt = fix->sub(x_m.ell + x.e_bias() - 1, x_e);

  shift_amt.signed_ = false;

  FixArray x_m_int_ws = fix->right_shift(x_m, shift_amt, x_m.ell, all_1.data);
  FixArray x_m_int = fix->left_shift(x_m_int_ws, shift_amt, x_m.ell, x_m.ell, all_0.data);

  print_fix(shift_amt);

  print_fix(x_e);

  print_fix(x_m);

  print_fix(x_m_int_ws);

  print_fix(x_m_int);

  print_fp(x_copy);

  return x_copy;
}

FPArray FPMath::exp(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  // sign zero
  BoolArray x_s, x_z;

  // mantissa exponent
  FixArray x_m, x_e;

  // returns (BoolArray x.s, BoolArray x.z, FixArray x.m, FixArray x.e)
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, x.size, 1);

  FPArray pos_x = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m.data, x_e.data, x.m_bits, x.e_bits);
  BoolArray pos_x_ge_LOGE2 = fp_op->GE<double>(pos_x, double(LOGE2));
  FixArray shift_amt = fix->add(x_e, 1 - x.e_bias());

  std::cout << "1 - x.ebias()" << x.e_bias() << std::endl;
  shift_amt.signed_ = false;
  FixArray m = fix->left_shift(x_m, shift_amt, x.m_bits + 10, 9, all_1.data);
  m.s += 1;
  FixArray N = fix->mul(m, uint64_t(LOG2E * (1ULL << FP_INTMD_M_BITS)),
                        x.m_bits + FP_INTMD_M_BITS + 11, all_0.data);
  N.s += FP_INTMD_M_BITS;
  N = fix->round_ties_to_even(N, x.m_bits + 1 + FP_INTMD_M_BITS);

  FixArray f = fix->mul(N, uint64_t(LOGE2 * (1ULL << (FP_INTMD_M_BITS + 7))),
                        FP_INTMD_M_BITS + 8, all_0.data);
  f.s = FP_INTMD_M_BITS + 7;
  f = fix->sub(fix->scale_up(m, FP_INTMD_M_BITS + 8, FP_INTMD_M_BITS + 7), f);
  BoolArray msb_f, zero_f;
  tie(msb_f, zero_f) = fix->MSB_and_zero_test(f);
  BoolArray a = bool_op->XOR(x_s, msb_f);
  f = fix->if_else(msb_f, fix->mul(f, -1), f);
  N = fix->if_else(x_s, fix->mul(N, -1), N);
  N.signed_ = true;

  FixArray delta_m = f;
  FixArray delta_e = fix->input(PUBLIC, x.size, uint64_t(0), x_e.signed_, x_e.ell, x_e.s);
  fp_op->normalize(delta_m, delta_e, FP_INTMD_M_BITS + 7 - x.e_bias());
  delta_m = fix->truncate_reduce(delta_m, 7);
  delta_e = fix->if_else(zero_f, 0, delta_e);
  FPArray delta = fp_op->input(x.party, x.size, all_0.data, zero_f.data,
          delta_m.data, delta_e.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  // if x < LOGE2, a = x_s, N = 0, delta = x
  FixArray x_m_prec = fix->scale_up(x_m, FP_INTMD_M_BITS + 1, FP_INTMD_M_BITS);
  FixArray x_e_prec = x_e;
  FPArray pos_x_prec = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m_prec.data, x_e_prec.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  a = bool_op->if_else(pos_x_ge_LOGE2, a, x_s);
  N = fix->if_else(pos_x_ge_LOGE2, N, 0);
  delta = fp_op->if_else(pos_x_ge_LOGE2, delta, pos_x_prec);
  BoolArray delta_s, delta_z;
  tie(delta_s, delta_z, delta_m, delta_e) = fp_op->get_components(delta);

  // if delta < 2^-14, mu = 1.0
  BoolArray cond = fix->LT(delta_e, -24 + delta.e_bias());
  // secret-sharing one as fp_op add does not support PUBLIC FPArrays
  FPArray one = fp_op->input<float>(ALICE, x.size, 1.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  FPArray mu_if = one;

  // else evaluate spline
  // if delta < 2^-6
  BoolArray cond1 = fix->LT(delta_e, -6 + delta.e_bias());
  FixArray idx_1 =
      get_idx_from_input(fix, delta_m, delta_e, 0, 5, 24 - delta.e_bias());
  vector<FPArray> theta_1 =
      fp_op->GetCoeffs(exp_coeffs_1, exp_knots_bits_1, idx_1, 18,
                       FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  // else if delta >= 2^-6
  FixArray idx_2 =
      get_idx_from_input(fix, delta_m, delta_e, 5, 3, 6 - delta.e_bias());
  vector<FPArray> theta_2 =
      fp_op->GetCoeffs(exp_coeffs_2, exp_knots_bits_2, idx_2, 44,
                       FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  assert(theta_1.size() == theta_2.size());
  assert(theta_1.size() == 6);
  vector<FPArray> theta(theta_1.size());
  for (int i = 0; i < theta_1.size(); i++) {
    theta[i] = fp_op->if_else(cond1, theta_1[i], theta_2[i]);
  }
  for (int i = 0; i < 3; i++) {
    theta[i] = fp_op->if_else(a, theta[i + 3], theta[i]);
  }
  FPArray mu_else = fp_op->mul(theta[2], delta, false);
  mu_else = fp_op->add(mu_else, theta[1], true, true, false);
  mu_else = fp_op->mul(mu_else, delta, false);
  mu_else = fp_op->add(mu_else, theta[0], true, true, false);
  FPArray mu = fp_op->if_else(cond, mu_if, mu_else);

  BoolArray mu_s, mu_z;
  FixArray mu_m, mu_e;
  tie(mu_s, mu_z, mu_m, mu_e) = fp_op->get_components(mu);
  N = fix->extend(N, x.e_bits + 2);
  FixArray gamma_e = fix->add(mu_e, N);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(mu_m, oflow_threshold);
  FixArray gamma_m_if =
      fix->round_ties_to_even(mu_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray gamma_m = fix->if_else(rnd_no_oflow, gamma_m_if, (1ULL << x.m_bits));
  gamma_e = fix->if_else(rnd_no_oflow, gamma_e, fix->add(gamma_e, 1));
  FPArray gamma = fp_op->input(x.party, x.size, all_0.data, all_0.data,
          gamma_m.data, gamma_e.data, x.m_bits, x.e_bits);

  // Special Cases
  // Case I
  BoolArray cond2 = fp_op->GE(x, 88.72283172607421875);
  // secret-sharing inf as fp_op if_else does not support PUBLIC FPArrays
  FPArray inf = fp_op->input<double>(ALICE, x.size, INFINITY, x.m_bits, x.e_bits, false);
  gamma = fp_op->if_else(cond2, inf, gamma);

  // Case II
  BoolArray cond3 = fp_op->LT(x, -87.33654022216796875);
  // secret-sharing zero as fp_op if_else does not support PUBLIC FPArrays
  FPArray zero = fp_op->input<double>(ALICE, x.size, 0.0, x.m_bits, x.e_bits, false);
  gamma = fp_op->if_else(cond3, zero, gamma);

  // Case III
  BoolArray cond4 = fix->LE(x_e, -25 + x.e_bias());
  // secret-sharing one32 as fp_op if_else does not support PUBLIC FPArrays
  FPArray one32 = fp_op->input<double>(ALICE, x.size, 1.0, x.m_bits, x.e_bits, false);
  gamma = fp_op->if_else(cond4, one32, gamma);

  return gamma;
}

FPArray FPMath::erf(const FPArray &x) {
  assert(x.party != PUBLIC);
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);
  assert(FP_INTMD_E_BITS == x.e_bits);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);

  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));

  FixArray x_m_prec = fix->scale_up(x_m, FP_INTMD_M_BITS + 1, FP_INTMD_M_BITS);
  FPArray delta = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m_prec.data, x_e.data, FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  BoolArray delta_s, delta_z;
  FixArray delta_m, delta_e;
  tie(delta_s, delta_z, delta_m, delta_e) = fp_op->get_components(delta);

  // if delta < 1
  BoolArray cond1 = fix->LT(delta_e, delta.e_bias());
  BoolArray e_lt_neg_5 = fix->LT(delta_e, delta.e_bias() - 5);
  FixArray delta_e_prime =
      fix->if_else(e_lt_neg_5, delta.e_bias() - 6, delta_e);
  FixArray idx_1 =
      get_idx_from_input(fix, delta_m, delta_e_prime, 3, 3, 6 - delta.e_bias());
  vector<FPArray> theta_1;
  theta_1 = fp_op->GetCoeffs(erf_coeffs_1, erf_knots_bits_1, idx_1, 24,
                             FP_INTMD_M_BITS, FP_INTMD_E_BITS);
  // else if delta >= 1
  FixArray idx_2 =
      get_idx_from_input(fix, delta_m, delta_e, 5, 1, -1 * delta.e_bias());
  vector<FPArray> theta_2;
  theta_2 = fp_op->GetCoeffs(erf_coeffs_2, erf_knots_bits_2, idx_2, 46,
                             FP_INTMD_M_BITS, FP_INTMD_E_BITS);

  assert(theta_1.size() == theta_2.size());
  assert(theta_1.size() == 4);
  vector<FPArray> theta(theta_1.size());
  for (int i = 0; i < theta_1.size(); i++) {
    theta[i] = fp_op->if_else(cond1, theta_1[i], theta_2[i]);
  }
  FPArray mu = fp_op->mul(theta[3], delta, false);
  mu = fp_op->add(mu, theta[2], true, true, false);
  mu = fp_op->mul(mu, delta, false);
  mu = fp_op->add(mu, theta[1], true, true, false);
  mu = fp_op->mul(mu, delta, false);
  mu = fp_op->add(mu, theta[0], true, true, false);

  // Special Cases
  // Special Case I (x < 2^-12)
  BoolArray cond2 = fix->LT(x_e, -12 + delta.e_bias());
  FPArray two_inv_sqrt_pi = fp_op->input<double>(PUBLIC, x.size, TWO_INV_SQRT_PI,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  mu = fp_op->if_else(cond2, fp_op->mul(delta, two_inv_sqrt_pi, true), mu);

  // Case I (x >= 3.875)
  FPArray pos_x = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m.data, x_e.data, x.m_bits, x.e_bits);
  BoolArray cond3 = fp_op->GE(pos_x, 3.875);
  // secret-sharing one as fp_op add does not support PUBLIC FPArrays
  FPArray one = fp_op->input<double>(ALICE, x.size, 1.0,
          FP_INTMD_M_BITS, FP_INTMD_E_BITS, false);
  FPArray gamma = fp_op->if_else(cond3, one, mu);

  // reduce precision to normal
  BoolArray gamma_s, gamma_z;
  FixArray gamma_m, gamma_e;
  tie(gamma_s, gamma_z, gamma_m, gamma_e) = fp_op->get_components(gamma);
  uint64_t oflow_threshold = (1ULL << (FP_INTMD_M_BITS + 1)) -
                             (1ULL << (FP_INTMD_M_BITS - x.m_bits - 1));
  BoolArray rnd_no_oflow = fix->LT(gamma_m, oflow_threshold);
  FixArray ret_m_if =
      fix->round_ties_to_even(gamma_m, FP_INTMD_M_BITS - x.m_bits);
  FixArray ret_m = fix->if_else(rnd_no_oflow, ret_m_if, (1ULL << x.m_bits));
  FixArray ret_e = fix->if_else(rnd_no_oflow, gamma_e, fix->add(gamma_e, 1));
  BoolArray ret_s = x_s;

  FPArray ret = fp_op->input(x.party, x.size, ret_s.data, gamma_z.data,
          ret_m.data, ret_e.data, x.m_bits, x.e_bits);

  return ret;
}


FPArray FPMath::sigmoid_fp32(const FPArray &x) {
  assert(x.party != PUBLIC) ;
  assert(x.m_bits == 23) ;
  assert(x.e_bits == 8) ;

  FPArray one_flat = fp_op->input<float>(ALICE, x.size, (float)1.0, x.m_bits, x.e_bits) ;

  return fp_op->div(
    one_flat, 
    fp_op->add(
      one_flat,
      this->exp(fp_op->flip_sign(x))
    )
  ) ;
}

FPArray FPMath::sigmoid_bf16(const FPArray &x) {
  // currently only supports bfloat16
  assert(x.party != PUBLIC);
  assert(x.m_bits == 7);
  assert(x.e_bits == 8);

  BoolArray x_s, x_z;
  FixArray x_m, x_e;
  tie(x_s, x_z, x_m, x_e) = fp_op->get_components(x);
  BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));

  x_e.signed_ = false;
  // idx = s || (e + 8) mod 2^4 || m - 2^7 = 12 bits
  FixArray idx_hi_s = fix->scale_up(fix->mul(fix->B2A(x_s, false, 5), 1ULL << 4), x.m_bits + 5, x.m_bits);
  // only 4 bits of exponent are needed
  FixArray idx_hi = fix->scale_up(fix->reduce(fix->add(x_e, 8 - x.e_bias()), 5), x.m_bits + 5, x.m_bits);
  FixArray idx_lo = fix->extend(fix->sub(x_m, 1ULL << x.m_bits), x.m_bits + 5);
  FixArray idx = fix->add(idx_hi_s, fix->add(idx_hi, idx_lo));
  // print_fix(idx);

  // all outputs are positive, so ignoring sign bit
  FixArray y_int = fix->LUT(sigmoid_bfloat16, idx, false, 15, x.m_bits);
  FixArray y_m = fix->add(fix->extend(fix->reduce(y_int, x.m_bits), x.m_bits + 1), 1ULL << x.m_bits);
  FixArray y_e = fix->truncate_reduce(y_int, x.m_bits);
  y_e.signed_ = true;
  y_e = fix->extend(y_e, x.e_bits + 2);
  FPArray y = fp_op->input(x.party, x.size, all_0.data, all_0.data,
          y_m.data, y_e.data, x.m_bits, x.e_bits);
  // print_fix(y_int);
  // print_fp(y);
  // print_fp(x);

  FPArray pos_x = fp_op->input(x.party, x.size, all_0.data, x_z.data,
          x_m.data, x_e.data, x.m_bits, x.e_bits);
  BoolArray cond1 = fp_op->LT(pos_x, 93.0);
  BoolArray cond2 = fix->GE(x_e, -8 + x.e_bias());
  FPArray zero_fp = fp_op->input<float>(ALICE, x.size, 0.0, x.m_bits, x.e_bits, false);
  FPArray y_ = fp_op->if_else(x_s, zero_fp, 1.0);

  y = fp_op->if_else(cond1, y, y_);
  y = fp_op->if_else(cond2, y, 0.5);
  return y;
}

FPArray FPMath::tanh_bf16(const FPArray &x) {
  assert(x.party != PUBLIC) ;
  assert(x.m_bits == 7);
  assert(x.e_bits == 8);

  int sz = x.size ;
  int m_bits, e_bits ;

  m_bits = x.m_bits ;
  e_bits = x.e_bits ;

  FPArray one_flat = fp_op->input<float>(ALICE, sz, (float)1.0, m_bits, e_bits) ;
  FPArray two_flat = fp_op->input<float>(ALICE, sz, (float)2.0, m_bits, e_bits) ;

  FPArray sig = fp_op->mul(
    this->sigmoid_bf16(
      fp_op->mul(two_flat, x)
    ), 
    two_flat) ;
  return fp_op->sub(sig, one_flat) ;
}

FPArray FPMath::tanh_fp32(const FPArray &x) {
  assert(x.party != PUBLIC) ;
  assert(x.m_bits == 23);
  assert(x.e_bits == 8);

  int sz = x.size ;
  int m_bits, e_bits ;

  m_bits = x.m_bits ;
  e_bits = x.e_bits ;

  FPArray one_flat = fp_op->input<float>(ALICE, sz, (float)1.0, m_bits, e_bits) ;
  FPArray two_flat = fp_op->input<float>(ALICE, sz, (float)2.0, m_bits, e_bits) ;

  FPArray sig = fp_op->mul(
    this->sigmoid_fp32(
      fp_op->mul(two_flat, x)
    ), 
    two_flat) ;
  return fp_op->sub(sig, one_flat) ;
}

vector<FPArray> FPMath::softmax_beacon(const vector<FPArray>& x) {
  int N = x.size();
  int n = x[0].size;
  int m_bits = x[0].m_bits;
  int e_bits = x[0].e_bits;
  assert(m_bits > 0);
  for(int i = 1; i < N; i++) {
    assert(x[i].party != PUBLIC);
    assert(x[i].m_bits == m_bits);
    assert(x[i].e_bits == e_bits);
    assert(x[i].size == n);
  }
  if (x[0].m_bits == BFLOAT16_M_BITS && x[0].e_bits == BFLOAT16_E_BITS) {
    vector<FPArray> y(x.size());
    for (int i = 0; i < x.size(); i++) {
      y[i] = fp_op->bfloat16_to_FP32(x[i]);
    }
    y = softmax_beacon(y);
    FPArray y_concat = concat(y);
    y_concat = fp_op->FP32_to_bfloat16(y_concat);
    for (int i = 0; i < x.size(); i++) {
      y[i] = y_concat.subset(i*x[0].size, (i+1)*x[0].size);
    }
    return y;
  }
  FPArray x_max = fp_op->max(x);
  FPArray x_max_flat(party, N*n, m_bits, e_bits);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      x_max_flat.s[i*n + j] = x_max.s[i];
      x_max_flat.z[i*n + j] = x_max.z[i];
      x_max_flat.m[i*n + j] = x_max.m[i];
      x_max_flat.e[i*n + j] = x_max.e[i];
    }
  }
  FPArray x_flat = concat(x);
  FPArray shifted_x_flat = fp_op->flip_sign(fp_op->sub(x_max_flat, x_flat, false, true, true));
  FPArray e_x_flat = this->exp(shifted_x_flat);

  vector<FPArray> e_x(N);
  for (int i = 0; i < N; i++) {
    e_x[i] = FPArray(party, n, m_bits, e_bits);
    memcpy(e_x[i].s, e_x_flat.s + i*n, n*sizeof(uint8_t));
    memcpy(e_x[i].z, e_x_flat.z + i*n, n*sizeof(uint8_t));
    memcpy(e_x[i].m, e_x_flat.m + i*n, n*sizeof(uint64_t));
    memcpy(e_x[i].e, e_x_flat.e + i*n, n*sizeof(uint64_t));
  }
  vector<FPArray> e_x_tr(n);
  for (int i = 0; i < n; i++) {
    e_x_tr[i] = FPArray(party, N, m_bits, e_bits);
    for (int j = 0; j < N; j++) {
      e_x_tr[i].s[j] = e_x[j].s[i];
      e_x_tr[i].z[j] = e_x[j].z[i];
      e_x_tr[i].m[j] = e_x[j].m[i];
      e_x_tr[i].e[j] = e_x[j].e[i];
    }
  }
  FPArray sum_e_x = fp_op->vector_sum(e_x);
  vector<FPArray> ret_tr = fp_op->div(e_x_tr, sum_e_x, false);
  vector<FPArray> ret(N);
  for (int i = 0; i < N; i++) {
    ret[i] = FPArray(party, n, m_bits, e_bits);
    for (int j = 0; j < n; j++) {
      ret[i].s[j] = ret_tr[j].s[i];
      ret[i].z[j] = ret_tr[j].z[i];
      ret[i].m[j] = ret_tr[j].m[i];
      ret[i].e[j] = ret_tr[j].e[i];
    }
  }
  return ret;
}

vector<FPArray> FPMath::softmax_secfloat(const vector<FPArray>& x) {
  int N = x.size();
  int n = x[0].size;
  int m_bits = x[0].m_bits;
  int e_bits = x[0].e_bits;
  assert(m_bits > 0);
  for(int i = 1; i < N; i++) {
    assert(x[i].party != PUBLIC);
    assert(x[i].m_bits == m_bits);
    assert(x[i].e_bits == e_bits);
    assert(x[i].size == n);
  }
  FPArray x_max = fp_op->max(x);
  FPArray x_max_flat(party, N*n, m_bits, e_bits);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      x_max_flat.s[i*n + j] = x_max.s[i];
      x_max_flat.z[i*n + j] = x_max.z[i];
      x_max_flat.m[i*n + j] = x_max.m[i];
      x_max_flat.e[i*n + j] = x_max.e[i];
    }
  }

  FPArray x_flat = concat(x);
  FPArray shifted_x_flat = fp_op->flip_sign(fp_op->sub(x_max_flat, x_flat, false, true, true));

  FPArray e_x_flat = this->exp(shifted_x_flat);

  vector<FPArray> e_x_tr(n);
  for (int i = 0; i < n; i++) {
    e_x_tr[i] = FPArray(party, N, m_bits, e_bits);
    for (int j = 0; j < N; j++) {
      e_x_tr[i].s[j] = e_x_flat.s[j*n + i];
      e_x_tr[i].z[j] = e_x_flat.z[j*n + i];
      e_x_tr[i].m[j] = e_x_flat.m[j*n + i];
      e_x_tr[i].e[j] = e_x_flat.e[j*n + i];
    }
  }
  FPArray sum_e_x;
  {
    vector<FPArray> tmp = e_x_tr;
    int num_adds_old = n; int num_adds_curr = n/2;
    while(num_adds_old > 1) {
      int odd_num_adds = num_adds_old & 1;
      vector<FPArray> lhs(num_adds_curr); vector<FPArray> rhs(num_adds_curr);
      for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        lhs[j/2] = tmp[j]; rhs[j/2] = tmp[j+1];
      }
      FPArray lhs_concat = concat(lhs);
      FPArray rhs_concat = concat(rhs);
      lhs_concat = fp_op->add(lhs_concat, rhs_concat);
      for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        tmp[odd_num_adds + (j/2)] = lhs_concat.subset((j/2)*N, (j/2)*N + N);
      }
      num_adds_old = num_adds_curr + odd_num_adds;
      num_adds_curr = num_adds_old/2;
    }
    sum_e_x = tmp[0];
  }
  FPArray sum_e_x_replicated(party, N*n, m_bits, e_bits);
  for(int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      sum_e_x_replicated.s[i*n + j] = sum_e_x.s[i];
      sum_e_x_replicated.z[i*n + j] = sum_e_x.z[i];
      sum_e_x_replicated.m[i*n + j] = sum_e_x.m[i];
      sum_e_x_replicated.e[i*n + j] = sum_e_x.e[i];
    }
  }

  FPArray ret_flat = fp_op->div(e_x_flat, sum_e_x_replicated);
  vector<FPArray> ret(N);
  for (int i = 0; i < N; i++) {
    ret[i] = FPArray(party, n, m_bits, e_bits);
    memcpy(ret[i].s, ret_flat.s + i*n, n*sizeof(uint8_t));
    memcpy(ret[i].z, ret_flat.z + i*n, n*sizeof(uint8_t));
    memcpy(ret[i].m, ret_flat.m + i*n, n*sizeof(uint64_t));
    memcpy(ret[i].e, ret_flat.e + i*n, n*sizeof(uint64_t));
  }
  return ret;
}

std::tuple<vector<FixArray>, FixArray> FPMath::softmax_fix(const vector<FixArray>& x) {
  // std::cout << "Entering softmax fix" << std::endl;
  int N = x.size();
  int n = x[0].size;
  int ell = x[0].ell;
  int s = x[0].s;

  // for (int i = 0; i < N; i++){
  //   print_fix(x[i]);
  // }

  bool signed_ = x[0].signed_;
  // assert(m_bits > 0);
  for(int i = 1; i < N; i++) {
    assert(x[i].party != PUBLIC);
    assert(x[i].ell == ell);
    assert(x[i].s == s);
    assert(x[i].size == n);
  }
  FixArray x_max = fix->max(x);
  // x_max = fix->add(x_max, 1);
  FixArray x_max_flat(party, N*n, signed_, ell, s);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      x_max_flat.data[i*n + j] = x_max.data[i];
    }
  }

  // FixArray x_max_flat = fix->input(PUBLIC, N*n, 10<<s, signed_, ell, s);
  // print_fix(x_max_flat);
  // assert(0);

  FixArray x_flat = concat(x);
  FixArray shifted_x_flat = fix->sub(x_flat, x_max_flat);

  FixArray e_x_flat;
  FixArray l_short;

  tie(e_x_flat, l_short) = exp4(shifted_x_flat);
  // FixArray e_x_flat = shifted_x_flat;

  int exp_ell = 19;
  e_x_flat = fix->reduce(e_x_flat, exp_ell);

  vector<FixArray> e_x_tr(n);
  for (int i = 0; i < n; i++) {
    e_x_tr[i] = FixArray(party, N, signed_, exp_ell, s);
    for (int j = 0; j < N; j++) {
      e_x_tr[i].data[j] = e_x_flat.data[j*n + i];
    }
  }
  FixArray sum_e_x;
  {
    vector<FixArray> tmp = e_x_tr;
    int num_adds_old = n; int num_adds_curr = n/2;
    while(num_adds_old > 1) {
      int odd_num_adds = num_adds_old & 1;
      vector<FixArray> lhs(num_adds_curr); vector<FixArray> rhs(num_adds_curr);
      for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        lhs[j/2] = tmp[j]; rhs[j/2] = tmp[j+1];
      }
      FixArray lhs_concat = concat(lhs);
      FixArray rhs_concat = concat(rhs);
      lhs_concat = fix->add(lhs_concat, rhs_concat);
      for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        tmp[odd_num_adds + (j/2)] = lhs_concat.subset((j/2)*N, (j/2)*N + N);
      }
      num_adds_old = num_adds_curr + odd_num_adds;
      num_adds_curr = num_adds_old/2;
    }
    sum_e_x = tmp[0];
  }
  
  sum_e_x.signed_ = false;
  FixArray ret_flat = fix->div_batch(e_x_flat, sum_e_x, n ,exp_ell, s);

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  ret_flat = fix->extend(ret_flat, ell);

  vector<FixArray> ret(N);
  for (int i = 0; i < N; i++) {
    ret[i] = FixArray(party, n, signed_, ell, s);
    memcpy(ret[i].data, ret_flat.data + i*n, n*sizeof(uint64_t));
  }
  return make_tuple(ret, l_short);
}

//lms_softmax
std::tuple<vector<FixArray>, FixArray> FPMath::softmax_fix2(const vector<FixArray>& x) {
  using Clock = std::chrono::high_resolution_clock;




  int N = x.size();
  int n = x[0].size;
  int ell = x[0].ell;
  int s = x[0].s;
  bool signed_ = x[0].signed_;

  for(int i = 1; i < N; i++) {
    assert(x[i].party != PUBLIC);
    assert(x[i].ell == ell);
    assert(x[i].s == s);
    assert(x[i].size == n);
  }



  FixArray x_max = fix->max(x);
  FixArray x_max_flat(party, N*n, signed_, ell, s);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      x_max_flat.data[i*n + j] = x_max.data[i];
    }
  }



  FixArray x_flat = concat(x);
  FixArray shifted_x_flat = fix->sub(x_flat, x_max_flat);



  FixArray e_x_flat;

  e_x_flat = exp_lut(shifted_x_flat);



  int exp_ell = 19;
  e_x_flat = fix->reduce(e_x_flat, exp_ell);
  vector<FixArray> e_x_tr(n);
  for (int i = 0; i < n; i++) {
    e_x_tr[i] = FixArray(party, N, signed_, exp_ell, s);
    for (int j = 0; j < N; j++) {
      e_x_tr[i].data[j] = e_x_flat.data[j*n + i];
    }
  }


  // ---- Step 5: reduce 求和 ----
  FixArray sum_e_x;
  {
    vector<FixArray> tmp = e_x_tr;
    int num_adds_old = n; int num_adds_curr = n/2;
    while(num_adds_old > 1) {
      int odd_num_adds = num_adds_old & 1;
      vector<FixArray> lhs(num_adds_curr); 
      vector<FixArray> rhs(num_adds_curr);
      for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        lhs[j/2] = tmp[j]; rhs[j/2] = tmp[j+1];
      }
      FixArray lhs_concat = concat(lhs);
      FixArray rhs_concat = concat(rhs);
      lhs_concat = fix->add(lhs_concat, rhs_concat);
      for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        tmp[odd_num_adds + (j/2)] = lhs_concat.subset((j/2)*N, (j/2)*N + N);
      }
      num_adds_old = num_adds_curr + odd_num_adds;
      num_adds_curr = num_adds_old/2;
    }
    sum_e_x = tmp[0];
  }
  sum_e_x.signed_ = false;


  // ---- Step 6: 除法归一化 ----
  FixArray ret_flat = fix->div_batch(e_x_flat, sum_e_x, n, exp_ell, s);


  // ---- Step 7: 扩展并恢复矩阵格式 ----
  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  ret_flat = fix->extend(ret_flat, ell);

  vector<FixArray> ret(N);
  for (int i = 0; i < N; i++) {
    ret[i] = FixArray(party, n, signed_, ell, s);
    memcpy(ret[i].data, ret_flat.data + i*n, n*sizeof(uint64_t));
  }
  FixArray l_short;


  return make_tuple(ret, l_short);
}


vector<FixArray> FPMath::softmax_fix_iron_1(const vector<FixArray>& x) {
  int N = x.size();
  // for (int i = 0; i < N; i++){
  //   print_fix(x[i]);
  // }
  int n = x[0].size;
  int ell = x[0].ell;
  int s = x[0].s;
  bool signed_ = x[0].signed_;
  // assert(m_bits > 0);
  for(int i = 1; i < N; i++) {
    assert(x[i].party != PUBLIC);
    assert(x[i].ell == ell);
    assert(x[i].s == s);
    assert(x[i].size == n);
  }
  FixArray x_max = fix->max_iron(x);
  // print_fix(x_max);
  FixArray x_max_flat(party, N*n, signed_, ell, s);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      x_max_flat.data[i*n + j] = x_max.data[i];
    }
  }

  FixArray x_flat = concat(x);
  FixArray shifted_x_flat = fix->sub(x_flat, x_max_flat);

  // FixArray e_x_flat = fix->exp(shifted_x_flat, ell, s);

  FixArray e_x_flat = lookup_table_exp(shifted_x_flat);

  vector<FixArray> e_x_tr(n);
  for (int i = 0; i < n; i++) {
    e_x_tr[i] = FixArray(party, N, signed_, ell, s);
    for (int j = 0; j < N; j++) {
      e_x_tr[i].data[j] = e_x_flat.data[j*n + i];
    }
  }
  FixArray sum_e_x;
  {
    vector<FixArray> tmp = e_x_tr;
    int num_adds_old = n; int num_adds_curr = n/2;
    while(num_adds_old > 1) {
      int odd_num_adds = num_adds_old & 1;
      vector<FixArray> lhs(num_adds_curr); vector<FixArray> rhs(num_adds_curr);
      for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        lhs[j/2] = tmp[j]; rhs[j/2] = tmp[j+1];
      }
      FixArray lhs_concat = concat(lhs);
      FixArray rhs_concat = concat(rhs);
      lhs_concat = fix->add(lhs_concat, rhs_concat);
      for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
        tmp[odd_num_adds + (j/2)] = lhs_concat.subset((j/2)*N, (j/2)*N + N);
      }
      num_adds_old = num_adds_curr + odd_num_adds;
      num_adds_curr = num_adds_old/2;
    }
    sum_e_x = tmp[0];
  }
  FixArray sum_e_x_replicated(party, N*n, signed_, ell, s);
  for(int i = 0; i < N; i++) {
    for (int j = 0; j < n; j++) {
      sum_e_x_replicated.data[i*n + j] = sum_e_x.data[i];
    }
  }
  sum_e_x_replicated.signed_ = false;
  FixArray ret_flat = fix->div(e_x_flat, sum_e_x_replicated, ell, s);
  
  // sum_e_x.signed_ = false;
  // FixArray ret_flat = fix->div_batch(e_x_flat, sum_e_x, n ,ell, s);

  // FixArray ret_flat = x_max_flat;
  vector<FixArray> ret(N);
  for (int i = 0; i < N; i++) {
    ret[i] = FixArray(party, n, signed_, ell, s);
    memcpy(ret[i].data, ret_flat.data + i*n, n*sizeof(uint64_t));
  }
  // for (int i = 0; i < N; i++){
  //   print_fix(ret[i]);
  // }
  return ret;
}

FixArray FPMath::lookup_table_exp(const FixArray& x){
  FixArray ret(party, x.size, x.signed_, x.ell, x.s);
  math->lookup_table_exp(x.size, x.data, ret.data, x.ell, x.ell, x.s, x.s);
  return ret;
}

// double gelu_y = 0.5*dbl_x*(1+tanh(sqrt(2/M_PI)*(dbl_x+0.044715*dbl_x*dbl_x*dbl_x)));
FixArray FPMath::gelu_iron(const FixArray& x){
  int N = x.size;
  int ell = x.ell;
  int s = x.s;

  // Constants

  FixArray cons_half = fix->input(PUBLIC, x.size, uint64_t(0.5 * pow(2, s)), false, ell, s);
  FixArray cons_less_half = fix->input(PUBLIC, x.size, uint64_t(0.044715 * pow(2, s)), false, ell, s);
  FixArray cons_some_pi = fix->input(PUBLIC, x.size, uint64_t((std::sqrt(2/M_PI)) * (1ULL << s)), false, ell, s);

  FixArray x_square = fix->mul(x, x, ell+s);
  x_square = fix->truncate_reduce(x_square, s);

  FixArray x_cube = fix->mul(x_square, x, ell+s);
  x_cube = fix->truncate_reduce(x_cube, s);

  FixArray x_cube_less_half = fix->mul(x_cube, cons_less_half, ell+s);
  x_cube_less_half = fix->truncate_reduce(x_cube_less_half, s);
  x_cube_less_half = fix->add(x_cube_less_half, x);

  FixArray x_cube_pi = fix->mul(x_cube_less_half, cons_some_pi, ell+s);
  x_cube_pi = fix->truncate_reduce(x_cube_pi, s);

  FixArray post_tanh = fix->tanh(x_cube_pi, ell, s);
  post_tanh = fix->add(post_tanh, 1 << s);

  FixArray half_x = fix->mul(x, cons_half, ell+s);
  half_x = fix->truncate_reduce(half_x, s);

  FixArray ret = fix->mul(half_x, post_tanh, ell+s);
  ret = fix->truncate_reduce(ret, s);

  return ret;
}

FixArray FPMath::gelu_approx(const FixArray& x){
  int N = x.size;
  int ell = x.ell;
  int s = x.s;

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N, 1);

  // Get y = abs(x)
  BoolArray msb_x = fix->MSB(x);
  FixArray neg_x = fix->mul(x, -1);
  FixArray y = fix->if_else(msb_x, neg_x, x);

  // z(y):= y * (y + (-4.901373660634577)) \in [-6.006, 0]
  FixArray cons_1 = fix->input(PUBLIC, N, uint64_t((-4.901373660634577) * (1 << s)), true, ell, s);
  FixArray z_y = fix->add(y, cons_1);

  // y > 0, z_y < 0
  z_y = fix->mul(z_y, y, ell+s, all_1.data, all_0.data);
  z_y = fix->truncate_reduce(z_y, s);

  // p(y):= (z(y)+y+(-24.822721454603112))*(z(y)+(31.65224017185284))+(785.77248299658)
  FixArray cons_2 = fix->input(PUBLIC, N, uint64_t((-24.822721454603112) * (1 << s)), true, ell, s);
  FixArray cons_3 = fix->input(PUBLIC, N, uint64_t((31.65224017185284) * (1 << s)), true, ell, s);
  FixArray cons_4 = fix->input(PUBLIC, N, uint64_t((785.77248299658) * (1 << s)), true, ell, s);

  // Negative
  FixArray lmul = fix->add(z_y, y);
  lmul = fix->add(lmul, cons_2);

  // Positive
  FixArray rmul = fix->add(z_y, cons_3);

  FixArray ladd = fix->mul(lmul, rmul, ell+s, all_1.data, all_0.data);
  ladd = fix->truncate_reduce(ladd, s);

  FixArray sum = fix->add(ladd, cons_4);

  FixArray cons_7 = fix->input(PUBLIC, N, uint64_t((0.02084861175412759) * (1 << s)), true, ell, s);
  FixArray p_y = fix->mul(sum, cons_7, ell+s, nullptr, all_0.data);
  p_y = fix->truncate_reduce(p_y, s);

  // 0.5x
  FixArray cons_5 = fix->input(PUBLIC, N, uint64_t((0.5) * (1 << s)), true, ell, s);
  FixArray half_x = fix->mul(x, cons_5, ell+s, nullptr, all_0.data);
  half_x = fix->truncate_reduce(half_x, s);

  // 0.5x + p(y)
  FixArray gelu_y = fix->add(half_x, p_y);

  // If x > 2.7 -> x
  // If x < -2.7 -> 0
  // Else 0.5x + p(y)

  // BoolArray gt27 = fix->GT(x, 2.7*(1 << s));
  // BoolArray lt_neg_27 = fix->LT(x, -2.7*(1 << s));

  // FixArray cons_6 = fix->input(PUBLIC, N, uint64_t(0 * (1 << s)), true, ell, s);

  // FixArray ret = fix->if_else(lt_neg_27, cons_6, gelu_y);
  // ret = fix->if_else(gt27, x, ret);

  BoolArray gt27 = fix->GT(y, 2.7*(1 << s));


  FixArray x_plus_y = fix->add(x, y);
  FixArray half_x_plus_y = fix->right_shift(x_plus_y, 1, all_0.data);
  half_x_plus_y.s = s;

  FixArray ret = fix->if_else(gt27, half_x_plus_y, gelu_y);
  return ret;
}

FixArray FPMath::gelu_approx_2(const FixArray& x){
  int N = x.size;
  int ell = x.ell;
  int s = x.s;

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N, 1);

  // Get y = abs(x)
  BoolArray msb_x = fix->MSB(x);
  FixArray neg_x = fix->mul(x, -1);
  FixArray y = fix->if_else(msb_x, neg_x, x);

  // z(y) = (0.14439048359960427*y - 0.7077117131613893) * y + 4.5702822654246535

  FixArray cons_1 = fix->input(PUBLIC, N, uint64_t((0.14439048359960427) * (1 << s)), true, ell, s);
  FixArray cons_2 = fix->input(PUBLIC, N, uint64_t((-0.7077117131613893) * (1 << s)), true, ell, s);
  FixArray cons_3 = fix->input(PUBLIC, N, uint64_t((4.5702822654246535) * (1 << s)), true, ell, s);
  FixArray cons_4 = fix->input(PUBLIC, N, uint64_t((-8.15444702051307) * (1 << s)), true, ell, s);
  FixArray cons_5 = fix->input(PUBLIC, N, uint64_t((16.382265425072532) * (1 << s)), true, ell, s);

  FixArray y_cons1 = fix->mul(y, cons_1, ell+s, all_0.data, all_0.data);
  y_cons1 = fix->truncate_reduce(y_cons1, s);
  FixArray zy_left = fix->add(y_cons1, cons_2);
  // < 0

  zy_left = fix->mul(zy_left, y, ell+s, all_1.data, all_0.data);
  zy_left = fix->truncate_reduce(zy_left, s);
  FixArray z_y = fix->add(zy_left, cons_3);
  // > 0

  // Gelu(x) = (z(y) + 0.14439048359960427 |x| - 8.15444702051307) * y + 16.382265425072532  + 0.5x

  FixArray p_y_left = fix->add(z_y, y_cons1);
  p_y_left = fix->add(p_y_left, cons_4);
  p_y_left = fix->mul(p_y_left, z_y, ell+s, all_1.data ,all_0.data);
  p_y_left = fix->truncate_reduce(p_y_left, s);

  FixArray p_y = fix->add(p_y_left, cons_5);

  // 0.5x
  FixArray cons_6 = fix->input(PUBLIC, N, uint64_t((0.5) * (1 << s)), true, ell, s);
  FixArray half_x = fix->mul(x, cons_6, ell+s, nullptr, all_0.data);
  half_x = fix->truncate_reduce(half_x, s);

  // 0.5x + p(y)
  FixArray gelu_y = fix->add(half_x, p_y);

  BoolArray lt27 = fix->LT(y, 2.7*(1 << s));


  FixArray x_plus_y = fix->add(x, y);
  FixArray half_x_plus_y = fix->right_shift(x_plus_y, 1, all_0.data);
  half_x_plus_y.s = s;

  FixArray ret = fix->if_else(lt27, gelu_y, half_x_plus_y);

  BoolArray msb_ret = bool_op->AND(msb_x, lt27);

  ret = fix->extend(ret, 37, msb_ret.data);
  ret =fix->right_shift(ret, 7, msb_ret.data);
  
  return ret;
}

/*
FixArray FPMath::gelu_approx_3(const FixArray& x){

//这一部分可不变
  //获取大小
  int N = x.size;
  //整数位数
  int ell = x.ell;
  //小数部分位数（缩放因子）
  int s = x.s;
  //即对于1.5,会存储为整数1.5*2^s
//不变
  //全0全1构造
  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N, 1);
//不变
  // y=|x|
  //取符号位
  BoolArray msb_x = fix->MSB(x);
  //获取-x
  FixArray neg_x = fix->mul(x, -1);
  //由符号位获取y=|x|
  FixArray y = fix->if_else(msb_x, neg_x, x);
//p(x)=-0.0001506671x^8+00043702779x^6-00512144957x^4+0.3828738634x^2+0.5x+0.0028445838

 FixArray c8 = fix->input(PUBLIC, N, (uint64_t)(-0.0001506671 * (1ULL << s)), true, ell, s);
 FixArray c6 = fix->input(PUBLIC, N, (uint64_t)(0.0043702779 * (1ULL << s)), true, ell, s);
 FixArray c4 = fix->input(PUBLIC, N, (uint64_t)(-0.0512144957 * (1ULL << s)), true, ell, s);
 FixArray c2 = fix->input(PUBLIC, N, (uint64_t)(0.3828738634 * (1ULL << s)), true, ell, s);
 FixArray c0 = fix->input(PUBLIC, N, (uint64_t)(0.0028445838 * (1ULL << s)), true, ell, s);
// P_even(y) = ( ( (c8*y^2 + c6)*y^2 + c4)*y^2 + c2)*y^2 + c0
  //计算y^2,基本单元
 FixArray y2 = fix->mul(y, y, ell + s, all_0.data, all_0.data);
 y2 = fix->truncate_reduce(y2, s);
  //c8 * y^2
 FixArray p_even_y = fix->mul(c8, y2, ell + s, all_1.data, all_0.data);
 p_even_y = fix->truncate_reduce(p_even_y, s);
 //c8 * y^2+c6
 p_even_y = fix->add(p_even_y, c6);
  //(c8*y2 + c6) * y2
  //(c8*y2 + c6)符号需要先判断
  BoolArray msb_p_even_y1 = fix->MSB(p_even_y);
  p_even_y = fix->mul(p_even_y, y2, ell + s, msb_p_even_y1.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  //(c8*y^2 + c6)*y^2 + c4
  p_even_y = fix->add(p_even_y, c4);
  
  //((c8*y^2 + c6)*y^2 + c4)*y^2
  BoolArray msb_p_even_y2 = fix->MSB(p_even_y);
  p_even_y = fix->mul(p_even_y, y2, ell + s, msb_p_even_y2.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  // ( (c8*y^2 + c6)*y^2 + c4)*y^2 + c2
  p_even_y = fix->add(p_even_y, c2);
  
  //( ( (c8*y^2 + c6)*y^2 + c4)*y^2 + c2)*y^2
  BoolArray msb_p_even_y3 = fix->MSB(p_even_y);
  p_even_y = fix->mul(p_even_y, y2, ell + s, msb_p_even_y3.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  
  //( ( (c8*y^2 + c6)*y^2 + c4)*y^2 + c2)*y^2 + c0
  p_even_y = fix->add(p_even_y, c0);
  
//构造
  // 0.5x
  FixArray cons_6 = fix->input(PUBLIC, N, uint64_t((0.5) * (1 << s)), true, ell, s);
  FixArray half_x = fix->mul(x, cons_6, ell+s, nullptr, all_0.data);
  half_x = fix->truncate_reduce(half_x, s);

  // 0.5x + p(y)
  FixArray gelu_y = fix->add(half_x, p_even_y);

//找到断点
  //改为自己的断点
  BoolArray lt312 = fix->LT(y, 3.12*(1 << s));


//可取
  //两端值计算，z=(x+|x|)/2
  FixArray x_plus_y = fix->add(x, y);
  //通过位移实现除法，右移2^1，相当于除以2
  FixArray half_x_plus_y = fix->right_shift(x_plus_y, 1, all_0.data);
  half_x_plus_y.s = s;

  FixArray ret = fix->if_else(lt312, gelu_y, half_x_plus_y);

  BoolArray msb_ret = bool_op->AND(msb_x, lt312);

  ret = fix->extend(ret, 37, msb_ret.data);
  ret =fix->right_shift(ret, 7, msb_ret.data);
  
  return ret;
}
*/

FixArray FPMath::gelu_approx_3(const FixArray& x) {

  int N   = x.size;
  int ell = x.ell;
  int s   = x.s;

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));

  // 符号和绝对值
  BoolArray msb_x = fix->MSB(x);
  FixArray neg_x  = fix->mul(x, -1);
  FixArray y      = fix->if_else(msb_x, neg_x, x);
  y.s = s;

  // 系数（定点输入）
  FixArray c8 = fix->input(PUBLIC, N, (int64_t)(-0.0001506671 * (1ULL << s)), true, ell, s);
  FixArray c6 = fix->input(PUBLIC, N, (int64_t)( 0.0043702779 * (1ULL << s)), true, ell, s);
  FixArray c4 = fix->input(PUBLIC, N, (int64_t)(-0.0512144957 * (1ULL << s)), true, ell, s);
  FixArray c2 = fix->input(PUBLIC, N, (int64_t)( 0.3828738634 * (1ULL << s)), true, ell, s);
  FixArray c0 = fix->input(PUBLIC, N, (int64_t)( 0.0028445838 * (1ULL << s)), true, ell, s);

  // 计算 y^2
  FixArray y2 = fix->mul(y, y, ell+s, all_0.data, all_0.data);
  y2 = fix->truncate_reduce(y2, s);

  // 使用霍纳法则计算多项式 p(y) = c8*y^8 + c6*y^6 + c4*y^4 + c2*y^2 + c0
  // 重写为: p(y) = ((c8*y^2 + c6)*y^2 + c4)*y^2 + c2)*y^2 + c0
  FixArray p_even_y = fix->mul(c8, y2, ell+s, all_0.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  p_even_y = fix->add(p_even_y, c6);

  p_even_y = fix->mul(p_even_y, y2, ell+s, all_0.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  p_even_y = fix->add(p_even_y, c4);

  p_even_y = fix->mul(p_even_y, y2, ell+s, all_0.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  p_even_y = fix->add(p_even_y, c2);

  p_even_y = fix->mul(p_even_y, y2, ell+s, all_0.data, all_0.data);
  p_even_y = fix->truncate_reduce(p_even_y, s);
  p_even_y = fix->add(p_even_y, c0);

  // 0.5x
  FixArray cons_half = fix->input(PUBLIC, N, uint64_t(0.5*(1<<s)), true, ell, s);
  FixArray half_x    = fix->mul(x, cons_half, ell+s, nullptr, all_0.data);
  half_x = fix->truncate_reduce(half_x, s);

  // gelu近似表达式
  FixArray gelu_y = fix->add(half_x, p_even_y);

  // 分段阈值: 3.125 (25/8，二进制友好)
  FixArray thr = fix->input(PUBLIC, N, uint64_t((3.125)*(1<<s)), true, ell, s);
  BoolArray leq_thr = fix->LE(y, thr);

  // 另一侧: (x + |x|) / 2
  FixArray x_plus_y      = fix->add(x, y);
  FixArray half_x_plus_y = fix->right_shift(x_plus_y, 1, all_0.data);
  half_x_plus_y.s = s;   // 确保缩放一致

  // 分支逻辑（一次性 if_else）
  FixArray ret = fix->if_else(leq_thr, gelu_y, half_x_plus_y);
  ret.s = s; // 再保险，保证输出scale和输入一致

  return ret;
}

FixArray FPMath::tanh_inner_preprocess(const FixArray& x){
  int N = x.size;
  int ell = x.ell;
  int s = x.s;

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N, 1);

  // Const
  FixArray t0 = fix->input(PUBLIC, x.size, uint64_t((-4.259314087994767) * (1 << s)), true, ell, s);
  FixArray t1 = fix->input(PUBLIC, x.size, uint64_t((18.86353816972803) * (1 << s)), true, ell, s);
  FixArray t2 = fix->input(PUBLIC, x.size, uint64_t((-36.42402897526823) * (1 << s)), true, ell, s);
  FixArray t3 = fix->input(PUBLIC, x.size, uint64_t((-0.013232131886235352) * (1 << s)), true, ell, s);
  FixArray t4 = fix->input(PUBLIC, x.size, uint64_t((-3.3289339650097993) * (1 << s)), true, ell, s);
  FixArray t5 = fix->input(PUBLIC, x.size, uint64_t((-0.0024920889620412097) * (1 << s)), true, ell, s);

  // p1(x) = (x + t0)*x + t1 
  // Range: >0
  FixArray p1 = fix->add(x, t0);
  p1 = fix->mul(p1, x, ell+s, all_0.data, all_0.data);
  p1 = fix->truncate_reduce(p1, s);
  p1 = fix->add(p1, t1);

  // p2(x) = (p1(x) + x + t2)*p1(x)*x*t3 + t4*x + t5

  // (p1(x) + x + t2) < 0
  FixArray p2 = fix->add(p1, x);
  p2 = fix->add(p2, t2);

  // (p1(x) + x + t2)*p1(x) < 0
  p2 = fix->mul(p2, p1, ell + s, all_1.data, all_0.data);
  p2 = fix->truncate_reduce(p2, s);

  // p2(x) = (p1(x) + x + t2)*p1(x)*t3 >0
  p2 = fix->mul(p2, t3, ell+s, all_1.data, all_1.data);
  p2 = fix->truncate_reduce(p2, s);

  // p2(x) = (p1(x) + x + t2)*p1(x)*x*t3 < 0
  p2 = fix->mul(p2, x, ell+s, all_1.data, all_0.data);
  p2 = fix->truncate_reduce(p2, s);

  FixArray t4x = fix->mul(x, t4, ell+s, all_0.data, all_1.data);
  t4x = fix->truncate_reduce(t4x, s);

  p2 = fix->add(p2, t4x);
  p2 = fix->add(p2, t5);

  return p2;
}
FixArray FPMath::silu_approx(const FixArray& x) {
    int N   = x.size;
    int ell = x.ell;
    int s   = x.s;   // 定点小数位

    BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, N, 1);

    // ---------------------------
    // Step1: 计算 -x
    // ---------------------------
    FixArray neg_x = fix->mul(x, -1);

    // ---------------------------
    // Step2: 调用查表指数 exp_lut(-x)
    // （内部: 拆整数/小数，整数查 LUT [-18,0]，小数多项式近似）
    // ---------------------------
    FixArray exp_neg_x = exp_lut2(neg_x);

    // ---------------------------
    // Step3: 分母 (1 + e^(-x))
    // ---------------------------
    FixArray dn = fix->add(exp_neg_x, 1ULL << exp_neg_x.s);

    // 防溢出标准化修正 (和 sigmoid 一样的处理)
    BoolArray dn_eq_2 = fix->EQ(dn, 2ULL << dn.s);
    dn = fix->if_else(dn_eq_2, fix->sub(dn, 1), dn);

    // ---------------------------
    // Step4: 倒数 inv_dn = 1 / (1 + e^-x)
    // ---------------------------
    FixArray one_dn = fix->input(ALICE, N, 1, true, 2, 0);
    FixArray inv_dn = fix->div(one_dn, dn, s + 2, s, true);

    // ---------------------------
    // Step5: 乘回 x => SiLU(x) = x / (1 + e^-x)
    // ---------------------------
    FixArray silu = fix->mul(x, inv_dn, ell + s, all_1.data, all_0.data);
    silu = fix->truncate_reduce(silu, s);

    return silu;
}
FixArray FPMath::tanh_inner(const FixArray& x){
  int N = x.size;
  int ell = x.ell;
  int s = x.s;

  // Const
  FixArray a = fix->input(PUBLIC, x.size, uint64_t((-0.013232131886235352) * (1 << s)), true, ell, s);
  FixArray b = fix->input(PUBLIC, x.size, uint64_t((0.09948747962825866) * (1 << s)), true, ell, s);
  FixArray c = fix->input(PUBLIC, x.size, uint64_t((-0.20093640347818847) * (1 << s)), true, ell, s);
  FixArray d = fix->input(PUBLIC, x.size, uint64_t((-0.17616532856475706) * (1 << s)), true, ell, s);
  FixArray e = fix->input(PUBLIC, x.size, uint64_t((1.0542492677156243) * (1 << s)), true, ell, s);
  FixArray f = fix->input(PUBLIC, x.size, uint64_t((-0.0024920889620412097) * (1 << s)), true, ell, s);

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N, 1);

  // 
  FixArray x_square = fix->mul(x, x, ell+s, all_0.data, all_0.data);
  x_square = fix->truncate_reduce(x_square, s);

  FixArray x_cube = fix->mul(x_square, x, ell+s, all_0.data, all_0.data);
  x_cube = fix->truncate_reduce(x_cube, s);

  FixArray x_four = fix->mul(x_square, x_square, ell+s, all_0.data, all_0.data);
  x_four = fix->truncate_reduce(x_four, s);

  FixArray x_five = fix->mul(x_four, x, ell+s, all_0.data, all_0.data);
  x_five = fix->truncate_reduce(x_five, s);

  FixArray x_five_a = fix->mul(x_five, a, ell+s, all_0.data, all_1.data);
  x_five_a = fix->truncate_reduce(x_five_a, s);

  FixArray x_four_b = fix->mul(x_four, b, ell+s, all_0.data, all_0.data);
  x_four_b = fix->truncate_reduce(x_four_b, s);

  FixArray x_cube_c = fix->mul(x_cube, c, ell+s, all_0.data, all_1.data);
  x_cube_c = fix->truncate_reduce(x_cube_c, s);

  FixArray x_square_d = fix->mul(x_square, d, ell+s, all_0.data, all_1.data);
  x_square_d = fix->truncate_reduce(x_square_d, s);

  FixArray x_e = fix->mul(x, e, ell+s, all_0.data, all_0.data);
  x_e = fix->truncate_reduce(x_e, s);

  f = fix->add(f, x_e);
  f = fix->add(f, x_square_d);
  f = fix->add(f, x_cube_c);
  f = fix->add(f, x_four_b);
  f = fix->add(f, x_five_a);

  return f;
}

FixArray FPMath::tanh_approx(const FixArray& x){
  int N = x.size;
  int ell = x.ell;
  int s = x.s;

  BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N, 1);

  FixArray cons_2 = fix->input(PUBLIC, x.size, uint64_t((2 << s)), true, ell, s);
  FixArray cons_1 = fix->input(PUBLIC, x.size, uint64_t((1 << s)), true, ell, s);
  FixArray cons_neg_1 = fix->input(PUBLIC, x.size, uint64_t((-1 << s)), true, ell, s);

  BoolArray pos = fix->GT(x, 0);
  FixArray neg_x = fix->mul(x, -1);
  FixArray abs_x = fix->if_else(pos, x, neg_x);

  FixArray cond_fix = fix->B2A(pos, true, ell);
  cond_fix = fix->scale_up(cond_fix, ell, s);
  FixArray sign_x = fix->mul(cond_fix, cons_2, ell + s, all_0.data, all_0.data);
  sign_x = fix->truncate_reduce(sign_x, s);
  sign_x = fix->add(sign_x, cons_neg_1);

  BoolArray gt3 = fix->GT(abs_x, (uint64_t)(2.855 * (1 << s)));
  FixArray abs_tanh = fix->if_else(gt3, cons_1, tanh_inner_preprocess(abs_x));
  FixArray ret = fix->mul(abs_tanh, sign_x, ell+s, all_0.data);
  ret = fix->truncate_reduce(ret, s);

  return ret;
}

vector<FixArray> FPMath::layer_norm_fix(const vector<FixArray>& x, FixArray& w, FixArray&b){
  int N = x.size();
  int n = x[0].size;
  int ell = x[0].ell;
  int s = x[0].s;
  bool signed_ = x[0].signed_;

  BoolArray all_0 = bool_op->input(ALICE, N*n, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N*n, 1);

  FixArray sum = fix->tree_sum(x);

  FixArray dn = fix->input(PUBLIC, sum.size, uint64_t(((1.0 / n) * pow(2, 2*s))), true, ell, 2*s);
  FixArray avg = fix->mul(sum, dn, ell+2*s, nullptr, all_0.data);
  avg = fix->truncate_reduce(avg, 2*s);

  FixArray avg_flat(party, N*n, sum.signed_, ell, s);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < n; j++){
      avg_flat.data[i*n + j] = avg.data[i];
    }
  }

  FixArray x_flat = concat(x);
  FixArray x_flat_avg = fix->sub(x_flat, avg_flat);
  BoolArray msb_x_avg = fix->MSB(x_flat_avg);
  FixArray x_flat_avg_square = fix->mul(x_flat_avg, x_flat_avg, ell + s, msb_x_avg.data, msb_x_avg.data);
  x_flat_avg_square = fix->truncate_reduce(x_flat_avg_square, s);

  vector<FixArray> square_group(N);
  for(int i = 0; i < N; i++){
    square_group[i] = FixArray(party, n, signed_, ell, s);
    memcpy(square_group[i].data, &x_flat_avg_square.data[i*n], n*sizeof(uint64_t));
  }

  FixArray square_sum = fix->tree_sum(square_group);
  square_sum = fix->mul(square_sum, dn, ell+2*s, all_0.data, all_0.data);
  square_sum = fix->truncate_reduce(square_sum, 2*s);
  FixArray sigma = sqrt(square_sum, true);

  FixArray sigma_flat(party, N*n, sum.signed_, ell, s);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < n; j++){
      sigma_flat.data[i*n + j] = sigma.data[i];
    }
  }

  FixArray x_avg_sigma = fix->mul(x_flat_avg, sigma_flat, ell+s, msb_x_avg.data, all_0.data);
  x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);

  // Weight and Bias
  // x_avg_sigma = fix->mul(x_avg_sigma, w, ell+s);
  // x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
  // x_avg_sigma = fix->add(x_avg_sigma, b);

  // Hack!
  // x_avg_sigma = fix->extend(x_avg_sigma, 64);

  vector<FixArray> ret(N);
  for(int i = 0; i < N; i++){
    ret[i] = FixArray(party, n, signed_, ell, s);
    memcpy(ret[i].data, &x_avg_sigma.data[i*n], n*sizeof(uint64_t));
  }
  return ret;
}


vector<FixArray> FPMath::rms_norm_fix(const vector<FixArray>& x, FixArray& w, FixArray& b){
  int N = x.size();
  int n = x[0].size;
  int ell = x[0].ell;
  int s = x[0].s;
  bool signed_ = x[0].signed_;

  // Flatten 输入
  FixArray x_flat = concat(x);

  // Step1: 计算平方 (x^2)
  BoolArray msb_x = fix->MSB(x_flat);  
  FixArray x_square = fix->mul(x_flat, x_flat, ell + s, msb_x.data, msb_x.data);
  x_square = fix->truncate_reduce(x_square, s);

  // Step2: 按批（group）求和 (每个样本 n 维求和)
  vector<FixArray> square_group(N);
  for(int i = 0; i < N; i++){
    square_group[i] = FixArray(party, n, signed_, ell, s);
    memcpy(square_group[i].data, &x_square.data[i*n], n*sizeof(uint64_t));
  }
  FixArray square_sum = fix->tree_sum(square_group);

  // Step3: 除以 n 得到均方 (mean of squares)
  FixArray dn = fix->input(PUBLIC, square_sum.size, 
                           uint64_t(((1.0 / n) * pow(2, 2*s))), 
                           true, ell, 2*s);
  square_sum = fix->mul(square_sum, dn, ell+2*s, nullptr, nullptr);
  square_sum = fix->truncate_reduce(square_sum, 2*s);

  // Step4: RMS = sqrt(mean_of_squares)
  FixArray rms = sqrt(square_sum, true);

  // Step5: broadcast RMS
  FixArray rms_flat(party, N*n, signed_, ell, s);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < n; j++){
      rms_flat.data[i*n + j] = rms.data[i];
    }
  }

  // Step6: x / RMS (即归一化)
  BoolArray msb_x2 = fix->MSB(x_flat);
  FixArray x_normed = fix->mul(x_flat, rms_flat, ell+s, msb_x2.data, nullptr);
  x_normed = fix->truncate_reduce(x_normed, s);

  // Step7: 拆分成 batch 形式返回
  vector<FixArray> ret(N);
  for(int i = 0; i < N; i++){
    ret[i] = FixArray(party, n, signed_, ell, s);
    memcpy(ret[i].data, &x_normed.data[i*n], n*sizeof(uint64_t));
  }

  return ret;
}


vector<FixArray> FPMath::layer_norm_iron(const vector<FixArray>& x, FixArray& w, FixArray&b){
  int N = x.size();
  int n = x[0].size;
  int ell = x[0].ell;
  int s = x[0].s;
  bool signed_ = x[0].signed_;

  BoolArray all_0 = bool_op->input(ALICE, N*n, uint8_t(0));
  BoolArray all_1 = bool_op->input(ALICE, N*n, 1);

  FixArray sum = fix->tree_sum(x);

  FixArray dn = fix->input(PUBLIC, sum.size, uint64_t(((1.0 / n) * pow(2, 2*s))), true, ell, 2*s);
  FixArray avg = fix->mul(sum, dn, ell+2*s, nullptr, all_0.data);
  avg = fix->truncate_reduce(avg, 2*s);

  FixArray avg_flat(party, N*n, sum.signed_, ell, s);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < n; j++){
      avg_flat.data[i*n + j] = avg.data[i];
    }
  }

  FixArray x_flat = concat(x);
  FixArray x_flat_avg = fix->sub(x_flat, avg_flat);
  FixArray x_flat_avg_square = fix->mul(x_flat_avg, x_flat_avg, ell + s);
  x_flat_avg_square = fix->truncate_reduce(x_flat_avg_square, s);

  vector<FixArray> square_group(N);
  for(int i = 0; i < N; i++){
    square_group[i] = FixArray(party, n, signed_, ell, s);
    memcpy(square_group[i].data, &x_flat_avg_square.data[i*n], n*sizeof(uint64_t));
  }

  FixArray square_sum = fix->tree_sum(square_group);
  square_sum = fix->mul(square_sum, dn, ell+2*s, all_0.data, all_0.data);
  square_sum = fix->truncate_reduce(square_sum, 2*s);
  FixArray sigma = sqrt(square_sum, true);

  FixArray sigma_flat(party, N*n, sum.signed_, ell, s);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < n; j++){
      sigma_flat.data[i*n + j] = sigma.data[i];
    }
  }

  FixArray x_avg_sigma = fix->mul(x_flat_avg, sigma_flat, ell+s, nullptr, all_0.data);
  x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);

  // Weight and Bias
  x_avg_sigma = fix->mul(x_avg_sigma, w, ell+s);
  x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
  x_avg_sigma = fix->add(x_avg_sigma, b);

  // Hack!
  // x_avg_sigma = fix->extend(x_avg_sigma, 64);

  vector<FixArray> ret(N);
  for(int i = 0; i < N; i++){
    ret[i] = FixArray(party, n, signed_, ell, s);
    memcpy(ret[i].data, &x_avg_sigma.data[i*n], n*sizeof(uint64_t));
  }
  return ret;
}

FixArray FPMath::sqrt(const FixArray& x, bool recp_sqrt){
  FixArray ret(party, x.size, x.signed_, x.ell, x.s);
  math->sqrt(x.size, x.data, ret.data, x.ell, x.ell, x.s, x.s, recp_sqrt);
  return ret;
}

FixArray FPMath::tanh_iron(const FixArray& x){
  FixArray ret(party, x.size, x.signed_, x.ell, x.s);
  math->tanh(x.size, x.data, ret.data, x.ell, x.ell, x.s, x.s);
  return ret;
}

FixArray FPMath::gt_p_sub(const FixArray& x, const FixArray& p){
  BoolArray gt = fix->GT(x, p);
  FixArray sub = fix->sub(x, p);
  return fix->if_else(gt, sub, x);
}

void FPMath::print(const FixArray& x){
  print_fix(x);
}

BoolArray bitonic_reverse(const BoolArray &x, int array_size, int cur_depth){
  BoolArray ret(x.party, x.size);
  int block_size = 2*cur_depth;
  int num_block = array_size / block_size;
  for(int i = 0; i < num_block; i++){
    
      for(int j = 0; j < cur_depth; j++){
        int index = i*cur_depth + j;
        if(i % 2 == 1){
          ret.data[index] = x.data[index] ^ ((x.party != BOB) ? 1 : 0);
        } else {
          ret.data[index] = x.data[index];
        }
    }
  }
  return ret;
}

tuple<FixArray, FixArray, FixArray> FPMath::bitonic_sort_and_swap(
  const FixArray& x_, FixArray softmax_v_, FixArray h1_, bool swap){
    FixArray x = x_;
    FixArray softmax_v = softmax_v_;
    FixArray h1 = h1_;

    int array_size = x.size;
    int max_depth = array_size / 2;
    int cur_depth = 1;

    int common_dim;

    while(cur_depth <= max_depth){
      int cur_iter = cur_depth;
      while(cur_iter > 0){
        int block_size = 2*cur_iter;
        int num_block = array_size / block_size;

        vector<int> index_left;
        vector<int> index_right;

        FixArray array_left(party, x.size / 2, x.signed_, x.ell, x.s);
        FixArray array_right(party, x.size / 2, x.signed_, x.ell, x.s);
        FixArray array_reverse(party, x.size, x.signed_, x.ell, x.s);

        FixArray softmax_v_reverse;
        FixArray h1_reverse;

        if(swap){
          common_dim = softmax_v.size / x.size;
          assert(common_dim == 768);

          softmax_v_reverse = fix->input(party, softmax_v.size, (uint64_t)0 ,softmax_v.signed_, softmax_v.ell, softmax_v.s);
          h1_reverse = fix->input(party, h1.size, (uint64_t)0, h1.signed_, h1.ell, h1.s);

          for(int i = 0; i < num_block; i++){
            for(int j = 0; j < cur_iter; j++){
              int pos_x = i*block_size + j;
              int pos_y = i*block_size + j + cur_iter;
              index_left.push_back(pos_x);
              index_right.push_back(pos_y);
              array_reverse.data[pos_x] = x.data[pos_y];
              array_reverse.data[pos_y] = x.data[pos_x];

              memcpy(
                &softmax_v_reverse.data[pos_x*common_dim], 
                &softmax_v.data[pos_y*common_dim], 
                common_dim*sizeof(uint64_t)
              );

              memcpy(
                &softmax_v_reverse.data[pos_y*common_dim], 
                &softmax_v.data[pos_x*common_dim], 
                common_dim*sizeof(uint64_t)
              );

              memcpy(
                &h1_reverse.data[pos_x*common_dim], 
                &h1.data[pos_y*common_dim], 
                common_dim*sizeof(uint64_t)
              );

              memcpy(
                &h1_reverse.data[pos_y*common_dim], 
                &h1.data[pos_x*common_dim], 
                common_dim*sizeof(uint64_t)
              );

            }
          }
        } else{
          for(int i = 0; i < num_block; i++){
            for(int j = 0; j < cur_iter; j++){
              index_left.push_back(i*block_size + j);
              index_right.push_back(i*block_size + j + cur_iter);
              array_reverse.data[i*block_size + j] = x.data[i*block_size + j + cur_iter];
              array_reverse.data[i*block_size + j + cur_iter] = x.data[i*block_size + j];
            }
          }
        }

        for(int i = 0; i < array_size / 2; i++){
          array_left.data[i] = x.data[index_left[i]];
          array_right.data[i] = x.data[index_right[i]];
        }

        // print_fix(array_left);
        // print_fix(array_right);
        // print_fix(array_reverse);
        

        BoolArray lt = fix->LT(array_left, array_right);
        BoolArray cmp_extend = BoolArray(party, lt.size*2);

        // Reverse some comparisons
        BoolArray cmp = bitonic_reverse(lt, array_size, cur_depth);
        for(int i = 0; i < num_block; i++){
          for(int j = 0; j < cur_iter; j++){
            cmp_extend.data[i*block_size + j] = 
              cmp.data[i*cur_iter + j];
            cmp_extend.data[i*block_size + j + cur_iter] = 
              cmp.data[i*cur_iter + j];
          }
        }
        // print_bool(lt);  
        // print_bool(cmp);
        // print_bool(cmp_extend);
        // assert(0);

        // print_fix(x);
        x = fix->if_else(cmp_extend, x, array_reverse);
        // print_fix(x);

        if(swap){
          BoolArray cmp_flat = BoolArray(party, cmp_extend.size*common_dim);
          for(int i = 0; i < cmp_flat.size; i++){
            cmp_flat.data[i] = cmp_extend.data[i / common_dim];
          }

          softmax_v = fix->if_else(cmp_flat, softmax_v, softmax_v_reverse);
          h1 = fix->if_else(cmp_flat, h1, h1_reverse);
        }

        cur_iter /= 2;
      }
      cur_depth*=2;
    }

    return make_tuple(x, softmax_v, h1);
}
