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

#ifndef FLOATING_POINT_MATH_H__
#define FLOATING_POINT_MATH_H__

#include "FloatingPoint/floating-point.h"
#include "Math/math-functions.h"

class FPMath {
public:
  int party;
  sci::IOPack *iopack;
  sci::OTPack *otpack;
  BoolOp *bool_op;
  FixOp *fix;
  FPOp *fp_op;
  MathFunctions *math;

  FPMath(int party, sci::IOPack *iopack, sci::OTPack *otpack) {
    this->party = party;
    this->iopack = iopack;
    this->otpack = otpack;
    this->bool_op = new BoolOp(party, iopack, otpack);
    this->fix = new FixOp(party, iopack, otpack);
    this->fp_op = new FPOp(party, iopack, otpack);
    this->math = new MathFunctions(party, iopack, otpack);
  }

  ~FPMath() {
    delete bool_op;
    delete fix;
    delete fp_op;
  }

  // Floating-Point Math Functions: returns OP(x[i]), OP = {sinpi, cospi, tanpi, exp2, log2, exp, ln, erf}
  // x must be secret-shared
  FPArray sinpi(const FPArray &x);
  FPArray cospi(const FPArray &x);
  FPArray tanpi(const FPArray &x);
  FPArray exp2(const FPArray &x);
  FPArray log2(const FPArray &x);
  FPArray exp(const FPArray &x);
  FPArray ln(const FPArray &x);
  FPArray erf(const FPArray &x);
  vector<FPArray> softmax(const vector<FPArray>& x);

  FPArray exp3(const FPArray &x);
  std::tuple<FixArray, FixArray> exp4(const FixArray &x);
  //lms_e^x
  FixArray exp_lut(const FixArray &x);
  FixArray exp_lut2(const FixArray &x);

  FPArray sigmoid_bf16(const FPArray &x);
  FPArray sigmoid_fp32(const FPArray &x);
  FPArray tanh_bf16(const FPArray &x);
  FPArray tanh_fp32(const FPArray &x);

  vector<FPArray> softmax_beacon(const vector<FPArray>& x);
  vector<FPArray> softmax_secfloat(const vector<FPArray>& x);
  std::tuple<vector<FixArray>, FixArray> softmax_fix(const vector<FixArray>& x);
  //lsm_softmax
  std::tuple<vector<FixArray>, FixArray> softmax_fix2(const vector<FixArray>& x);

  vector<FixArray> softmax_fix_iron_1(const vector<FixArray>& x);
  
  FixArray lookup_table_exp(const FixArray& x);

  FixArray gelu_iron(const FixArray& x);

  FixArray gelu_approx(const FixArray& x);
  FixArray gelu_approx_2(const FixArray& x);
  //lms_gelu
  FixArray gelu_approx_3(const FixArray& x);
  
  FixArray silu_approx(const FixArray& x);

  FixArray tanh_inner(const FixArray& x);

  FixArray tanh_inner_preprocess(const FixArray& x);

  FixArray tanh_approx(const FixArray& x);

  FixArray tanh_iron(const FixArray& x);

  FixArray gt_p_sub(const FixArray& x, const FixArray& p);

  vector<FixArray> layer_norm_fix(const vector<FixArray>& x, FixArray& w, FixArray&b);
  vector<FixArray> layer_norm_iron(const vector<FixArray>& x, FixArray& w, FixArray&b);
  
  vector<FixArray> rms_norm_fix(const vector<FixArray>& x, FixArray& w, FixArray&b);

  FixArray sqrt(const FixArray& x, bool recp_sqrt);

  std::tuple<FixArray, FixArray, FixArray> bitonic_sort_and_swap(
    const FixArray& x, FixArray softmax_v_, FixArray h1_, bool swap);

  void print(const FixArray& x);

  // FixArray recip_approx(const FixArray& x);
  
  // vector<FixArray> softmax_fix_iron_2(const vector<FixArray>& x);
};

#endif // FLOATING_POINT_MATH_H__
