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

#include "nonlinear.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>

using namespace sci;
using namespace std;

#define MAX_THREADS 12

int party, port = 32000;
int num_threads = 2;
string address = "127.0.0.1";

int32_t dim = num_threads*2;
int32_t array_size = 2;
int32_t bw_x = 37;
int32_t bw_y = 37;
int32_t s_x = 12;
int32_t s_y = 12;
int32_t input_size = dim*array_size;

bool signed_ = true;

uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << 14) - 1));
uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << 14) - 1));

void softmax_double(const double* input, double* output, int dim, int array_size) {
  
  for (int i = 0; i < dim; i++){
    double sumExp = 0.0;
    // Compute the exponential of each input element and accumulate the sum
    for (int j = 0; j < array_size; ++j) {
      double expValue = std::exp(input[j + i*array_size]);
      output[j + i*array_size] = expValue;
      sumExp += expValue;
    }
    
    // Normalize the exponential values by dividing each by the sum
    for (int j = 0; j < array_size; ++j) {
      output[j + i*array_size] /= sumExp;
    }
  }
}

uint64_t computeULPErr(double calc, double actual, int SCALE) {
  int64_t calc_fixed = (double(calc) * (1ULL << SCALE));
  int64_t actual_fixed = (double(actual) * (1ULL << SCALE));
  uint64_t ulp_err = (calc_fixed - actual_fixed) > 0
                         ? (calc_fixed - actual_fixed)
                         : (actual_fixed - calc_fixed);
  return ulp_err;
}


int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of operation operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  /********** Setup IO and Base OTs ***********/
  /********************************************/

  NonLinear nl(party, address, port);
  std::cout << "All Base OTs Done" << std::endl;

  /************ Generate Test Data ************/
  /********************************************/

  // char fix_key[] = "\x61\x7e\xcd\xa2\xa0\x51\x1e\x96"
  //                      "\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";

  PRG128 prg(fix_key);

  uint64_t *x = new uint64_t[input_size];
  uint64_t *y = new uint64_t[input_size];

  prg.random_data(x, dim * array_size * sizeof(uint64_t));

  for (int i = 0; i < dim; i++) {
    // FixArray tmp(party, array_size, true, bw_x, s_x);
    for(int j=0;j < array_size;j++)
       x[i*array_size+j] &= mask_x;
  }

  std::cout << "All data done" << std::endl;
  /************** Fork Threads ****************/
  /********************************************/
  nl.softmax(num_threads, x, y, dim, array_size, bw_x, s_x);

  std::cout << "Nonlinear computation done" << std::endl;


  /************** Verification ****************/
  /********************************************/
  if (party == ALICE) {
    nl.iopackArr[0]->io->send_data(x, input_size * sizeof(uint64_t));
    nl.iopackArr[0]->io->send_data(y, input_size * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *x0 = new uint64_t[input_size];
    uint64_t *y0 = new uint64_t[input_size];
    nl.iopackArr[0]->io->recv_data(x0, input_size * sizeof(uint64_t));
    nl.iopackArr[0]->io->recv_data(y0, input_size * sizeof(uint64_t));

    uint64_t total_err = 0;
    uint64_t max_ULP_err = 0;
    
    // TODO: Change for Softmax
    double* dbl_x = new double[input_size];
    double* dbl_y = new double[input_size];
    double* dbl_ref = new double[input_size];
    for (int i = 0; i < input_size; i++) {
      dbl_x[i] = (signed_val(x0[i] + x[i], bw_x)) / double(1LL << s_x);
      dbl_y[i] = (signed_val(y0[i] + y[i], bw_y)) / double(1LL << s_y);
    }

    softmax_double(dbl_x, dbl_ref, dim, array_size);

    for (int i = 0; i < input_size; i++) {
      uint64_t err = computeULPErr(dbl_y[i], dbl_ref[i], s_y);
      if (err > 10){
        std::cout << "ULP Error: " << dbl_x[i] << "," << dbl_y[i] << "," << dbl_ref[i] << ","
      << err << endl;
      }
      
      total_err += err;
      max_ULP_err = std::max(max_ULP_err, err);
    }
    std::cerr << "Average ULP error: " << total_err / input_size << endl;
    std::cerr << "Max ULP error: " << max_ULP_err << endl;
    std::cerr << "Number of tests: " << input_size << endl;

    delete[] x0;
    delete[] y0;
  }

//   std::cout << "Number of operation/s:\t" << (double(dim) / t) * 1e6 << std::endl;
//   std::cout << "operation Time\t" << t / (1000.0) << " ms" << endl;
//   std::cout << "operation Bytes Sent\t" << total_comm << " bytes" << endl;

  /******************* Cleanup ****************/
  /********************************************/
  delete[] x;
  delete[] y;
//   for (int i = 0; i < num_threads; i++) {
//     delete iopackArr[i];
//     delete otpackArr[i];
//   }
}
