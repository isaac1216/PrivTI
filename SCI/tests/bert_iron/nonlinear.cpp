#include "nonlinear.h"

NonLinear::NonLinear(int party, string address, int port){
    this->party = party;
    this->address = address;
    this->port = port;

    for(int i = 0; i < MAX_THREADS; ++i){
        this->iopackArr[i] = new IOPack(party, port + i, address);
        if (i & 1) {
            this->otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
            this->fpmath[i] = new FPMath(3 - party, iopackArr[i], otpackArr[i]);
        } else {
            this->otpackArr[i] = new OTPack(iopackArr[i], party);
            this->fpmath[i] = new FPMath(party, iopackArr[i], otpackArr[i]);
        }
    }
}

NonLinear::NonLinear(){}

NonLinear::~NonLinear(){
  for (int i = 0; i < MAX_THREADS; i++) {
    // delete this->iopackArr[i];
    // delete this->otpackArr[i];
    // delete this->fpmath[i];
  }
}

void softmax_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }
  vector<FixArray> output_array;
  tie(output_array, ignore) = fpmath->softmax_fix(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::softmax(int nthreads, uint64_t* input, uint64_t* output, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                softmax_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void softmax_irons_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }
  vector<FixArray> output_array = fpmath->softmax_fix_iron_1(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::softmax_iron(int nthreads, uint64_t* input, uint64_t* output, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                softmax_irons_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void layer_norm_thread(int tid, int party, uint64_t *x, uint64_t *y, uint64_t *w, uint64_t *b, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    FixArray input = fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s);
    // input = fpmath->fix->right_shift(input, 4);
    // input = fpmath->fix->extend(input, ell);
    input_array.push_back(input);
  }
  FixArray w_array = fpmath->fix->input(this_party, num_ops*array_size, w, true, ell, s);
  FixArray b_array = fpmath->fix->input(this_party, num_ops*array_size, b, true, ell, s);
  vector<FixArray> output_array = fpmath->layer_norm_iron(input_array, w_array, b_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::layer_norm(int nthreads, uint64_t* input, uint64_t* output, uint64_t* weight, uint64_t* bias, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                layer_norm_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                &weight[offset*array_size], 
                &bias[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gelu_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->gelu_approx_2(input);
  output = fpmath->fix->extend(output, 64);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gelu(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gelu_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gelu_iron_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->gelu_iron(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gelu_iron(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gelu_iron_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void tanh_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->tanh_approx(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::tanh(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                tanh_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void tanh_iron_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->tanh_iron(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::tanh_iron(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                tanh_iron_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gt_p_sub_thread(int tid, int party, uint64_t *x, uint64_t p, uint64_t *y, int num_ops, int ell, int s_in, int s_out, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  // if input > p, then sub p
  // sub p/2 anyway
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s_in);
  // FixArray p_array = fpmath->fix->input(PUBLIC, num_ops, p, true, ell, s_in);
  FixArray p_2_array = fpmath->fix->input(PUBLIC, num_ops, (p-1)/2, true, ell, s_in);
  // FixArray output = fpmath->gt_p_sub(input, p_array);
  FixArray output = fpmath->fix->sub(input, p_2_array);
  if(s_in > s_out){
    output = fpmath->fix->right_shift(output, s_in - s_out);
  } else if(s_in < s_out){
    output = fpmath->fix->mul(output, 1<<(s_out - s_in));
  }
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gt_p_sub(int nthreads, uint64_t* input, uint64_t p, uint64_t* output, int size, int ell, int s_in, int s_out){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gt_p_sub_thread, 
                i, 
                party, 
                &input[offset], 
                p,
                &output[offset], 
                lnum_ops,
                ell,
                s_in, 
                s_out,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void mul_thread(int tid, int party, uint64_t *x, uint64_t* z, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  BoolArray all_1 = fpmath->bool_op->input(ALICE, num_ops, uint8_t(1));
  FixArray input_x = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray input_y = fpmath->fix->input(this_party, num_ops, z, true, ell, s);
  BoolArray msb_x = fpmath->fix->MSB(input_x);
  BoolArray msb_y = fpmath->fix->MSB(input_y);
  FixArray output = fpmath->fix->mul(input_x, input_y, ell + s, msb_x.data, msb_y.data);
  output = fpmath->fix->truncate_reduce(output, s);
  output = fpmath->fix->extend(output, 64);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void  NonLinear::n_matrix_mul(
  int nthreads, 
  uint64_t* input_1,
  uint64_t* input_2, 
  uint64_t* output, 
  int n, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell, 
  int s){

  // dim3 x dim2
  uint64_t* input_2_trans = new uint64_t[n*dim2*dim3];
  for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim2*dim3;
    for(int j = 0; j < dim3; j ++){
      for(int i = 0; i < dim2; i++){
        input_2_trans[j*dim2 + i + matrix_offset] = 
          input_2[i*dim3 + j + matrix_offset];
      }
    }
  }

  
  uint64_t* input_1_dup = new uint64_t[n*dim1*dim2*dim3];
  uint64_t* input_2_dup = new uint64_t[n*dim1*dim2*dim3];
  for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim1*dim2*dim3;
    int matrix_offset_2 = nm*dim2*dim3;
    for(int i = 0; i < dim1; i++){
      memcpy(&input_2_dup[i*dim2*dim3 + matrix_offset], &input_2_trans[matrix_offset_2], dim2*dim3*sizeof(uint64_t));
    }
  }

  

  for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim1*dim2*dim3;
    int matrix_offset_2 = nm*dim1*dim2;
    for(int i = 0; i < dim1; i++){
      for(int j = 0; j < dim3; j++){
        memcpy(&input_1_dup[i*dim2*dim3 + j*dim2 + matrix_offset], &input_1[i*dim2 + matrix_offset_2], dim2*sizeof(uint64_t));
      }
    }
  }

  int size = n*dim1*dim2*dim3;
  // int size = 128;
  uint64_t* output_tmp = new uint64_t[n*dim1*dim2*dim3];
  std::thread threads[nthreads];

  int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                mul_thread, 
                i, 
                party, 
                &input_1_dup[offset], 
                &input_2_dup[offset], 
                &output_tmp[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }

    // print_ss(&input_2_dup[0], 8, ell, s);
    // print_ss(&input_1_dup[0], 8, ell, s);
    // print_ss(&output_tmp[0], 8, ell, s);
    // return;

    for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim1*dim2*dim3;
    int matrix_offset_2 = nm*dim1*dim3;
      for(int i = 0; i < dim1; i++){
        for(int k = 0; k < dim3; k++){
          output[i*dim3 + k + matrix_offset_2] = 0;
          for(int j = 0; j < dim2; j++){
            output[i*dim3 + k + matrix_offset_2] += output_tmp[i*dim3*dim2 + k*dim2 + j + matrix_offset];
          }
        }
      }
    }

    delete[] input_2_trans;
    delete[] input_1_dup;
    delete[] input_2_dup;
    delete[] output_tmp;
}

void matmul_thread(
  int tid, 
  int party, 
  uint64_t *a, 
  uint64_t* b, 
  uint64_t *c, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell, 
  int s_in_1,
  int s_in_2,
  int s_out, 
  FPMath *fpmath) {
  
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  int extra_scale = s_in_1 + s_in_2 - s_out;
  fpmath->fix->mult->matrix_multiplication(
      dim1,
      dim2,
      dim3,
      a,
      b,
      c,
      ell,
      ell,
      ell + extra_scale,
      true,
      true,
      true
    );
    BoolArray all_1 = fpmath->bool_op->input(ALICE, dim1*dim3, uint8_t(1));
    FixArray ret = fpmath->fix->input(this_party, dim1*dim3, c, true, ell+extra_scale, s_in_1 + s_in_2);
    ret = fpmath->fix->truncate_reduce(ret, extra_scale);
    ret = fpmath->fix->extend(ret, 64);
    memcpy(c, ret.data, (dim1*dim3)*sizeof(uint64_t));
}

void  NonLinear::n_matrix_mul_iron(
  int nthreads, 
  uint64_t* input_1,
  uint64_t* input_2, 
  uint64_t* output, 
  int n, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell, 
  int s_in_1,
  int s_in_2,
  int s_out){

  std::thread threads[n];

  for (int i = 0; i < n; ++i) {
      threads[i] =
          std::thread(
              matmul_thread, 
              i, 
              party, 
              &input_1[i*dim1*dim2],
              &input_2[i*dim2*dim3],
              &output[i*dim1*dim3],
              dim1,
              dim2,
              dim3,
              ell,
              s_in_1,
              s_in_2,
              s_out,
              this->fpmath[i]);
  }
  for (int i = 0; i < n; ++i) {
      threads[i].join();
  }
}

void right_shift_thread(int tid, int party, uint64_t *x, int a, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->fix->right_shift(input, a);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::right_shift(int nthreads, uint64_t* input, int a, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                right_shift_thread, 
                i, 
                party, 
                &input[offset], 
                a,
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void NonLinear::print_ss(uint64_t* input, int length, int ell, int s){
  FixArray tmp = fpmath[0]->fix->input(party, length, input, true, ell, s);
  fpmath[0]->print(tmp);
}

FixArray NonLinear::to_public(uint64_t* input, int length, int ell, int s){
  FixArray tmp = fpmath[0]->fix->input(party, length, input, true, ell, s);
  tmp = fpmath[0]->fix->extend(tmp, 64);
  tmp = fpmath[0]->fix->output(PUBLIC, tmp);
  return tmp;
}