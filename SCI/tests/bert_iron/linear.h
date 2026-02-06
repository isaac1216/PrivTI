#ifndef LINEAR_H__
#define LINEAR_H__

#include "he.h"
#include "FloatingPoint/fp-math.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <math.h>
#include "bert_utils.h"

#define PACKING_NUM 12

#define INPUT_DIM 128
#define COMMON_DIM 768
#define OUTPUT_DIM 64
#define INTER_DIM 3072

#define ATTENTION_LAYERS 12

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 12
struct FCMetadata {
  int slot_count;
//   int32_t pack_num;
//   int32_t inp_ct;
  // Filter is a matrix
  int32_t filter_h;
  int32_t filter_w;
//   int32_t filter_size;
  // Image is a matrix
  int32_t image_size;
  int nw;
  int kw;
};

struct PreprocessParams_1{
    vector<vector<vector<Plaintext>>> encoded_mats1;
	vector<vector<vector<Plaintext>>> encoded_mats2;
	vector<vector<vector<Plaintext>>> encoded_mats3;
	vector<vector<Plaintext>> encoded_bias1;
	vector<vector<Plaintext>> encoded_bias2;
	vector<vector<Plaintext>> encoded_bias3;
};

struct PreprocessParams_2{
    vector<vector<Plaintext>> encoded_mat;
   	vector<Plaintext> encoded_bias;
};

class Linear
{
public:
	int party;
	NetIO *io;
	FCMetadata data;

	HE *he_8192;

	// Fix linking error
	uint64_t p_mod;

	FCMetadata data_lin1;
    FCMetadata data_lin2;
    FCMetadata data_lin3;
    FCMetadata data_lin4;

	// Attention
	vector<PreprocessParams_1> pp_1;
	vector<PreprocessParams_2> pp_2;
	vector<PreprocessParams_2> pp_3;
	vector<PreprocessParams_2> pp_4;

	// Layer Norm
    vector<vector<uint64_t>> w_ln_1;
    vector<vector<uint64_t>> b_ln_1;

    vector<vector<uint64_t>> w_ln_2;
    vector<vector<uint64_t>> b_ln_2;

	// Pooling
    vector<vector<uint64_t>> w_p;
    vector<uint64_t> b_p;

    // Classification
    vector<vector<uint64_t>> w_c;
    vector<uint64_t> b_c;

	Linear();

	Linear(int party, NetIO *io);

	~Linear();

	void configure();

	void generate_new_keys();

	PreprocessParams_1 params_preprocessing_ct_pt_1(
		HE* he,
		vector<vector<vector<uint64_t>>> w_q,
		vector<vector<vector<uint64_t>>> w_k,
		vector<vector<vector<uint64_t>>> w_v,
		vector<vector<uint64_t>> b_q,
		vector<vector<uint64_t>> b_k,
		vector<vector<uint64_t>> b_v,
		const FCMetadata &data
	);

	PreprocessParams_2 params_preprocessing_ct_pt_2(
		HE* he,
		int32_t input_dim, 
		int32_t common_dim, 
		int32_t output_dim,
		vector<vector<uint64_t>> w,
		vector<uint64_t> b,
		const FCMetadata &data
	);

	

	void weights_preprocess(BertModel &bm);


	vector<Ciphertext> linear_1(
		HE* he,
		vector<Ciphertext> input_cts, 
		PreprocessParams_1 &pp,
		const FCMetadata &data
		);
	
	vector<Ciphertext> linear_2(
		HE* he,
		vector<Ciphertext> input_cts, 
		PreprocessParams_2 &pp,
		const FCMetadata &data
		);

	vector<vector<Plaintext>> preprocess_matrix(HE* he, const uint64_t *const *matrix, const FCMetadata &data);
	vector<Plaintext> preprocess_bias(HE* he, const uint64_t *matrix, const FCMetadata &data);
	Plaintext encode_vector(HE* he, const uint64_t *vec, const FCMetadata &data);
	vector<Ciphertext> preprocess_vec(HE* he, vector<uint64_t> &input, const FCMetadata &data);
	vector<Ciphertext> preprocess_ptr(HE* he, uint64_t* input, const FCMetadata &data);
	vector<Plaintext> preprocess_ptr_plaintext(HE* he, uint64_t* input, const FCMetadata &data);

	vector<Plaintext> preprocess_noise_1(HE* he, const uint64_t *secret_share, const FCMetadata &data);
	vector<Plaintext> preprocess_noise_2(HE* he, const uint64_t *secret_share, const FCMetadata &data);
	vector<vector<vector<uint64_t>>> bert_postprocess_noise(HE* he, vector<Plaintext> &enc_noise, const FCMetadata &data);

	vector<vector<vector<uint64_t>>> 
    pt_postprocess_1(
		HE* he,
        vector<Plaintext> &pts, 
        const FCMetadata &data, 
		const bool &col_packing);
	


    void pt_postprocess_2(
		HE* he,
        vector<Plaintext> &pts, 
		uint64_t* output,
        const FCMetadata &data, 
		const bool &col_packing);

	void concat( 
		uint64_t* input,
		uint64_t* output,
		int n,
		int dim1,
		int dim2);

	void plain_col_packing_preprocess(
		uint64_t* input, 
		uint64_t * output,
		uint64_t plain_mod,
		int common_dim,
		int input_dim);
};

#endif
