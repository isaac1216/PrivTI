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
struct FCMetadata
{
	int slot_count;
	int32_t pack_num;
	int32_t inp_ct;
	// Filter is a matrix
	int32_t filter_h;
	int32_t filter_w;
	int32_t filter_size;
	// Image is a matrix
	int32_t image_size;
};

struct PreprocessParams_1{
   	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wq_pack;
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wk_pack;
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wv_pack;

	vector<vector<Plaintext>> bq_pack;
	vector<vector<Plaintext>> bk_pack;
	vector<vector<Plaintext>> bv_pack;

	vector<Plaintext> cross_masks;
};

struct PreprocessParams_2{
    pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> cross_mat_single;
    vector<Plaintext> cross_bias_single;
};

class Linear
{
public:
	int party;
	NetIO *io;

	bool prune;

	HE *he_8192;
	HE *he_8192_tiny;
	HE *he_8192_ln;
	// HE *he_4096;

	// Fix linking error
	uint64_t p_mod;

	FCMetadata data_lin1_0;
	FCMetadata data_lin1_1;

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

	vector<vector<Plaintext>> w_ln_1_pt;
	vector<vector<Plaintext>> w_ln_2_pt;

	// Pooling
    vector<vector<uint64_t>> w_p;
    vector<uint64_t> b_p;

    // Classification
    vector<vector<uint64_t>> w_c;
    vector<uint64_t> b_c;

	Linear();

	Linear(int party, NetIO *io, bool prune);

	~Linear();

	void configure();

	void generate_new_keys();

	PreprocessParams_1 params_preprocessing_ct_ct(
		HE* he,
		vector<vector<vector<uint64_t>>> w_q,
		vector<vector<vector<uint64_t>>> w_k,
		vector<vector<vector<uint64_t>>> w_v,
		vector<vector<uint64_t>> b_q,
		vector<vector<uint64_t>> b_k,
		vector<vector<uint64_t>> b_v,
		const FCMetadata &data
	);

	PreprocessParams_2 params_preprocessing_ct_pt(
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

	// concat on dim1
	// output: dim2 x (dim1xdim3)
	vector<vector<uint64_t>> concat_vec(
		uint64_t* att,
		int n,
		int dim1,
		int dim2);

	// concat on dim1
	// output: dim2 x (dim1xdim3)
	void concat(
		uint64_t* input,
		uint64_t* output,
		int n,
		int dim1,
		int dim2);

 	vector<Plaintext> generate_cross_packing_masks(HE* he, const FCMetadata &data);

	vector<Ciphertext> rotation_by_one_depth3(
	HE* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k);

	vector<Ciphertext>
	bert_efficient_preprocess_vec(
		HE* he,
		vector<uint64_t> &input,
		const FCMetadata &data);
	
	
	uint64_t* bert_cross_packing_postprocess(
		HE* he,
		vector<Ciphertext> &cts, 
		const FCMetadata &data);
	
	void plain_cross_packing_postprocess(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);
	
	void plain_cross_packing_postprocess_v(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);
	
	void plain_col_packing_preprocess(
		uint64_t* input, 
		uint64_t * output,
		uint64_t plain_mod,
		int common_dim,
		int input_dim);
	
	void plain_col_packing_preprocess_vec(
		vector<vector<uint64_t>> input, 
		uint64_t * output,
		uint64_t plain_mod,
		int common_dim,
		int input_dim);
	
	void plain_col_packing_postprocess(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);

	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_matrix(
		HE* he,
		const uint64_t *const *matrix1,
		const uint64_t *const *matrix2,
		const FCMetadata &data);

	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>>
	bert_cross_packing_single_matrix(
		HE* he,
		const vector<vector<vector<uint64_t>>> &weights, 
		const FCMetadata &data);
	
	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_single_matrix_2(
		HE* he,
		const uint64_t *const *matrix1,
		const uint64_t *const *matrix2,
		const FCMetadata &data);

	vector<vector<Plaintext>> bert_cross_packing_bias(
		HE* he,
		const vector<vector<uint64_t>> &bias, 
		const FCMetadata &data);

	
	vector<Plaintext> bert_cross_packing_bias_2(
		HE* he,
		const uint64_t *matrix, 
		const FCMetadata &data);

	// Cipher * Plain with BSGS for the first layer (Q,K,V)
	void bert_cipher_plain_bsgs(
		HE* he,
		const vector<Ciphertext> &cts, 
		const vector<pair<vector<vector<Plaintext>>, 
		vector<vector<Plaintext>>>> &wq_pack, 
		const vector<pair<vector<vector<Plaintext>>,
		vector<vector<Plaintext>>>> &wk_pack, 
		const vector<pair<vector<vector<Plaintext>>, 
		vector<vector<Plaintext>>>> &wv_pack, 
		const vector<vector<Plaintext>> &bq_pack, 
		const vector<vector<Plaintext>> &bk_pack, 
		const vector<vector<Plaintext>> &bv_pack, 
		const FCMetadata &data, 
		vector<Ciphertext> &result);

	// Cipher * Plain with BSGS for other linear layers
	void bert_cipher_plain_bsgs_2(
		HE* he,
		const vector<Ciphertext> &cts, 
		const vector<vector<Plaintext>> &enc_mat1, 
		const vector<vector<Plaintext>> &enc_mat2, 
		const vector<Plaintext> &enc_bias, 
		const FCMetadata &data, 
		vector<Ciphertext> &result);

	// Cipher * Cipher in the attention mechanism
	void bert_cipher_cipher_cross_packing(
		HE* he,
		const FCMetadata &data,
		const vector<Ciphertext> &Cipher_plain_result,
		const vector<Plaintext> &cross_masks,
		vector<Ciphertext> &results);

	// Softmax * V

	// Pack the sharing of softmax result on the client and encrypt
	vector<Ciphertext> preprocess_softmax_s1(
		HE* he,
		uint64_t* matrix, 
		const FCMetadata &data);
	
	// Pack the sharing of softmax result on the server and encode to plaintext polynomials
	vector<vector<vector<Plaintext>>> 
	preprocess_softmax_s2(HE* he, const uint64_t *matrix, const FCMetadata &data);

	void client_S1_V_R(HE* he, const uint64_t *softmax_s1,uint64_t* V, uint64_t* result, const FCMetadata &data);

	void bert_postprocess_V(HE* he,  uint64_t* input, uint64_t* result, const FCMetadata &data, const bool &col_packing);
	void bert_postprocess_V_enc(HE* he, vector<Ciphertext> cts, uint64_t* result, const FCMetadata &data, const bool &col_packing);

	vector<vector<vector<Plaintext>>>
	preprocess_softmax_v_r(HE* he, const uint64_t *matrix, const FCMetadata &data);

	vector<vector<vector<Plaintext>>> bert_softmax_v_packing_single_matrix(
		HE* he,
		const vector<vector<vector<uint64_t>>> &weights, 
		const FCMetadata &data);

	void bert_softmax_V(
		HE* he, vector<Ciphertext> &softmax_s1, 
		vector<vector<vector<Plaintext>>> &softmax_s2, 
		vector<Ciphertext> &V, 
		vector<vector<vector<Plaintext>>> &R, 
		const FCMetadata &data, 
		vector<Ciphertext> &result);

	vector<Ciphertext> w_ln(HE* he, vector<Ciphertext> ln, vector<Plaintext> w);

	void preprocess_softmax(const uint64_t *input, uint64_t *output, const FCMetadata &data);

	void softmax_v(
		HE* he,
		vector<vector<vector<Ciphertext>>> &softmax_s2, 
		vector<Ciphertext> &V, 
		const FCMetadata &data, 
		vector<Ciphertext> &result);

	vector<vector<Plaintext>> softmax_mask_ct_ct(HE* he, const FCMetadata &data);

	vector<vector<vector<Ciphertext>>> preprocess_softmax_s1_ct_ct(HE* he, const vector<Ciphertext> &matrix, const FCMetadata &data, vector<vector<Plaintext>> &mask);

};

#endif
