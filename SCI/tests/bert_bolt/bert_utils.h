#ifndef BERT_UTILS_H
#define BERT_UTILS_H

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include <vector>
#include <sstream>

using namespace std;

struct BertModel {

    // Attention
    vector<vector<vector<vector<uint64_t>>>> w_q;
    vector<vector<vector<vector<uint64_t>>>> w_k;
    vector<vector<vector<vector<uint64_t>>>> w_v;
    vector<vector<vector<uint64_t>>> w_o;
    vector<vector<vector<uint64_t>>> w_i_1;
    vector<vector<vector<uint64_t>>> w_i_2;

    vector<vector<vector<uint64_t>>> b_q;
    vector<vector<vector<uint64_t>>> b_k;
    vector<vector<vector<uint64_t>>> b_v;
    vector<vector<uint64_t>> b_o;
    vector<vector<uint64_t>> b_i_1;
    vector<vector<uint64_t>> b_i_2;

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
};

vector<vector<uint64_t>> read_data(const string& filename);
vector<uint64_t> read_bias(const string& filename, int output_dim) ;

vector<vector<vector<uint64_t>>> read_qkv_weights(const string& filename);
vector<vector<uint64_t>> read_qkv_bias(const string& filename);

BertModel load_model(string model_dir, int num_class);

#endif