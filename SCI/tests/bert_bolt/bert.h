#ifndef BERT_H__
#define BERT_H__

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include "linear.h"
#include "nonlinear.h"

#define NL_NTHREADS 32
#define NL_ELL 37

#define GELU_ELL 20
#define GELU_SCALE 11
#define NL_SCALE 12
#define NUM_CLASS 2

// #define BERT_DEBUG
#define BERT_PERF
// #define BERT_SAVE_RESULTS


using namespace std;

class Bert
{
public:
    int party;
    string address;
    int port;

    NetIO *io;

    Linear lin;
    NonLinear nl;

    bool prune;

    Bert(int party, int port, string address, string model_path, bool prune = false);
    ~Bert();


    void he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output, bool ring);
    vector<Ciphertext> ss_to_he_server(HE* he, uint64_t* input, int length, int bw);

    void he_to_ss_client(HE* he, uint64_t* output, int length, const FCMetadata &data);
    void ss_to_he_client(HE* he, uint64_t* input, int length, int bw);

    void pc_bw_share_server(
        uint64_t* wp,
        uint64_t* bp,
        uint64_t* wc,
        uint64_t* bc
        );
    void pc_bw_share_client(
        uint64_t* wp,
        uint64_t* bp,
        uint64_t* wc,
        uint64_t* bc
    );

    void ln_share_server(
        int layer_id,
        vector<uint64_t> &wln_input,
        vector<uint64_t> &bln_input,
        uint64_t* wln,
        uint64_t* bln, 
        const FCMetadata &data
    );

    void ln_share_client(
        uint64_t* wln,
        uint64_t* bln, 
        const FCMetadata &data
    );

    void softmax_v(
        HE* he,
        vector<Ciphertext> enc_v,
        uint64_t* s_softmax,
        uint64_t* s_v,
        uint64_t* s_softmax_v, 
        const FCMetadata &data
    );

    vector<double> run(string input_fname, string mask_fname);

    inline uint64_t get_comm();
    inline uint64_t get_round();

    void print_p_share(uint64_t* s, uint64_t p, int len);
    void check_p_share(uint64_t* s, uint64_t p, int len, uint64_t* ref);
	
};

#endif
