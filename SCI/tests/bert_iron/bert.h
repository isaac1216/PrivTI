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

    Bert(int party, int port, string address, string model_path);
    ~Bert();


    vector<Plaintext> he_to_ss_server(HE* he, vector<Ciphertext> in, const FCMetadata &data);
    vector<Ciphertext> ss_to_he_server(HE* he, uint64_t* input, const FCMetadata &data);

    vector<Plaintext> he_to_ss_client(HE* he, int length);
    void ss_to_he_client(HE* he, uint64_t* input, const FCMetadata &data);

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
        uint64_t* bln
    );

    void ln_share_client(
        uint64_t* wln,
        uint64_t* bln
    );

    void run_server();

    int run_client(string input_fname);

    vector<double> run(string input_fname, string mask_fname);

    inline uint64_t get_comm();
    inline uint64_t get_round();
	
};

#endif