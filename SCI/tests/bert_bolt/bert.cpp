#include "bert.h"

#ifdef BERT_PERF
double t_total_linear1 = 0;
double t_total_linear2 = 0;
double t_total_linear3 = 0;
double t_total_linear4 = 0;

double t_total_pruning = 0;

double t_total_softmax = 0;
double t_total_mul_v = 0;
double t_total_gelu = 0;
double t_total_ln_1 = 0;
double t_total_ln_2 = 0;
double t_total_tanh = 0;

double t_total_repacking = 0;
double t_total_gt_sub = 0;
double t_total_shift = 0;

double t_total_conversion = 0;

double t_total_ln_share = 0;

uint64_t c_linear_1 = 0;
uint64_t c_linear_2 = 0;
uint64_t c_linear_3 = 0;
uint64_t c_linear_4 = 0;
uint64_t c_softmax = 0;
uint64_t c_pruning = 0;
uint64_t c_gelu = 0;
uint64_t c_ln1 = 0;
uint64_t c_ln2 = 0;
uint64_t c_softmax_v = 0;
uint64_t c_shift = 0;
uint64_t c_gt_sub = 0;
uint64_t c_tanh = 0;
uint64_t c_pc = 0;

uint64_t r_linear_1 = 0;
uint64_t r_linear_2 = 0;
uint64_t r_linear_3 = 0;
uint64_t r_linear_4 = 0;
uint64_t r_softmax = 0;
uint64_t r_pruning = 0;
uint64_t r_gelu = 0;
uint64_t r_ln1 = 0;
uint64_t r_ln2 = 0;
uint64_t r_softmax_v = 0;
uint64_t r_shift = 0;
uint64_t r_gt_sub = 0;
uint64_t r_tanh = 0;
uint64_t r_pc = 0;

double n_rounds = 0;
double n_comm = 0;

inline uint64_t Bert::get_comm(){
    uint64_t total_comm = io->counter;
    for(int i = 0; i < MAX_THREADS; i++){
        total_comm += nl.iopackArr[i]->get_comm();
    }
    uint64_t ret_comm = total_comm - n_comm;
    n_comm = total_comm;
    return ret_comm;
}

inline uint64_t Bert::get_round(){
    uint64_t total_round = io->num_rounds;
    for(int i = 0; i < MAX_THREADS; i++){
        total_round += nl.iopackArr[i]->get_rounds();
    }
    uint64_t ret_round = total_round - n_rounds;
    n_rounds = total_round;
    return ret_round;
}
#endif 

inline double interval(chrono::_V2::system_clock::time_point start){
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
}

string replace_2(string str, string substr1, string substr2) {
    size_t index = str.find(substr1, 0); 
    str.replace(index, substr1.length(), substr2);
    return str;
}

void save_to_file(uint64_t* matrix, size_t rows, size_t cols, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << (int64_t)matrix[i * cols + j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

void save_to_file_vec(vector<vector<uint64_t>> matrix, size_t rows, size_t cols, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << matrix[i][j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

void print_pt(HE* he, Plaintext &pt, int len) {
    vector<uint64_t> dest(len, 0ULL);
    he->encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < 16; i++){
        if(dest[i] > he->plain_mod_2) {
            cout << (int64_t)(dest[i] - he->plain_mod) << " ";
        } else{
            cout << dest[i] << " ";
        }
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

void print_ct(HE* he, Ciphertext &ct, int len){
    Plaintext pt;
    he->decryptor->decrypt(ct, pt);
    cout << "Noise budget: ";
    cout << YELLOW << he->decryptor->invariant_noise_budget(ct) << " ";
    cout << RESET << endl;
    print_pt(he, pt, len);
}

Bert::Bert(int party, int port, string address, string model_path, bool prune){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "> Setup Linear" << endl;
    this->lin = Linear(party, io, prune);
    cout << "> Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port+1);

    this->prune = prune;

    if(party == ALICE){
        cout << "> Loading and preprocessing weights on server" << endl;
        #ifdef BERT_PERF
        auto t_load_model = high_resolution_clock::now();
        #endif 

        struct BertModel bm = 
            load_model(model_path, NUM_CLASS);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
        #endif 

        lin.weights_preprocess(bm);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
        #endif 
    }
    cout << "> Bert intialized done!" << endl << endl;

}

Bert::~Bert() {
    
}

void Bert::he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output, bool ring){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 

    PRG128 prg;
    int dim = in.size();
    int slot_count = he->poly_modulus_degree;
	prg.random_mod_p<uint64_t>(output, dim*slot_count, he->plain_mod);

    Plaintext pt_p_2;
    vector<uint64_t> p_2(slot_count, he->plain_mod_2);
    he->encoder->encode(p_2, pt_p_2);

    vector<Ciphertext> cts;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int j = 0; j < slot_count; ++j){
            tmp[j] = output[i*slot_count + j];
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct; 
        he->evaluator->sub_plain(in[i], pt, ct);
        if(ring){
            he->evaluator->add_plain_inplace(ct, pt_p_2);
        }
        // print_pt(he, pt, 8192);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
}

vector<Ciphertext> Bert::ss_to_he_server(HE* he, uint64_t* input, int length, int bw){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Plaintext> share_server;
    int dim = length / slot_count;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int j = 0; j < slot_count; ++j){
            tmp[j] = neg_mod(signed_val(input[i*slot_count + j], bw), (int64_t)plain_mod);
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        share_server.push_back(pt);
    }

    vector<Ciphertext> share_client(dim);
    recv_encrypted_vector(he->context, io, share_client);
    for(int i = 0; i < dim; i++){
        he->evaluator->add_plain_inplace(share_client[i], share_server[i]);
    }
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
    return share_client;

    // io->send_data(input, length*sizeof(uint64_t));
    // int slot_count = he->poly_modulus_degree;
    // uint64_t plain_mod = he->plain_mod;
    // int dim = length / slot_count;
    // vector<Ciphertext> share_client(dim);
    // recv_encrypted_vector(he->context, io, share_client);
    // return share_client;
}

void Bert::he_to_ss_client(HE* he, uint64_t* output, int length, const FCMetadata &data){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    vector<Ciphertext> cts(length);
    recv_encrypted_vector(he->context, io, cts);
    for(int i = 0; i < length; i++){
        vector<uint64_t> plain(data.slot_count, 0ULL);
        Plaintext tmp;
        he->decryptor->decrypt(cts[i], tmp);
        he->encoder->decode(tmp, plain);
        std::copy(plain.begin(), plain.end(), &output[i*data.slot_count]);
    }
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
}

void Bert::ss_to_he_client(HE* he, uint64_t* input, int length, int bw){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Ciphertext> cts;
    int dim = length / slot_count;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int j = 0; j < slot_count; ++j){
             tmp[j] = neg_mod(signed_val(input[i*slot_count + j], bw), (int64_t)plain_mod);
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct; 
        he->encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 

    // uint64_t* input_server = new uint64_t[length];
    // io->recv_data(input_server, length*sizeof(uint64_t));
    // int slot_count = he->poly_modulus_degree;
    // uint64_t plain_mod = he->plain_mod;
    // vector<Ciphertext> cts;
    // int dim = length / slot_count;
    // for(int i = 0; i < dim; i++){
    //     vector<uint64_t> tmp(slot_count);
    //     for(int j = 0; j < slot_count; ++j){
    //          tmp[j] = neg_mod(signed_val(input[i*slot_count + j] + input_server[i*slot_count + j], bw), (int64_t)plain_mod);
    //     }
    //     Plaintext pt;
    //     he->encoder->encode(tmp, pt);
    //     Ciphertext ct; 
    //     he->encryptor->encrypt(pt, ct);
    //     cts.push_back(ct);
    // }
    // send_encrypted_vector(io, cts);
}

void Bert::ln_share_server(
    int layer_id,
    vector<uint64_t> &wln_input,
    vector<uint64_t> &bln_input,
    uint64_t* wln,
    uint64_t* bln, 
    const FCMetadata &data
){
    #ifdef BERT_PERF
    auto t_ln_share = high_resolution_clock::now();
    #endif 

    int length = 2*COMMON_DIM;
    uint64_t* random_share = new uint64_t[length];

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for(int i = 0; i < length; i++){
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length*sizeof(uint64_t));

    for(int i = 0; i < COMMON_DIM; i++){
        random_share[i] = (wln_input[i] -  random_share[i]) & mask_x;
        random_share[i + COMMON_DIM] = 
            (bln_input[i] -  random_share[i + COMMON_DIM]) & mask_x;
    }

    for(int i = 0; i < data.image_size; i++){
        memcpy(&wln[i*COMMON_DIM], random_share, COMMON_DIM*sizeof(uint64_t));
        memcpy(&bln[i*COMMON_DIM], &random_share[COMMON_DIM], COMMON_DIM*sizeof(uint64_t));
    }

    delete[] random_share;
    #ifdef BERT_PERF
    t_total_ln_share += interval(t_ln_share);
    #endif 
}

void Bert::ln_share_client(
    uint64_t* wln,
    uint64_t* bln, 
    const FCMetadata &data
){

    #ifdef BERT_PERF
    auto t_ln_share = high_resolution_clock::now();
    #endif 

    int length = 2*COMMON_DIM;

    uint64_t* share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    for(int i = 0; i <  data.image_size; i++){
        memcpy(&wln[i*COMMON_DIM], share, COMMON_DIM*sizeof(uint64_t));
        memcpy(&bln[i*COMMON_DIM], &share[COMMON_DIM], COMMON_DIM*sizeof(uint64_t));
    }
    delete[] share;
    #ifdef BERT_PERF
    t_total_ln_share += interval(t_ln_share);
    #endif 
}

void Bert::pc_bw_share_server(
    uint64_t* wp,
    uint64_t* bp,
    uint64_t* wc,
    uint64_t* bc
    ){
    int wp_len = COMMON_DIM*COMMON_DIM;
    int bp_len = COMMON_DIM;
    int wc_len = COMMON_DIM*NUM_CLASS;
    int bc_len = NUM_CLASS;

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    int length =  wp_len + bp_len + wc_len + bc_len;
    uint64_t* random_share = new uint64_t[length];

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for(int i = 0; i < length; i++){
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length*sizeof(uint64_t));

    // Write wp share
    int offset = 0;
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < COMMON_DIM; j++){
            wp[i*COMMON_DIM + j] = (lin.w_p[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }

    // Write bp share
    for(int i = 0; i < COMMON_DIM; i++){
        bp[i] = (lin.b_p[i] - random_share[offset]) & mask_x;
        offset++;
    }

    // Write w_c share
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < NUM_CLASS; j++){
            wc[i*NUM_CLASS + j] = (lin.w_c[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }

    // Write b_c share
    for(int i = 0; i < NUM_CLASS; i++){
        bc[i] = (lin.b_c[i]- random_share[offset]) & mask_x;
        offset++;
    } 
}

void Bert::pc_bw_share_client(
    uint64_t* wp,
    uint64_t* bp,
    uint64_t* wc,
    uint64_t* bc
    ){
    int wp_len = COMMON_DIM*COMMON_DIM;
    int bp_len = COMMON_DIM;
    int wc_len = COMMON_DIM*NUM_CLASS;
    int bc_len = NUM_CLASS; 
    int length =  wp_len + bp_len + wc_len + bc_len;  

    uint64_t* share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    memcpy(wp, share, wp_len*sizeof(uint64_t));
    memcpy(bp, &share[wp_len], bp_len*sizeof(uint64_t));
    memcpy(wc, &share[wp_len + bp_len], wc_len*sizeof(uint64_t));
    memcpy(bc, &share[wp_len + bp_len + wc_len], bc_len*sizeof(uint64_t));
}

void Bert::softmax_v(
    HE* he,
    vector<Ciphertext> enc_v,
    uint64_t* s_softmax,
    uint64_t* s_v,
    uint64_t* s_softmax_v, 
    const FCMetadata &data
){
    if(party == ALICE){
        // Server

        vector<vector<vector<Plaintext>>> S2_pack = lin.preprocess_softmax_s2(he, s_softmax, data);

        vector<Ciphertext> enc_s1_pack(data.image_size * data.image_size * 12 / data.slot_count );

        recv_encrypted_vector(he->context, io, enc_s1_pack);

        vector<vector<vector<Plaintext>>> R_pack = 
            lin.preprocess_softmax_v_r(he, s_v, data);

        vector<Ciphertext> softmax_V_result(12 * data.image_size * data.filter_w / data.slot_count);

        lin.bert_softmax_V(he, enc_s1_pack, S2_pack, enc_v, R_pack, data, softmax_V_result);

        // send_encrypted_vector(io, softmax_V_result);

        uint64_t* softmax_v_server = new uint64_t[12 * data.image_size * data.filter_w];

        he_to_ss_server(he, softmax_V_result, softmax_v_server, true);

        lin.plain_cross_packing_postprocess_v(softmax_v_server, s_softmax_v, true, data);

        // for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
        //     s_softmax_v[i] = 0;
        // }

        delete[] softmax_v_server;

    } else{
        // Client

        uint64_t* softmax_v_client = new uint64_t[12 * data.image_size * data.filter_w];
        uint64_t* softmax_v_server = new uint64_t[12 * data.image_size * data.filter_w];

        vector<Ciphertext> enc_s1 = lin.preprocess_softmax_s1(
            he,
            s_softmax,
            data
        );

        send_encrypted_vector(io, enc_s1);
 
        // TODO: s_v column packing
        lin.client_S1_V_R(he, s_softmax, s_v, softmax_v_client, data); 

        int cts_len = 12 * data.image_size * data.filter_w / data.slot_count;
        he_to_ss_client(he, softmax_v_server, cts_len, data);

        lin.plain_cross_packing_postprocess_v(softmax_v_server, s_softmax_v, true, data);

        // vector<Ciphertext> enc_y_r(12 * data.image_size * data.filter_w / data.slot_count);
        // recv_encrypted_vector(he->context, io, enc_y_r);

        // uint64_t *y_r = new uint64_t[enc_y_r.size() * data.slot_count];

        // lin.bert_postprocess_V_enc(he, enc_y_r, y_r, data, true);

        // for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
        //     s_softmax_v[i] = y_r[i] + softmax_v_client[i];
        // }

        for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
            if (softmax_v_client[i] > he->plain_mod_2) {
                s_softmax_v[i] += softmax_v_client[i] - he->plain_mod;
            } else{
                s_softmax_v[i] += softmax_v_client[i];
            }
            s_softmax_v[i] = neg_mod((int64_t)s_softmax_v[i], (int64_t)he->plain_mod);
        }

        delete[] softmax_v_client;
        delete[] softmax_v_server;

    }

}

void Bert::print_p_share(uint64_t* s, uint64_t p, int len){
    if(party == ALICE){
        io->send_data(s, len*sizeof(uint64_t));
    } else{
        uint64_t* s1 = new uint64_t[len];
        io->recv_data(s1, len*sizeof(uint64_t));

        for(int i = 0; i < len; i++){
            uint64_t tmp = (s[i] + s1[i]) % p;
            if(tmp > (p / 2)){
                tmp -= p;
            }
            cout << (int64_t)tmp << " ";
        }
        cout << endl;
    }
}

void Bert::check_p_share(uint64_t* s, uint64_t p, int len, uint64_t* ref){
    if(party == ALICE){
        io->send_data(s, len*sizeof(uint64_t));
    } else{
        uint64_t* s1 = new uint64_t[len];
        io->recv_data(s1, len*sizeof(uint64_t));

        for(int i = 0; i < len; i++){
            uint64_t tmp = (s[i] + s1[i]) % p;
            if(tmp > (p / 2)){
                tmp -= p;
            }
            if((int64_t)tmp != (int64_t)ref[i]){
                cout << "Error: " << (int64_t)tmp << " " << (int64_t)ref[i] << endl;
            }
        }
    }
}

vector<double> Bert::run(string input_fname, string mask_fname){
    // Server: Alice
    // Client: Bob

    int input_dim = INPUT_DIM;
    if (prune){
        input_dim /= 2;
    }

    vector<uint64_t> softmax_mask;
    uint64_t h1_cache_12_original[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h1_cache_12[input_dim*COMMON_DIM] = {0};
    uint64_t h4_cache_12[input_dim*COMMON_DIM] = {0};
    uint64_t h98[COMMON_DIM] = {0};

    vector<Ciphertext> h1;
    vector<Ciphertext> h2;
    vector<Ciphertext> h4;
    vector<Ciphertext> h6;

    #ifdef BERT_PERF
    n_rounds += io->num_rounds;
    n_comm += io->counter;

    for(int i = 0; i < MAX_THREADS; i++){
        n_rounds += nl.iopackArr[i]->get_rounds();
        n_comm += nl.iopackArr[i]->get_comm();
    }

    auto t_linear1 = high_resolution_clock::now();
    auto t_linear2 = high_resolution_clock::now();
    auto t_linear3 = high_resolution_clock::now();
    auto t_linear4 = high_resolution_clock::now();
    #endif

    if(party == ALICE){
        // -------------------- Preparing -------------------- //
        // Receive cipher text input
        int cts_size = INPUT_DIM*COMMON_DIM / lin.data_lin1_0.slot_count;
        h1.resize(cts_size);

        #ifdef BERT_PERF
        t_linear1 = high_resolution_clock::now();
        #endif
        
        recv_encrypted_vector(lin.he_8192->context, io, h1);
        cout << "> Receive input cts from client " << endl;
    } else{
        cout << "> Loading inputs" << endl;
        vector<vector<uint64_t>> input_plain = read_data(input_fname);
        softmax_mask = read_bias(mask_fname, 128);

        cout << "> Repacking to column" << endl;

        // Column Packing
        vector<uint64_t> input_col(COMMON_DIM * INPUT_DIM);
        for (int j = 0; j < COMMON_DIM; j++){
            for (int i = 0; i < INPUT_DIM; i++){
                input_col[j*INPUT_DIM + i] = neg_mod(((int64_t)input_plain[i][j]) >> 7, (int64_t)lin.he_8192->plain_mod);
                if(prune){
                    h1_cache_12_original[i*COMMON_DIM + j] = input_plain[i][j];
                } else{
                    h1_cache_12[i*COMMON_DIM + j] = input_plain[i][j];
                }
                
            }
        }

        cout << "> Send to client" << endl;

        // Send cipher text input
        vector<Ciphertext> h1_cts = 
            lin.bert_efficient_preprocess_vec(lin.he_8192, input_col, lin.data_lin1_0);
            
        #ifdef BERT_PERF
        t_linear1 = high_resolution_clock::now();
        #endif

        send_encrypted_vector(io, h1_cts);
    }

    cout << "> --- Entering Attention Layers ---" << endl;
    for(int layer_id = 0; layer_id < ATTENTION_LAYERS; ++layer_id){
        {
            // -------------------- Linear #1 -------------------- //
            // w/ input pruning

            // Layer 0:
            // softmax input: 12*128*128
            // softmax output: 12*128*128
            // v: 12*128*64
            // softmax_v: 12*128*64

            // softmax_v(pruned): 12*64*64

            // Layer 1-11:
            // softmax input: 12*64*64
            // softmax output: 12*64*64
            // v: 12*64*64
            // softmax_v: 12*64*64

            // w/o input pruning
            
            // Layer 0-11:
            // softmax input: 12*128*128
            // softmax output(pruned): 128*128*128
            // v: 12*128*64
            // softmax_v: 12*128*64

            FCMetadata data = lin.data_lin1_0;
            if(layer_id > 0){
                data = lin.data_lin1_1;
            }

            int softmax_dim = data.image_size;

            int qk_size = PACKING_NUM*softmax_dim*softmax_dim;
            int v_size = PACKING_NUM*softmax_dim*OUTPUT_DIM;
            int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
            int softmax_v_size = PACKING_NUM*softmax_dim*OUTPUT_DIM; 
            int h2_col_size = PACKING_NUM*lin.data_lin1_1.image_size*OUTPUT_DIM; 

            int qk_v_size = qk_size + v_size;
            uint64_t* qk_v_cross = new uint64_t[qk_v_size];
            uint64_t* v_matrix_row = new uint64_t[v_size];
            uint64_t* softmax_input_row = new uint64_t[qk_size];
            uint64_t* softmax_output_row = new uint64_t[softmax_output_size];
            uint64_t* softmax_output_pack = new uint64_t[softmax_output_size];
            uint64_t* softmax_l_row = new uint64_t[softmax_output_size];
            uint64_t* softmax_l_col = new uint64_t[softmax_output_size];
            uint64_t* softmax_v_pack = new uint64_t[softmax_v_size];
            uint64_t* softmax_v_row = new uint64_t[softmax_v_size];
            uint64_t* softmax_v_col = new uint64_t[softmax_v_size];
            uint64_t* h2_col = new uint64_t[h2_col_size];
            vector<Ciphertext> enc_v;
                    
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #1 HE" << endl;
                auto t_start = std::chrono::high_resolution_clock::now();

                vector<Ciphertext> q_k_v = lin.linear_1(
                    lin.he_8192,
                    h1,
                    lin.pp_1[layer_id],
                    data
                );
                auto t_end = std::chrono::high_resolution_clock::now();
		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		
                cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << " (" << duration_ms << " ms)" << endl;


                int qk_offset = qk_size / data.slot_count;

                enc_v = { q_k_v.begin() + qk_offset, q_k_v.end()};

                parms_id_type parms_id = q_k_v[0].parms_id();
                shared_ptr<const SEALContext::ContextData> context_data = lin.he_8192->context->get_context_data(parms_id);
                
                #pragma omp parallel for
                for (int i = 0; i < qk_offset; i++) {
                    flood_ciphertext(q_k_v[i], context_data, SMUDGING_BITLEN_bert1);
                    lin.he_8192->evaluator->mod_switch_to_next_inplace(q_k_v[i]);
                    lin.he_8192->evaluator->mod_switch_to_next_inplace(q_k_v[i]);
                }

                vector<Ciphertext> q_k = {q_k_v.begin(), q_k_v.begin() + qk_offset};
                // vector<Ciphertext> v = { q_k_v.begin() + qk_offset, q_k_v.end()};

                he_to_ss_server(lin.he_8192, q_k, qk_v_cross, true);
                // he_to_ss_server(lin.he_8192, v, &qk_v_cross[qk_size], false);
            } else{
                int qk_cts_len = qk_size / lin.he_8192->poly_modulus_degree;
                int v_cts_len = v_size / lin.he_8192->poly_modulus_degree;
                he_to_ss_client(lin.he_8192, qk_v_cross, qk_cts_len, data);
                // he_to_ss_client(lin.he_8192, &qk_v_cross[qk_size], v_cts_len, data);
            }

            #ifdef BERT_PERF
            t_total_linear1 += interval(t_linear1);

            c_linear_1 += get_comm();
            r_linear_1 += get_round();
            auto t_repacking = high_resolution_clock::now();
            #endif 

            lin.plain_cross_packing_postprocess(
                qk_v_cross, 
                softmax_input_row,
                // we need row packing
                false,
                data);
            
            // lin.plain_cross_packing_postprocess_v(
            //     &qk_v_cross[qk_size], 
            //     v_matrix_row,
            //     true,
            //     data);

            #ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            // Scale: Q*V 22 
            nl.gt_p_sub(
                NL_NTHREADS,
                softmax_input_row,
                lin.he_8192->plain_mod,
                softmax_input_row,
                qk_size,
                NL_ELL,
                22,
                22
            );

            #ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_shift = high_resolution_clock::now();
            #endif 

            
            // Rescale QK to 12
            if(layer_id == 2){
                nl.right_shift(
                    NL_NTHREADS,
                    softmax_input_row,
                    21 - NL_SCALE,
                    softmax_input_row,
                    qk_size,
                    NL_ELL,
                    21
                );
            } else{
                nl.right_shift(
                    NL_NTHREADS,
                    softmax_input_row,
                    22 - NL_SCALE,
                    softmax_input_row,
                    qk_size,
                    NL_ELL,
                    22
                );
            }

            #ifdef BERT_SAVE_RESULTS
            FixArray softmax_input_row_pub = nl.to_public(softmax_input_row, qk_size, NL_ELL, NL_SCALE);
            FixArray v_matrix_row_pub = nl.to_public(v_matrix_row, v_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(softmax_input_row_pub.data, qk_size, 1, replace_2("./ppnlp/qk_matrix_X.txt", "X", to_string(layer_id)).c_str());
                save_to_file(v_matrix_row_pub.data, v_size, 1, replace_2("./ppnlp/v_matrix_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_softmax = high_resolution_clock::now();
            #endif 

            if (party == BOB){
                // Add mask
                for(int i = 0; i < PACKING_NUM; i++){
                    int offset_nm = i*softmax_dim*softmax_dim;
                    for(int j = 0; j < softmax_dim; j++){
                        int offset_row = j*softmax_dim;
                        for (int k = 0; k < softmax_dim; k++){
                            softmax_input_row[offset_nm + offset_row + k] += 
                                softmax_mask[k] * 4096;
                        }
                    }
                }
            }

            // Softmax
            nl.softmax(
                NL_NTHREADS,
                softmax_input_row,
                softmax_output_row,
                softmax_l_row,
                12*softmax_dim,
                softmax_dim,
                NL_ELL,
                NL_SCALE);

            #ifdef BERT_SAVE_RESULTS
            FixArray softmax_l_pub = nl.to_public(softmax_l_row, softmax_output_size, 25, 0);
            if(party == ALICE){
                save_to_file(softmax_l_pub.data, 1536*softmax_dim, 1, replace_2("./ppnlp/softmax_l_X.txt", "X", to_string(layer_id)).c_str());
            }
            FixArray softmax_pub = nl.to_public(softmax_output_row, softmax_output_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(softmax_pub.data, 1536*softmax_dim, 1, replace_2("./ppnlp/softmax_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_softmax += interval(t_softmax);
            c_softmax += get_comm();
            r_softmax += get_round();
            auto t_mul_v = high_resolution_clock::now();
            #endif 

            lin.preprocess_softmax(
                softmax_output_row,
                softmax_output_pack,
                data
            );

            if(party == ALICE){
                vector<Ciphertext> enc_softmax = ss_to_he_server(
                    lin.he_8192, 
                    softmax_output_pack,
                    softmax_output_size, 
                    NL_ELL);

                auto soft_mask_ct = lin.softmax_mask_ct_ct(lin.he_8192, data);
                auto pack_softmax_ct = lin.preprocess_softmax_s1_ct_ct(lin.he_8192, enc_softmax, data, soft_mask_ct);
                vector<Ciphertext> softmax_V_result(12 * data.image_size * data.filter_w / data.slot_count);
                lin.softmax_v(lin.he_8192, pack_softmax_ct, enc_v, data, softmax_V_result);
                he_to_ss_server(lin.he_8192, softmax_V_result, softmax_v_pack, true);
                
            } else{
                ss_to_he_client(
                    lin.he_8192, 
                    softmax_output_pack, 
                    softmax_output_size, 
                    NL_ELL);

                int cts_len = 12 * data.image_size * data.filter_w / data.slot_count;
                he_to_ss_client(lin.he_8192, softmax_v_pack, cts_len, data);
            }

            lin.plain_cross_packing_postprocess_v(softmax_v_pack, softmax_v_col, true, data);

            // for(int i = 0; i < softmax_output_size; i++){
            //     softmax_output_row[i] = 
            //         neg_mod(signed_val(softmax_output_row[i], NL_ELL), (int64_t)lin.he_8192->plain_mod);
            // }

            // softmax_v(
            //     lin.he_8192,
            //     enc_v,
            //     softmax_output_row,
            //     v_matrix_row,
            //     softmax_v_col,
            //     data
            // );

            #ifdef BERT_PERF
            t_total_mul_v += interval(t_mul_v);
            c_softmax_v += get_comm();
            r_softmax_v += get_round();
            auto t_gt_sub_2 = high_resolution_clock::now();
            #endif 


            nl.gt_p_sub(
                NL_NTHREADS,
                softmax_v_col,
                lin.he_8192->plain_mod,
                softmax_v_col,
                softmax_v_size,
                NL_ELL,
                23,
                6
            );

            #ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub_2);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_pruning = high_resolution_clock::now();
            #endif 


            #ifdef BERT_SAVE_RESULTS
            FixArray softmax_v_row_pub = nl.to_public(softmax_v_col, softmax_v_size, NL_ELL, 6);
            if(party == ALICE){
                save_to_file(softmax_v_row_pub.data, 768*softmax_dim, 1, replace_2("./ppnlp/softmax_v_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            if(prune && layer_id == 0){

                #pragma omp parallel for
                for(int i = 0; i < INPUT_DIM; i ++){
                    for(int j = 0; j < OUTPUT_DIM*PACKING_NUM; j++){
                        int row_offset = i*OUTPUT_DIM*PACKING_NUM + j;
                        int col_offset = j*INPUT_DIM + i;
                        softmax_v_row[row_offset] = softmax_v_col[col_offset];
                    }
                }
                
                #pragma omp parallel for
                for(int pack_id = 0; pack_id < PACKING_NUM; pack_id++){
                    for(int i = 0; i < INPUT_DIM; i ++){
                        for(int j = 0; j < INPUT_DIM; j++){
                            int row_offset = pack_id*INPUT_DIM*INPUT_DIM + i*INPUT_DIM + j;
                            int col_offset = pack_id*INPUT_DIM*INPUT_DIM + j*INPUT_DIM + i;
                            softmax_l_col[row_offset] = softmax_l_row[col_offset];
                        }
                    }
                }


                uint64_t* h2_row = new uint64_t[h2_col_size];

                nl.pruning(
                    softmax_l_col,
                    PACKING_NUM,
                    INPUT_DIM,
                    INPUT_DIM,
                    NL_ELL - NL_SCALE,
                    0,
                    softmax_v_row,
                    NL_ELL,
                    6,
                    h1_cache_12_original,
                    NL_ELL,
                    NL_SCALE,
                    INPUT_DIM,
                    COMMON_DIM,
                    h2_row,
                    h1_cache_12
                );

                lin.plain_col_packing_preprocess(
                    h2_row,
                    h2_col,
                    lin.he_8192->plain_mod,
                    lin.data_lin1_1.image_size,
                    COMMON_DIM
                );
            } else{
                memcpy(h2_col, softmax_v_col, h2_col_size*sizeof(uint64_t));
            }

            #ifdef BERT_SAVE_RESULTS
            FixArray h2_col_pub = nl.to_public(h2_col, h2_col_size, NL_ELL, 6);
            if(party == ALICE){
                save_to_file(h2_col_pub.data, h2_col_size, 1, replace_2("./ppnlp/h2_col_X.txt", "X", to_string(layer_id)).c_str());
            }

            FixArray h1_cache_12_pub = nl.to_public(h1_cache_12, h2_col_size, NL_ELL, 12);
            if(party == ALICE){
                save_to_file(h1_cache_12_pub.data, h2_col_size, 1, replace_2("./ppnlp/h1_cache_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_pruning += interval(t_pruning);
            c_pruning += get_comm();
            r_pruning += get_round();
            
            t_linear2 = high_resolution_clock::now();
            #endif 


            if(party == ALICE){
                h2 = ss_to_he_server(
                    lin.he_8192_tiny, 
                    h2_col,
                    h2_col_size, 
                    NL_ELL);
            } else{
                ss_to_he_client(
                    lin.he_8192_tiny, 
                    h2_col, 
                    h2_col_size, 
                    NL_ELL);
            }
            delete [] qk_v_cross;
            delete [] v_matrix_row;
            delete [] softmax_input_row;
            delete [] softmax_output_row;
            delete [] softmax_v_row;
            delete [] softmax_v_col;
            delete [] softmax_l_row;
            delete [] softmax_l_col;
            delete [] h2_col;
        }

        // -------------------- Linear #2 -------------------- //
        {
            FCMetadata data = lin.data_lin2;

            int ln_size = data.image_size*COMMON_DIM;
            int ln_cts_size = ln_size / lin.he_8192_tiny->poly_modulus_degree;
            uint64_t* ln_input_cross = new uint64_t[ln_size];
            uint64_t* ln_input_row = new uint64_t[ln_size];
            uint64_t* ln_output_row = new uint64_t[ln_size];
            uint64_t* ln_output_col = new uint64_t[ln_size];
            uint64_t* ln_wx = new uint64_t[ln_size];

            uint64_t* ln_weight = new uint64_t[ln_size];
            uint64_t* ln_bias = new uint64_t[ln_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #2 HE" << endl;
                vector<Ciphertext> h3 = lin.linear_2(
                    lin.he_8192_tiny,
                    h2, 
                    lin.pp_2[layer_id],
                    data
                );
                cout << "-> Layer - " << layer_id << ": Linear #2 HE done " << endl;
                he_to_ss_server(lin.he_8192_tiny, h3, ln_input_cross, true);
                ln_share_server(
                    layer_id,
                    lin.w_ln_1[layer_id],
                    lin.b_ln_1[layer_id],
                    ln_weight,
                    ln_bias,
                    data
                );
            } else{
                vector<Ciphertext> h3(ln_cts_size);
                he_to_ss_client(lin.he_8192_tiny, ln_input_cross, ln_cts_size, lin.data_lin2);
                ln_share_client(
                    ln_weight,
                    ln_bias,
                    data
                );
            }

            #ifdef BERT_PERF
            t_total_linear2 += interval(t_linear2);

            c_linear_2 += get_comm();
            r_linear_2 += get_round();
            auto t_repacking = high_resolution_clock::now();
            #endif 

            lin.plain_col_packing_postprocess(
                ln_input_cross,
                ln_input_row,
                false,
                data
            );

            #ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            nl.gt_p_sub(
                NL_NTHREADS,
                ln_input_row,
                lin.he_8192_tiny->plain_mod,
                ln_input_row,
                ln_size,
                NL_ELL,
                NL_SCALE,
                NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_input_row_pub = nl.to_public(ln_input_row, ln_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_input_row_pub.data, ln_size, 1, replace_2("./ppnlp/ln_input_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_ln_1 = high_resolution_clock::now();
            #endif 

            // nl.print_ss(ln_input_row, 16, NL_ELL, NL_SCALE);
            // return {};

            for(int i = 0; i < ln_size; i++){
                ln_input_row[i] += h1_cache_12[i];
            }

            // Layer Norm
            nl.layer_norm(
                NL_NTHREADS,
                ln_input_row,
                ln_output_row,
                ln_weight,
                ln_bias,
                data.image_size,
                COMMON_DIM,
                NL_ELL,
                NL_SCALE
            );

            // wx
            if(party == ALICE){
                vector<Ciphertext> ln = ss_to_he_server(
                    lin.he_8192_ln, 
                    ln_output_row,
                    ln_size, 
                    NL_ELL);
                
                vector<Ciphertext> ln_w = lin.w_ln(lin.he_8192_ln, ln, lin.w_ln_1_pt[layer_id]);
                he_to_ss_server(lin.he_8192_ln, ln_w, ln_wx, true);
            } else{
                ss_to_he_client(
                    lin.he_8192_ln, 
                    ln_output_row, 
                    ln_size, 
                    NL_ELL);
                int cts_size = ln_size / lin.he_8192_ln->poly_modulus_degree;
                he_to_ss_client(lin.he_8192_ln, ln_wx, cts_size ,data);
            }

            nl.gt_p_sub(
                NL_NTHREADS,
                ln_wx,
                lin.he_8192_ln->plain_mod,
                ln_wx,
                ln_size,
                NL_ELL,
                2*NL_SCALE,
                NL_SCALE
            );

            uint64_t ell_mask = (1ULL << (NL_ELL)) - 1;

            for(int i = 0; i < ln_size; i++){
                ln_wx[i] += ln_bias[i] & ell_mask;
            }


            #ifdef BERT_PERF
            t_total_ln_1 += interval(t_ln_1);
            c_ln1 += get_comm();
            r_ln1 += get_round();
            auto t_shift = high_resolution_clock::now();
            #endif 

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_output_row_pub = nl.to_public(ln_output_row, ln_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_output_row_pub.data, ln_size, 1, replace_2("./ppnlp/ln_output_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            memcpy(h4_cache_12, ln_wx, ln_size*sizeof(uint64_t));

            nl.right_shift(
                NL_NTHREADS,
                ln_wx,
                NL_SCALE - 5,
                ln_output_row,
                ln_size,
                NL_ELL,
                NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_repacking_2 = high_resolution_clock::now();
            #endif 

            // FixArray tmp = nl.to_public(ln_output_row, 128*768, 64, 5);
            // save_to_file(tmp.data, 128, 768, "./inter_result/linear3_input.txt");

            lin.plain_col_packing_preprocess(
                ln_output_row,
                ln_output_col,
                lin.he_8192_tiny->plain_mod,
                data.image_size,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_repacking += interval(t_repacking_2);

            t_linear3 = high_resolution_clock::now();
            #endif 

            if(party == ALICE){
                h4 = ss_to_he_server(
                    lin.he_8192_tiny, 
                    ln_output_col,
                    ln_size, 
                    NL_ELL);
            } else{
                ss_to_he_client(lin.he_8192_tiny, ln_output_col, ln_size, 
                NL_ELL);
            }


            delete[] ln_input_cross;
            delete[] ln_input_row;
            delete[] ln_output_row;
            delete[] ln_output_col;
            delete[] ln_weight;
            delete[] ln_bias;
        }
    

        // -------------------- Linear #3 -------------------- //
        {
            FCMetadata data = lin.data_lin3;

            int gelu_input_size = data.image_size*3072;
            int gelu_cts_size = gelu_input_size / lin.he_8192_tiny->poly_modulus_degree;
            uint64_t* gelu_input_cross =
                new uint64_t[gelu_input_size];
            uint64_t* gelu_input_col =
                new uint64_t[gelu_input_size];
            uint64_t* gelu_output_col =
                new uint64_t[gelu_input_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #3 HE" << endl;
                vector<Ciphertext> h5 = lin.linear_2(
                lin.he_8192_tiny,
                h4, 
                lin.pp_3[layer_id],
                data
                );

                cout << "-> Layer - " << layer_id << ": Linear #3 HE done " << endl;
                he_to_ss_server(lin.he_8192_tiny, h5, gelu_input_cross, true);
            } else{
                he_to_ss_client(lin.he_8192_tiny, gelu_input_cross, gelu_cts_size, data);
            }

            #ifdef BERT_PERF
            t_total_linear3 += interval(t_linear3);

            c_linear_3 += get_comm();
            r_linear_3 += get_round();
            auto t_repacking = high_resolution_clock::now();
            #endif 


            lin.plain_col_packing_postprocess(
                gelu_input_cross,
                gelu_input_col,
                true,
                data
            );

            #ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            // mod p
            nl.gt_p_sub(
                NL_NTHREADS,
                gelu_input_col,
                lin.he_8192_tiny->plain_mod,
                gelu_input_col,
                gelu_input_size,
                GELU_ELL,
                11,
                GELU_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray gelu_input_col_pub = nl.to_public(gelu_input_col, gelu_input_size, GELU_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(gelu_input_col_pub.data, gelu_input_size, 1, replace_2("./ppnlp/gelu_input_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            // nl.reduce(
            //     NL_NTHREADS,
            //     gelu_input_col,
            //     gelu_input_col,
            //     gelu_input_size,
            //     NL_ELL,
            //     GELU_ELL,
            //     GELU_SCALE
            // );

            #ifdef BERT_SAVE_RESULTS
            FixArray gelu_reduce_col_pub = nl.to_public(gelu_input_col, gelu_input_size, GELU_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(gelu_reduce_col_pub.data, gelu_input_size, 1, replace_2("./ppnlp/gelu_reduce_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_gelu = high_resolution_clock::now();
            #endif 

            nl.gelu(
                NL_NTHREADS,
                gelu_input_col,
                gelu_output_col,
                gelu_input_size,
                GELU_ELL,
                GELU_SCALE
            );

            #ifdef BERT_PERF
            t_total_gelu += interval(t_gelu);
            c_gelu += get_comm();
            r_gelu += get_round();
            auto t_shift = high_resolution_clock::now();
            #endif 

            // nl.right_shift(
            //     NL_NTHREADS,
            //     gelu_output_col,
            //     NL_SCALE - 4,
            //     gelu_output_col,
            //     gelu_input_size,
            //     GELU_ELL,
            //     NL_SCALE
            // );

            // // int tmp = get_comm();
            // // int tmp_round = get_round();

            // nl.cancel_wrap(
            //     NL_NTHREADS,
            //     gelu_output_col,
            //     gelu_output_col,
            //     gelu_input_size,
            //     GELU_ELL,
            //     NL_SCALE
            // );

            // cout << "Extension cost: " << get_comm() << " Bytes, " << get_round() << " rounds." << endl;

            // nl.convert_l_to_p(
            //     NL_NTHREADS,
            //     gelu_output_col,
            //     gelu_output_col,
            //     1,
            //     2,
            //     gelu_input_size,
            //     GELU_ELL,
            //     NL_SCALE
            // );

            // cout << "Extension cost: " << get_comm() << " Bytes, " << get_round() << " rounds." << endl;

            // return {};

            
            #ifdef BERT_SAVE_RESULTS
            FixArray gelu_output_col_pub = nl.to_public(gelu_output_col, gelu_input_size, GELU_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(gelu_output_col_pub.data, gelu_input_size, 1, replace_2("./ppnlp/gelu_output_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_SAVE_RESULTS
            FixArray gelu_cancel_col_pub = nl.to_public(gelu_output_col, gelu_input_size, GELU_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(gelu_cancel_col_pub.data, gelu_input_size, 1, replace_2("./ppnlp/gelu_cancel_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();

            t_linear4 = high_resolution_clock::now();
            #endif 

            // FixArray tmp = nl.to_public(gelu_output_col, 128*3072, 64, 4);
            // save_to_file(tmp.data, 128, 3072, "./inter_result/linear4_input.txt");

            // return 0;

            if(party == ALICE){
                h6 = ss_to_he_server(
                    lin.he_8192_tiny, 
                    gelu_output_col,
                    gelu_input_size, 
                    NL_ELL);
            } else{
                ss_to_he_client(
                    lin.he_8192_tiny, 
                    gelu_output_col, 
                    gelu_input_size, 
                    NL_ELL);
            }

            delete[] gelu_input_cross;
            delete[] gelu_input_col;
            delete[] gelu_output_col;
        }

        {
            FCMetadata data = lin.data_lin4;

            int ln_2_input_size = data.image_size*COMMON_DIM;
            int ln_2_cts_size = ln_2_input_size/lin.he_8192_tiny->poly_modulus_degree;

            uint64_t* ln_2_input_cross =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_input_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_col =
                new uint64_t[ln_2_input_size];

            uint64_t* ln_2_wx = 
                new uint64_t[ln_2_input_size];

            uint64_t* ln_weight_2 = new uint64_t[ln_2_input_size];
            uint64_t* ln_bias_2 = new uint64_t[ln_2_input_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #4 HE " << endl;

                vector<Ciphertext> h7 = lin.linear_2(
                    lin.he_8192_tiny,
                    h6, 
                    lin.pp_4[layer_id],
                    data
                );
                
                cout << "-> Layer - " << layer_id << ": Linear #4 HE done" << endl;
                he_to_ss_server(lin.he_8192_tiny, h7, ln_2_input_cross, true);
                ln_share_server(
                    layer_id,
                    lin.w_ln_2[layer_id],
                    lin.b_ln_2[layer_id],
                    ln_weight_2,
                    ln_bias_2,
                    data
                );
            } else{
                he_to_ss_client(lin.he_8192_tiny, ln_2_input_cross, ln_2_cts_size, data);
                ln_share_client(
                    ln_weight_2,
                    ln_bias_2,
                    data
                );
            }

            #ifdef BERT_PERF
            t_total_linear4 += interval(t_linear4);

            c_linear_4 += get_comm();
            r_linear_4 += get_round();
            auto t_repacking = high_resolution_clock::now();
            #endif 
            // Post Processing
            lin.plain_col_packing_postprocess(
                ln_2_input_cross,
                ln_2_input_row,
                false,
                data
            );

            #ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            // mod p
            if(layer_id == 9 || layer_id == 10){
                nl.gt_p_sub(
                    NL_NTHREADS,
                    ln_2_input_row,
                    lin.he_8192_tiny->plain_mod,
                    ln_2_input_row,
                    ln_2_input_size,
                    NL_ELL,
                    8,
                    NL_SCALE
                );
            } else{
                nl.gt_p_sub(
                    NL_NTHREADS,
                    ln_2_input_row,
                    lin.he_8192_tiny->plain_mod,
                    ln_2_input_row,
                    ln_2_input_size,
                    NL_ELL,
                    9,
                    NL_SCALE
                );
            }

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_2_input_row_pub = nl.to_public(ln_2_input_row, ln_2_input_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_2_input_row_pub.data, ln_2_input_size, 1, replace_2("./ppnlp/ln_2_input_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_ln = high_resolution_clock::now();
            #endif 

            for(int i = 0; i < ln_2_input_size; i++){
                ln_2_input_row[i] += h4_cache_12[i];
            }

            nl.layer_norm(
                NL_NTHREADS,
                ln_2_input_row,
                ln_2_output_row,
                ln_weight_2,
                ln_bias_2,
                data.image_size,
                COMMON_DIM,
                NL_ELL,
                NL_SCALE
            );

            // wx
            if(party == ALICE){
                vector<Ciphertext> ln = ss_to_he_server(
                    lin.he_8192_ln, 
                    ln_2_output_row,
                    ln_2_input_size, 
                    NL_ELL);
                vector<Ciphertext> ln_w = lin.w_ln(lin.he_8192_ln, ln, lin.w_ln_2_pt[layer_id]);
                he_to_ss_server(lin.he_8192_ln, ln_w, ln_2_wx, true);
            } else{
                ss_to_he_client(
                    lin.he_8192_ln, 
                    ln_2_output_row,
                    ln_2_input_size, 
                    NL_ELL);
                int cts_size = ln_2_input_size / lin.he_8192_ln->poly_modulus_degree;
                he_to_ss_client(lin.he_8192_ln, ln_2_wx, cts_size ,data);
            }

            nl.gt_p_sub(
                NL_NTHREADS,
                ln_2_wx,
                lin.he_8192_ln->plain_mod,
                ln_2_wx,
                ln_2_input_size,
                NL_ELL,
                2*NL_SCALE,
                NL_SCALE
            );

            uint64_t ell_mask = (1ULL << (NL_ELL)) - 1;

            for(int i = 0; i < ln_2_input_size; i++){
                ln_2_wx[i] += ln_bias_2[i] & ell_mask;
            }

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_2_output_row_pub = nl.to_public(ln_2_output_row, ln_2_input_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_2_output_row_pub.data, ln_2_input_size, 1, replace_2("./ppnlp/ln_2_output_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_ln_2 += interval(t_ln);
            c_ln2 += get_comm();
            r_ln2 += get_round();
            auto t_shift = high_resolution_clock::now();
            #endif 

            // update H1
            memcpy(h1_cache_12, ln_2_wx, ln_2_input_size*sizeof(uint64_t));

            // Rescale
            nl.right_shift(
                NL_NTHREADS,
                ln_2_wx,
                12 - 5,
                ln_2_output_row,
                ln_2_input_size,
                NL_ELL,
                NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_repacking_2 = high_resolution_clock::now();
            #endif 

            lin.plain_col_packing_preprocess(
                ln_2_output_row,
                ln_2_output_col,
                lin.he_8192_tiny->plain_mod,
                data.image_size,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_repacking += interval(t_repacking_2);

            t_linear1 = high_resolution_clock::now();
            #endif 

            if(layer_id == 11){
                // Using Scale of 12 as 
                memcpy(h98, h1_cache_12, COMMON_DIM*sizeof(uint64_t));
            } else{
                if(party == ALICE){
                    h1 = ss_to_he_server(
                    lin.he_8192, 
                    ln_2_output_col,
                    ln_2_input_size, 
                    NL_ELL);
                } else{
                    ss_to_he_client(
                        lin.he_8192, 
                        ln_2_output_col, 
                        ln_2_input_size, 
                        NL_ELL);
                }
            }

            delete[] ln_2_input_cross;
            delete[] ln_2_input_row;
            delete[] ln_2_output_row;
            delete[] ln_2_output_col;
            delete[] ln_weight_2;
            delete[] ln_bias_2;
        }
    }

    // Secret share Pool and Classification model
    uint64_t* wp = new uint64_t[COMMON_DIM*COMMON_DIM];
    uint64_t* bp = new uint64_t[COMMON_DIM];
    uint64_t* wc = new uint64_t[COMMON_DIM*NUM_CLASS];
    uint64_t* bc = new uint64_t[NUM_CLASS];

    uint64_t* h99 = new uint64_t[COMMON_DIM];
    uint64_t* h100 = new uint64_t[COMMON_DIM];
    uint64_t* h101 = new uint64_t[NUM_CLASS];

    cout << "-> Sharing Pooling and Classification params..." << endl;

    #ifdef BERT_PERF
    auto t_pc = high_resolution_clock::now();
    #endif 

    if(party == ALICE){
        pc_bw_share_server(
            wp,
            bp,
            wc,
            bc
        );
    } else{
        pc_bw_share_client(
            wp,
            bp,
            wc,
            bc
        );
    }

    // -------------------- POOL -------------------- //
    cout << "-> Layer - Pooling" << endl;
    nl.p_matrix_mul_iron(
        NL_NTHREADS,
        h98,
        wp,
        h99,
        1,
        COMMON_DIM,
        COMMON_DIM,
        NL_ELL,
        NL_ELL,
        NL_ELL,
        NL_SCALE,
        NL_SCALE,
        2*NL_SCALE
    );

    for(int i = 0; i < NUM_CLASS; i++){
        h99[i] += bp[i];
    }

    nl.right_shift(
        NL_NTHREADS,
        h99,
        NL_SCALE,
        h99,
        COMMON_DIM,
        NL_ELL,
        2*NL_SCALE
    );

    #ifdef BERT_PERF
    c_pc += get_comm();
    r_pc += get_round();
    auto t_tanh = high_resolution_clock::now();
    #endif 

    // -------------------- TANH -------------------- //
    nl.tanh(
        NL_NTHREADS,
        h99,
        h100,
        COMMON_DIM,
        NL_ELL,
        NL_SCALE
    );

    #ifdef BERT_PERF
    t_total_tanh += interval(t_tanh);
    c_tanh += get_comm();
    r_tanh += get_round();
    #endif 
    
    cout << "-> Layer - Classification" << endl;
    nl.n_matrix_mul_iron(
        NL_NTHREADS,
        h100,
        wc,
        h101,
        1,
        1,
        COMMON_DIM,
        NUM_CLASS,
        NL_ELL,
        NL_ELL,
        NL_ELL,
        NL_SCALE,
        NL_SCALE,
        2*NL_SCALE
    );

    for(int i = 0; i < NUM_CLASS; i++){
        h101[i] += bc[i];
    }

    nl.right_shift(
        1,
        h101,
        NL_SCALE,
        h101,
        NUM_CLASS,
        NL_ELL,
        2*NL_SCALE
    );

    #ifdef BERT_PERF
    c_pc += get_comm();
    r_pc += get_round();
    cout << "> [TIMING]: linear1 takes " << t_total_linear1 << " sec" << endl;
    cout << "> [TIMING]: linear2 takes " << t_total_linear2 << " sec" << endl;
    cout << "> [TIMING]: linear3 takes " << t_total_linear3 << " sec" << endl;
    cout << "> [TIMING]: linear4 takes " << t_total_linear4 << " sec" << endl;

    cout << "> [TIMING]: softmax takes " << t_total_softmax << " sec" << endl;
    cout << "> [TIMING]: pruning takes " << t_total_pruning << " sec" << endl;
    cout << "> [TIMING]: mul v takes " << t_total_mul_v << " sec" << endl;
    cout << "> [TIMING]: gelu takes " << t_total_gelu << " sec" << endl;
    cout << "> [TIMING]: ln_1 takes " << t_total_ln_1 << " sec" << endl;
    cout << "> [TIMING]: ln_2 takes " << t_total_ln_2 << " sec" << endl;
    cout << "> [TIMING]: tanh takes " << t_total_tanh << " sec" << endl;

    cout << "> [TIMING]: repacking takes " << t_total_repacking << " sec" << endl;
    cout << "> [TIMING]: gt_sub takes " << t_total_gt_sub << " sec" << endl;
    cout << "> [TIMING]: shift takes " << t_total_shift << " sec" << endl;

    cout << "> [TIMING]: conversion takes " << t_total_conversion << " sec" << endl;
    cout << "> [TIMING]: ln_share takes " << t_total_ln_share << " sec" << endl;


    cout << "> [TIMING]: Pool/Class takes " << interval(t_pc) << " sec" << endl; 

    cout << "> [NETWORK]: Linear 1 consumes: " << c_linear_1 << " bytes" << endl; 
    cout << "> [NETWORK]: Linear 2 consumes: " << c_linear_2 << " bytes" << endl; 
    cout << "> [NETWORK]: Linear 3 consumes: " << c_linear_3 << " bytes" << endl; 
    cout << "> [NETWORK]: Linear 4 consumes: " << c_linear_4 << " bytes" << endl; 

    cout << "> [NETWORK]: Softmax consumes: " << c_softmax << " bytes" << endl; 
    cout << "> [NETWORK]: GELU consumes: " << c_gelu << " bytes" << endl; 
    cout << "> [NETWORK]: Layer Norm 1 consumes: " << c_ln1 << " bytes" << endl; 
    cout << "> [NETWORK]: Layer Norm 2 consumes: " << c_ln2 << " bytes" << endl;
    cout << "> [NETWORK]: Tanh consumes: " << c_tanh << " bytes" << endl;

    cout << "> [NETWORK]: Softmax * V: " << c_softmax_v << " bytes" << endl; 
    cout << "> [NETWORK]: Pruning: " << c_pruning << " bytes" << endl; 
    cout << "> [NETWORK]: Shift consumes: " << c_shift << " bytes" << endl;
    cout << "> [NETWORK]: gt_sub consumes: " << c_gt_sub << " bytes" << endl;  
    
    cout << "> [NETWORK]: Pooling / C consumes: " << c_pc << " bytes" << endl; 

    cout << "> [NETWORK]: Linear 1 consumes: " << r_linear_1 << " rounds" << endl; 
    cout << "> [NETWORK]: Linear 2 consumes: " << r_linear_2 << " rounds" << endl; 
    cout << "> [NETWORK]: Linear 3 consumes: " << r_linear_3 << " rounds" << endl; 
    cout << "> [NETWORK]: Linear 4 consumes: " << r_linear_4 << " rounds" << endl; 

    cout << "> [NETWORK]: Softmax consumes: " << r_softmax << " rounds" << endl; 
    cout << "> [NETWORK]: GELU consumes: " << r_gelu << " rounds" << endl; 
    cout << "> [NETWORK]: Layer Norm 1 consumes: " << r_ln1 << " rounds" << endl; 
    cout << "> [NETWORK]: Layer Norm 2 consumes: " << r_ln2 << " rounds" << endl;
    cout << "> [NETWORK]: Tanh consumes: " << r_tanh << " rounds" << endl;

    cout << "> [NETWORK]: Softmax * V: " << r_softmax_v << " rounds" << endl; 
    cout << "> [NETWORK]: Pruning: " << r_pruning << " rounds" << endl; 
    cout << "> [NETWORK]: Shift consumes: " << r_shift << " rounds" << endl;
    cout << "> [NETWORK]: gt_sub consumes: " << r_gt_sub << " rounds" << endl;  
    
    cout << "> [NETWORK]: Pooling / C consumes: " << r_pc << " rounds" << endl; 

    // uint64_t total_rounds = io->num_rounds;
    // uint64_t total_comm = io->counter;

    // for(int i = 0; i < MAX_THREADS; i++){
    //     total_rounds += nl.iopackArr[i]->get_rounds();
    //     total_comm += nl.iopackArr[i]->get_comm();
    // }

    // cout << "> [NETWORK]: Communication rounds: " << total_rounds - n_rounds << endl; 
    // cout << "> [NETWORK]: Communication overhead: " << total_comm - n_comm << " bytes" << endl; 
    #endif 

    if(party == ALICE){
        io->send_data(h101, NUM_CLASS*sizeof(uint64_t));
        return {};
    } else{
        uint64_t* res = new uint64_t[NUM_CLASS];
        vector<double> dbl_result;
        io->recv_data(res, NUM_CLASS*sizeof(uint64_t));

        for(int i = 0; i < NUM_CLASS; i++){
            dbl_result.push_back((signed_val(res[i] + h101[i], NL_ELL)) / double(1LL << NL_SCALE));
        }
        return dbl_result;
    }
}
