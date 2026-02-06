#include "bert.h"

#ifdef BERT_PERF
double t_total_linear1 = 0;
double t_total_linear2 = 0;
double t_total_linear3 = 0;
double t_total_linear4 = 0;

double t_total_softmax = 0;
double t_total_mul_qk = 0;
double t_total_mul_v = 0;
double t_total_gelu = 0;
double t_total_ln_1 = 0;
double t_total_ln_2 = 0;
double t_total_tanh = 0;

double t_total_preproc = 0;
double t_total_postproc = 0;

double t_total_conversion = 0;

double t_total_ln_share = 0;

uint64_t c_linear_1 = 0;
uint64_t c_linear_2 = 0;
uint64_t c_linear_3 = 0;
uint64_t c_linear_4 = 0;
uint64_t c_softmax = 0;
uint64_t c_gelu = 0;
uint64_t c_ln1 = 0;
uint64_t c_ln2 = 0;
uint64_t c_softmax_v = 0;
uint64_t c_shift = 0;
uint64_t c_qk = 0;
uint64_t c_tanh = 0;
uint64_t c_pc = 0;

uint64_t r_linear_1 = 0;
uint64_t r_linear_2 = 0;
uint64_t r_linear_3 = 0;
uint64_t r_linear_4 = 0;
uint64_t r_softmax = 0;
uint64_t r_gelu = 0;
uint64_t r_ln1 = 0;
uint64_t r_ln2 = 0;
uint64_t r_softmax_v = 0;
uint64_t r_shift = 0;
uint64_t r_qk = 0;
uint64_t r_tanh = 0;
uint64_t r_pc = 0;

uint64_t n_rounds = 0;
uint64_t n_comm = 0;

//HE->SS SS->HE

double t_he2ss_after_lin1[ATTENTION_LAYERS] = {0};
double t_ss2he_before_lin2[ATTENTION_LAYERS] = {0};

double t_he2ss_after_lin2[ATTENTION_LAYERS] = {0};
double t_ss2he_before_lin3[ATTENTION_LAYERS] = {0};

double t_he2ss_after_lin3[ATTENTION_LAYERS] = {0};
double t_ss2he_before_lin4[ATTENTION_LAYERS] = {0};

double t_he2ss_after_lin4[ATTENTION_LAYERS] = {0};
double t_ss2he_before_next[ATTENTION_LAYERS] = {0};

inline uint64_t Bert::get_comm(){
    uint64_t total_comm = io->counter;
    for(int i = 0; i < MAX_THREADS; i++){
        total_comm += nl.iopackArr[i]->get_comm();
    }
    uint64_t ret = total_comm - n_comm;
    n_comm = total_comm;
    return ret;
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

Bert::Bert(int party, int port, string address, string model_path){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "> Setup Linear" << endl;
    this->lin = Linear(party, io);
    cout << "> Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port+1);

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

vector<Plaintext> Bert::he_to_ss_server(HE* he, vector<Ciphertext> in, const FCMetadata &data){
    
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    
    PRG128 prg;
    int dim = in.size();
    int slot_count = he->poly_modulus_degree;
    uint64_t *secret_share = new uint64_t[dim*slot_count];
	prg.random_data(secret_share, dim*slot_count*sizeof(uint64_t));
    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    for(int i = 0; i < dim*slot_count; i++){
        secret_share[i] &= mask_x;
    }

    vector<Plaintext> enc_noise;
    if(data.filter_w == OUTPUT_DIM){
        enc_noise = lin.preprocess_noise_1(he, secret_share, data);
    } else{
        enc_noise = lin.preprocess_noise_2(he, secret_share, data);
    }
    

    vector<uint64_t> p_2(slot_count, he->plain_mod_2);
    Plaintext pt_p_2 = lin.encode_vector(he, p_2.data(), data);

    vector<Ciphertext> cts;
    for(int i = 0; i < dim; i++){

        Ciphertext ct; 
        he->evaluator->sub_plain(in[i], enc_noise[i], ct);
        // he->evaluator->add_plain_inplace(ct, pt_p_2);
        // print_pt(he, pt, 8192);
        cts.push_back(ct);
    }
    
    send_encrypted_vector(io, cts);

    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 

    return enc_noise;
}

vector<Ciphertext> Bert::ss_to_he_server(HE* he, uint64_t* input, const FCMetadata &data){

    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 

    vector<Plaintext> share_server = lin.preprocess_ptr_plaintext(he, input, data);
    vector<Ciphertext> share_client(share_server.size());
    recv_encrypted_vector(he->context, io, share_client);
    for(int i = 0; i < share_server.size(); i++){
        he->evaluator->add_plain_inplace(share_client[i], share_server[i]);
    }
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
    return share_client;
}

vector<Plaintext> Bert::he_to_ss_client(HE* he, int length){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    vector<Ciphertext> cts(length);
    vector<Plaintext> pts;
    recv_encrypted_vector(he->context, io, cts);
    for(int i = 0; i < length; i++){
        Plaintext pt;
        //修改
        //he->decryptor->decrypt_keep_zero_coeff(cts[i], pt);
        he->decryptor->decrypt(cts[i], pt);
        pts.push_back(pt);
    }
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
    return pts;
}

void Bert::ss_to_he_client(HE* he, uint64_t* input, const FCMetadata &data){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    vector<Ciphertext> cts = lin.preprocess_ptr(he, input, data);
    send_encrypted_vector(io, cts);
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
}

void Bert::ln_share_server(
    int layer_id,
    vector<uint64_t> &wln_input,
    vector<uint64_t> &bln_input,
    uint64_t* wln,
    uint64_t* bln
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

    for(int i = 0; i < INPUT_DIM; i++){
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
    uint64_t* bln
){
    #ifdef BERT_PERF
    auto t_ln_share = high_resolution_clock::now();
    #endif 

    int length = 2*COMMON_DIM;

    uint64_t* share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    for(int i = 0; i < INPUT_DIM; i++){
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

vector<double> Bert::run(string input_fname, string mask_fname){
    // Server: Alice
    // Client: Bob

    vector<uint64_t> softmax_mask;

    uint64_t h1_cache_12[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h4_cache_12[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h98[COMMON_DIM] = {0};
    vector<Ciphertext> h1(lin.data_lin1.filter_h / lin.data_lin1.nw);
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
        #ifdef BERT_PERF
        t_linear1 = high_resolution_clock::now();
        #endif
        recv_encrypted_vector(lin.he_8192->context, io, h1);
        cout << "> Receive input cts from client " << endl;
    } else{
        cout << "> Loading inputs" << endl;
        vector<vector<uint64_t>> input_plain = read_data(input_fname);
        softmax_mask = read_bias(mask_fname, 128);

        // Column Packing
        vector<uint64_t> input_col(COMMON_DIM * INPUT_DIM);
        for (int j = 0; j < COMMON_DIM; j++){
            for (int i = 0; i < INPUT_DIM; i++){
                input_col[j*INPUT_DIM + i] = neg_mod((int64_t)input_plain[i][j], (int64_t)lin.he_8192->plain_mod);
                h1_cache_12[i*COMMON_DIM + j] = input_plain[i][j];
            }
        }

        // Send cipher text input
        vector<Ciphertext> h1_cts = 
            lin.preprocess_vec(lin.he_8192, input_col, lin.data_lin1);

        #ifdef BERT_PERF
        t_linear1 = high_resolution_clock::now();
        #endif

        send_encrypted_vector(io, h1_cts);
    }

    cout << "> --- Entering Attention Layers ---" << endl;
    for(int layer_id; layer_id < ATTENTION_LAYERS; ++layer_id){
        {

            // -------------------- Linear #1 -------------------- //
            int qk_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;

            int v_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;
            int softmax_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
            int att_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;  
            int qkv_size = 3*v_size;

            uint64_t* q_matrix_row = new uint64_t[v_size];
            uint64_t* k_matrix_row = new uint64_t[v_size];
            uint64_t* v_matrix_row = new uint64_t[v_size];

            uint64_t* k_trans_matrix_row = new uint64_t[v_size];

            uint64_t* softmax_input_row = new uint64_t[qk_size];
            uint64_t* softmax_output_row = new uint64_t[softmax_size];
            uint64_t* softmax_v_row = new uint64_t[att_size];
            uint64_t* h2_concate = new uint64_t[att_size];
            uint64_t* h2_col = new uint64_t[att_size];
            vector<Plaintext> pts;
                    
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #1 HE" << endl;
                                auto t_start = std::chrono::high_resolution_clock::now();
                vector<Ciphertext> q_k_v = lin.linear_1(
                    lin.he_8192,
                    h1,
                    lin.pp_1[layer_id],
                    lin.data_lin1
                );
                auto t_end = std::chrono::high_resolution_clock::now();
		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		
                cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << " (" << duration_ms << " ms)" << endl;
 #ifdef BERT_PERF
auto t_conv_he2ss_1 = high_resolution_clock::now();

#endif
                pts = he_to_ss_server(lin.he_8192, q_k_v, lin.data_lin1);
                
#ifdef BERT_PERF
double dt_conv_he2ss_1 = interval(t_conv_he2ss_1);
t_he2ss_after_lin1[layer_id] += dt_conv_he2ss_1;
cout << "> [CONV]: layer " << layer_id
     << " HE->SS after Linear1 = " << dt_conv_he2ss_1 << " sec" << endl;
#endif             
            } else{
                int cts_len = lin.data_lin1.filter_w / lin.data_lin1.kw * 3 * 12;
                pts = he_to_ss_client(lin.he_8192, cts_len);
            }

            #ifdef BERT_PERF
            t_total_linear1 += interval(t_linear1);

            c_linear_1 += get_comm();
            r_linear_1 += get_round();
            auto t_preproc = high_resolution_clock::now();
            #endif 

            auto qkv = lin.pt_postprocess_1(
                lin.he_8192,
                pts,
                lin.data_lin1,
                false
            );

            for(int i = 0; i < PACKING_NUM; i++){
                memcpy(
                    &q_matrix_row[i*INPUT_DIM*OUTPUT_DIM], 
                    qkv[i][0].data(), 
                    INPUT_DIM*OUTPUT_DIM*sizeof(uint64_t));
                
                memcpy(
                    &k_matrix_row[i*INPUT_DIM*OUTPUT_DIM], 
                    qkv[i][1].data(), 
                    INPUT_DIM*OUTPUT_DIM*sizeof(uint64_t));
                
                memcpy(
                    &v_matrix_row[i*INPUT_DIM*OUTPUT_DIM], 
                    qkv[i][2].data(), 
                    INPUT_DIM*OUTPUT_DIM*sizeof(uint64_t));
                
                transpose(
                    &k_matrix_row[i*INPUT_DIM*OUTPUT_DIM],
                    &k_trans_matrix_row[i*INPUT_DIM*OUTPUT_DIM],
                    INPUT_DIM,
                    OUTPUT_DIM
                );
            }

            nl.right_shift(
                NL_NTHREADS,
                q_matrix_row,
                NL_SCALE,
                q_matrix_row,
                v_size,
                NL_ELL,
                2*NL_SCALE
            );

            nl.right_shift(
                NL_NTHREADS,
                k_trans_matrix_row,
                NL_SCALE,
                k_trans_matrix_row,
                v_size,
                NL_ELL,
                2*NL_SCALE
            );

            nl.right_shift(
                NL_NTHREADS,
                v_matrix_row,
                NL_SCALE,
                v_matrix_row,
                v_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray q_matrix_row_pub = nl.to_public(q_matrix_row, v_size, NL_ELL, NL_SCALE);
            FixArray k_matrix_row_pub = nl.to_public(k_trans_matrix_row, v_size, NL_ELL, NL_SCALE);
            FixArray v_matrix_row_pub = nl.to_public(v_matrix_row, v_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(q_matrix_row_pub.data, v_size, 1, replace_2("./iron/q_matrix_X.txt", "X", to_string(layer_id)).c_str());
                save_to_file(k_matrix_row_pub.data, v_size, 1, replace_2("./iron/k_matrix_X.txt", "X", to_string(layer_id)).c_str());
                save_to_file(v_matrix_row_pub.data, v_size, 1, replace_2("./iron/v_matrix_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_mul_qk = high_resolution_clock::now();
            #endif 

            nl.n_matrix_mul_iron(
                NL_NTHREADS,
                q_matrix_row,
                k_trans_matrix_row,
                softmax_input_row,
                PACKING_NUM,
                INPUT_DIM,
                OUTPUT_DIM,
                INPUT_DIM,
                NL_ELL,
                NL_SCALE,
                NL_SCALE,
                NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_mul_qk += interval(t_mul_qk);
            c_qk += get_comm();
            r_qk += get_round();
            auto t_softmax = high_resolution_clock::now();
            #endif 

            if (party == BOB){
                // Add mask
                for(int i = 0; i < PACKING_NUM; i++){
                    int offset_nm = i*INPUT_DIM*INPUT_DIM;
                    for(int j = 0; j < INPUT_DIM; j++){
                        int offset_row = j*INPUT_DIM;
                        for (int k = 0; k < INPUT_DIM; k++){
                            softmax_input_row[offset_nm + offset_row + k] += 
                                softmax_mask[k] * 4096;
                        }
                    }
                }
            }
            
            // Softmax
            nl.softmax_iron(
                NL_NTHREADS,
                softmax_input_row,
                softmax_output_row,
                12*INPUT_DIM,
                INPUT_DIM,
                NL_ELL,
                NL_SCALE);

            #ifdef BERT_PERF
            t_total_softmax += interval(t_softmax);
            c_softmax += get_comm();
            r_softmax += get_round();
            auto t_mul_v = high_resolution_clock::now();
            #endif 

            // Rescale to 6
            nl.n_matrix_mul_iron(
                NL_NTHREADS,
                softmax_output_row,
                v_matrix_row,
                softmax_v_row,
                PACKING_NUM,
                INPUT_DIM,
                INPUT_DIM,
                OUTPUT_DIM,
                NL_ELL,
                NL_SCALE,
                NL_SCALE,
                NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray softmax_v_row_pub = nl.to_public(softmax_v_row, att_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(softmax_v_row_pub.data, att_size, 1, replace_2("./iron/softmax_v_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif


            lin.concat(softmax_v_row, h2_concate, 12, 128, 64); 

            #ifdef BERT_PERF
            t_total_mul_v += interval(t_mul_v);
            c_softmax_v += get_comm();
            r_softmax_v += get_round();
            auto t_postproc = high_resolution_clock::now();
            #endif  

            lin.plain_col_packing_preprocess(
                h2_concate,
                h2_col,
                lin.he_8192->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_postproc += interval(t_postproc);

            t_linear2 = high_resolution_clock::now();
            #endif 

#ifdef BERT_PERF
auto t_conv_ss2he_2 = high_resolution_clock::now();
#endif

if(party == ALICE){
    h2 = ss_to_he_server(lin.he_8192, h2_col, lin.data_lin2);
} else{
    ss_to_he_client(lin.he_8192, h2_col, lin.data_lin2);
}

#ifdef BERT_PERF
double dt_conv_ss2he_2 = interval(t_conv_ss2he_2);
t_ss2he_before_lin2[layer_id] += dt_conv_ss2he_2;
cout << "> [CONV]: layer " << layer_id
     << " SS->HE before Linear2 (h2) = " << dt_conv_ss2he_2 << " sec" << endl;
#endif


            delete [] q_matrix_row;
            delete [] k_matrix_row;
            delete [] v_matrix_row;
            delete [] k_trans_matrix_row;
            delete [] softmax_input_row;
            delete [] softmax_output_row;
            delete [] softmax_v_row;
            delete [] h2_concate;
            delete [] h2_col;
        }

        // -------------------- Linear #2 -------------------- //
        {
            int ln_size = INPUT_DIM*COMMON_DIM;
            uint64_t* ln_input_row = new uint64_t[ln_size];
            uint64_t* ln_output_row = new uint64_t[ln_size];
            uint64_t* ln_output_col = new uint64_t[ln_size];
            uint64_t* ln_weight = new uint64_t[ln_size];
            uint64_t* ln_bias = new uint64_t[ln_size];
            vector<Plaintext> pts;
            
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #2 HE" << endl;
                
                vector<Ciphertext> h3 = lin.linear_2(
                    lin.he_8192,
                    h2, 
                    lin.pp_2[layer_id],
                    lin.data_lin2
                );
            
                
                cout << "-> Layer - " << layer_id << ": Linear #2 HE done " << endl;

#ifdef BERT_PERF
auto t_conv_he2ss_3 = high_resolution_clock::now();
#endif
                pts = he_to_ss_server(lin.he_8192, h3, lin.data_lin2);
                ln_share_server(
                    layer_id,
                    lin.w_ln_1[layer_id],
                    lin.b_ln_1[layer_id],
                    ln_weight,
                    ln_bias
                );
 #ifdef BERT_PERF
double dt_conv_he2ss_3 = interval(t_conv_he2ss_3);
t_he2ss_after_lin2[layer_id] += dt_conv_he2ss_3;
cout << "> [CONV]: layer " << layer_id
     << " HE->SS after Linear2 = " << dt_conv_he2ss_3 << " sec" << endl;
#endif               
            } else{
                int cts_len = lin.data_lin2.filter_w / lin.data_lin2.kw;
                pts = he_to_ss_client(lin.he_8192, cts_len);
                ln_share_client(
                    ln_weight,
                    ln_bias
                );
            }


            #ifdef BERT_PERF
            t_total_linear2 += interval(t_linear2);

            c_linear_2 += get_comm();
            r_linear_2 += get_round();
            auto t_preproc = high_resolution_clock::now();
            #endif 

            lin.pt_postprocess_2(
                lin.he_8192,
                pts,
                ln_input_row,
                lin.data_lin2,
                false
            );

            nl.right_shift(
                NL_NTHREADS,
                ln_input_row,
                NL_SCALE,
                ln_input_row,
                ln_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_input_row_pub = nl.to_public(ln_input_row, ln_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_input_row_pub.data, ln_size, 1, replace_2("./iron/ln_input_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_ln_1 = high_resolution_clock::now();
            #endif 

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
                INPUT_DIM,
                COMMON_DIM,
                NL_ELL,
                NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_output_row_pub = nl.to_public(ln_output_row, ln_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_output_row_pub.data, ln_size, 1, replace_2("./iron/ln_output_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_ln_1 += interval(t_ln_1);
            c_ln1 += get_comm();
            r_ln1 += get_round();
            auto t_postproc = high_resolution_clock::now();
            #endif 

            memcpy(h4_cache_12, ln_output_row, ln_size*sizeof(uint64_t));

            lin.plain_col_packing_preprocess(
                ln_output_row,
                ln_output_col,
                lin.he_8192->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_postproc += interval(t_postproc);

            t_linear3 = high_resolution_clock::now();
            #endif 

   #ifdef BERT_PERF
auto t_conv_ss2he_4 = high_resolution_clock::now();
#endif

if(party == ALICE){
    h4 = ss_to_he_server(lin.he_8192, ln_output_col, lin.data_lin3);
} else{
    ss_to_he_client(lin.he_8192, ln_output_col, lin.data_lin3);
}

#ifdef BERT_PERF
double dt_conv_ss2he_4 = interval(t_conv_ss2he_4);
t_ss2he_before_lin3[layer_id] += dt_conv_ss2he_4;
cout << "> [CONV]: layer " << layer_id
     << " SS->HE before Linear3 (h4) = " << dt_conv_ss2he_4 << " sec" << endl;
#endif



            delete[] ln_input_row;
            delete[] ln_output_row;
            delete[] ln_output_col;
            delete[] ln_weight;
            delete[] ln_bias;
        }
    

        // -------------------- Linear #3 -------------------- //
        {
            int gelu_input_size = 128*3072;
            uint64_t* gelu_input_col =
                new uint64_t[gelu_input_size];
            uint64_t* gelu_output_col =
                new uint64_t[gelu_input_size];
            vector<Plaintext> pts;

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #3 HE" << endl;

                vector<Ciphertext> h5 = lin.linear_2(
                    lin.he_8192,
                    h4, 
                    lin.pp_3[layer_id],
                    lin.data_lin3
                );
            #ifdef BERT_PERF
auto t_conv_he2ss_5 = high_resolution_clock::now();
#endif

                cout << "-> Layer - " << layer_id << ": Linear #3 HE done " << endl;
                pts = he_to_ss_server(lin.he_8192, h5, lin.data_lin3);
                
                #ifdef BERT_PERF
double dt_conv_he2ss_5 = interval(t_conv_he2ss_5);
t_he2ss_after_lin3[layer_id] += dt_conv_he2ss_5;
cout << "> [CONV]: layer " << layer_id
     << " HE->SS after Linear3 = " << dt_conv_he2ss_5 << " sec" << endl;
#endif
            } else{
                int cts_len = lin.data_lin3.filter_w / lin.data_lin3.kw;
                pts = he_to_ss_client(lin.he_8192, cts_len);
            }

            lin.pt_postprocess_2(
                lin.he_8192,
                pts,
                gelu_input_col,
                lin.data_lin3,
                true
            );

            #ifdef BERT_PERF
            t_total_linear3 += interval(t_linear3);

            c_linear_3 += get_comm();
            r_linear_3 += get_round();
            auto t_preproc = high_resolution_clock::now();
            #endif 


            nl.right_shift(
                NL_NTHREADS,
                gelu_input_col,
                NL_SCALE,
                gelu_input_col,
                gelu_input_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray gelu_input_col_pub = nl.to_public(gelu_input_col, gelu_input_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(gelu_input_col_pub.data, gelu_input_size, 1, replace_2("./iron/gelu_input_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_gelu= high_resolution_clock::now();
            #endif 

            nl.gelu_iron(
                NL_NTHREADS,
                gelu_input_col,
                gelu_output_col,
                gelu_input_size,
                NL_ELL,
                NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray gelu_output_col_pub = nl.to_public(gelu_output_col, gelu_input_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(gelu_output_col_pub.data, gelu_input_size, 1, replace_2("./iron/gelu_output_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_gelu += interval(t_gelu);
            c_gelu += get_comm();
            r_gelu += get_round();

            t_linear4 = high_resolution_clock::now();
            #endif 

#ifdef BERT_PERF
auto t_conv_ss2he_6 = high_resolution_clock::now();
#endif

if(party == ALICE){
    h6 = ss_to_he_server(lin.he_8192, gelu_output_col, lin.data_lin4);
} else{
    ss_to_he_client(lin.he_8192, gelu_output_col, lin.data_lin4);
}

#ifdef BERT_PERF
double dt_conv_ss2he_6 = interval(t_conv_ss2he_6);
t_ss2he_before_lin4[layer_id] += dt_conv_ss2he_6;
cout << "> [CONV]: layer " << layer_id
     << " SS->HE before Linear4 (h6) = " << dt_conv_ss2he_6 << " sec" << endl;
#endif


            delete[] gelu_input_col;
            delete[] gelu_output_col;
        }

        // -------------------- Linear #4 -------------------- //
        {
            int ln_2_input_size = INPUT_DIM*COMMON_DIM;

            uint64_t* ln_2_input_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_col =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_weight_2 = new uint64_t[ln_2_input_size];
            uint64_t* ln_bias_2 = new uint64_t[ln_2_input_size];
            vector<Plaintext> pts;

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #4 HE " << endl;
                
                vector<Ciphertext> h7 = lin.linear_2(
                    lin.he_8192,
                    h6, 
                    lin.pp_4[layer_id],
                    lin.data_lin4
                );

                cout << "-> Layer - " << layer_id << ": Linear #4 HE done" << endl;
               #ifdef BERT_PERF
auto t_conv_he2ss_7 = high_resolution_clock::now();
#endif
                pts = he_to_ss_server(lin.he_8192, h7, lin.data_lin4);
                ln_share_server(
                    layer_id,
                    lin.w_ln_2[layer_id],
                    lin.b_ln_2[layer_id],
                    ln_weight_2,
                    ln_bias_2
                );
                #ifdef BERT_PERF
double dt_conv_he2ss_7 = interval(t_conv_he2ss_7);
t_he2ss_after_lin4[layer_id] += dt_conv_he2ss_7;
cout << "> [CONV]: layer " << layer_id
     << " HE->SS after Linear4 = " << dt_conv_he2ss_7 << " sec" << endl;
#endif
            } else{
                int cts_len = lin.data_lin4.filter_w / lin.data_lin4.kw;
                pts = he_to_ss_client(lin.he_8192, cts_len);
                ln_share_client(
                    ln_weight_2,
                    ln_bias_2
                );
            }

            #ifdef BERT_PERF
            t_total_linear4 += interval(t_linear4);

            c_linear_4 += get_comm();
            r_linear_4 += get_round();
            auto t_preproc = high_resolution_clock::now();
            #endif 

            lin.pt_postprocess_2(
                lin.he_8192,
                pts,
                ln_2_input_row,
                lin.data_lin4,
                false
            );

            nl.right_shift(
                NL_NTHREADS,
                ln_2_input_row,
                NL_SCALE,
                ln_2_input_row,
                ln_2_input_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_2_input_row_pub = nl.to_public(ln_2_input_row, ln_2_input_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_2_input_row_pub.data, ln_2_input_size, 1, replace_2("./iron/ln_2_input_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_ln_2 = high_resolution_clock::now();
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
                INPUT_DIM,
                COMMON_DIM,
                NL_ELL,
                NL_SCALE
            );

            #ifdef BERT_SAVE_RESULTS
            FixArray ln_2_output_row_pub = nl.to_public(ln_2_output_row, ln_2_input_size, NL_ELL, NL_SCALE);
            if(party == ALICE){
                save_to_file(ln_2_output_row_pub.data, ln_2_input_size, 1, replace_2("./iron/ln_2_output_X.txt", "X", to_string(layer_id)).c_str());
            }
            #endif

            #ifdef BERT_PERF
            t_total_ln_2 += interval(t_ln_2);
            c_ln2 += get_comm();
            r_ln2 += get_round();
            auto t_postproc = high_resolution_clock::now();
            #endif 


            // update H1
            memcpy(h1_cache_12, ln_2_output_row, ln_2_input_size*sizeof(uint64_t));

            lin.plain_col_packing_preprocess(
                ln_2_output_row,
                ln_2_output_col,
                lin.he_8192->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_postproc += interval(t_postproc);

            t_linear1 = high_resolution_clock::now();
            #endif 

if(layer_id == 11){
    memcpy(h98, h1_cache_12, COMMON_DIM*sizeof(uint64_t));
} else{

    #ifdef BERT_PERF
    auto t_conv_ss2he_8 = high_resolution_clock::now();
    #endif

    if(party == ALICE){
        h1 = ss_to_he_server(lin.he_8192, ln_2_output_col, lin.data_lin1);
    } else{
        ss_to_he_client(lin.he_8192, ln_2_output_col, lin.data_lin1);
    }

    #ifdef BERT_PERF
    double dt_conv_ss2he_8 = interval(t_conv_ss2he_8);
    t_ss2he_before_next[layer_id] += dt_conv_ss2he_8;
    cout << "> [CONV]: layer " << layer_id
         << " SS->HE before next layer (h1) = " << dt_conv_ss2he_8 << " sec" << endl;
    #endif
}


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
    nl.n_matrix_mul_iron(
        NL_NTHREADS,
        h98,
        wp,
        h99,
        1,
        1,
        COMMON_DIM,
        COMMON_DIM,
        NL_ELL,
        NL_SCALE,
        NL_SCALE,
        2*NL_SCALE
    );

    for(int i = 0; i < COMMON_DIM; i++){
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
    nl.tanh_iron(
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

    cout << "> [TIMING]: mul qk takes " << t_total_mul_qk << " sec" << endl;
    cout << "> [TIMING]: softmax takes " << t_total_softmax << " sec" << endl;
    cout << "> [TIMING]: mul v takes " << t_total_mul_v << " sec" << endl;
    cout << "> [TIMING]: gelu takes " << t_total_gelu << " sec" << endl;
    cout << "> [TIMING]: ln_1 takes " << t_total_ln_1 << " sec" << endl;
    cout << "> [TIMING]: ln_2 takes " << t_total_ln_2 << " sec" << endl;

    cout << "> [TIMING]: preprocessing takes " << t_total_preproc << " sec" << endl;
    cout << "> [TIMING]: postprocessing takes " << t_total_postproc << " sec" << endl;

    cout << "> [TIMING]: conversion takes " << t_total_conversion << " sec" << endl;
    cout << "> [TIMING]: ln_share takes " << t_total_ln_share << " sec" << endl;


    cout << "> [TIMING]: pool/classification takes" << interval(t_pc) << " sec" << endl; 

    cout << "> [NETWORK]: Linear 1 consumes: " << r_linear_1 << " rounds" << endl; 
    cout << "> [NETWORK]: Linear 2 consumes: " << r_linear_2 << " rounds" << endl; 
    cout << "> [NETWORK]: Linear 3 consumes: " << r_linear_3 << " rounds" << endl; 
    cout << "> [NETWORK]: Linear 4 consumes: " << r_linear_4 << " rounds" << endl; 

    cout << "> [NETWORK]: Softmax consumes: " << r_softmax << " rounds" << endl; 
    cout << "> [NETWORK]: GELU consumes: " << r_gelu << " rounds" << endl; 
    cout << "> [NETWORK]: Layer Norm 1 consumes: " << r_ln1 << " rounds" << endl; 
    cout << "> [NETWORK]: Layer Norm 2 consumes: " << r_ln2 << " rounds" << endl;
    cout << "> [TIMING]: tanh takes " << t_total_tanh << " sec" << endl;

    cout << "> [NETWORK]: Softmax * V: " << r_softmax_v << " rounds" << endl; 
    cout << "> [NETWORK]: Shift consumes: " << r_shift << " rounds" << endl;
    cout << "> [NETWORK]: qk consumes: " << r_qk << " rounds" << endl;  
    
    cout << "> [NETWORK]: Pooling / C consumes: " << r_pc << " rounds" << endl; 

    // uint64_t total_rounds = io->num_rounds;
    // uint64_t total_comm = io->counter;

    // for(int i = 0; i < MAX_THREADS; i++){
    //     total_rounds += nl.iopackArr[i]->get_rounds();
    //     total_comm += nl.iopackArr[i]->get_comm();
    // }

    // cout << "> [NETWORK]: Communication rounds: " << total_rounds - n_rounds << endl; 
    // cout << "> [NETWORK]: Communication overhead: " << total_comm - n_comm << " bytes" << endl; 

    cout << "> [NETWORK]: Linear 1 consumes: " << c_linear_1 << " bytes" << endl; 
    cout << "> [NETWORK]: Linear 2 consumes: " << c_linear_2 << " bytes" << endl; 
    cout << "> [NETWORK]: Linear 3 consumes: " << c_linear_3 << " bytes" << endl; 
    cout << "> [NETWORK]: Linear 4 consumes: " << c_linear_4 << " bytes" << endl; 

    cout << "> [NETWORK]: Softmax consumes: " << c_softmax << " bytes" << endl; 
    cout << "> [NETWORK]: GELU consumes: " << c_gelu << " bytes" << endl; 
    cout << "> [NETWORK]: Layer Norm 1 consumes: " << c_ln1 << " bytes" << endl; 
    cout << "> [NETWORK]: Layer Norm 2 consumes: " << c_ln2 << " bytes" << endl;
    cout << "> [NETWORK]: Tanh consumes: " << c_tanh << " bytes" << endl;

    cout << "> [NETWORK]: Q*K consumes: " << c_qk << " bytes" << endl; 
    cout << "> [NETWORK]: Softmax * V: " << c_softmax_v << " bytes" << endl; 
    cout << "> [NETWORK]: Shift consumes: " << c_shift << " bytes" << endl; 
    
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
    cout << "> [NETWORK]: Shift consumes: " << r_shift << " rounds" << endl;
    cout << "> [NETWORK]: qk consumes: " << r_qk << " rounds" << endl;  
    
    cout << "> [NETWORK]: Pooling / C consumes: " << r_pc << " rounds" << endl; 

    #endif 
    
#ifdef BERT_PERF
for(int i = 0; i < ATTENTION_LAYERS; i++){
    cout << "===== [CONV BREAKDOWN] layer " << i << " =====" << endl;
    cout << "HE->SS after Linear1: " << t_he2ss_after_lin1[i] << " sec" << endl;
    cout << "SS->HE before Linear2: " << t_ss2he_before_lin2[i] << " sec" << endl;
    cout << "HE->SS after Linear2: " << t_he2ss_after_lin2[i] << " sec" << endl;
    cout << "SS->HE before Linear3: " << t_ss2he_before_lin3[i] << " sec" << endl;
    cout << "HE->SS after Linear3: " << t_he2ss_after_lin3[i] << " sec" << endl;
    cout << "SS->HE before Linear4: " << t_ss2he_before_lin4[i] << " sec" << endl;
    cout << "HE->SS after Linear4: " << t_he2ss_after_lin4[i] << " sec" << endl;
    cout << "SS->HE before next layer: " << t_ss2he_before_next[i] << " sec" << endl;
}
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
