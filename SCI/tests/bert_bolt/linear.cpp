#include "linear.h"


void print_pt_l(HE* he, Plaintext &pt, int len) {
    vector<int64_t> dest(len, 0ULL);
    he->encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < 10; i++){
        cout << dest[i] << " ";
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

void print_ct_l(HE* he, Ciphertext &ct, int len){
    Plaintext pt;
    he->decryptor->decrypt(ct, pt);
    cout << "Noise budget: ";
    cout << YELLOW << he->decryptor->invariant_noise_budget(ct) << " ";
    cout << RESET << endl;
    print_pt_l(he, pt, len);
}

Linear::Linear(){}

Linear::Linear(int party, NetIO *io, bool prune) {
	this->party = party;
	this->io = io;
	this->he_8192 = new HE(
		party,
		io,
		8192,
		{54, 54, 55, 55},
		536903681
    );

    // this->he_8192_tiny = new HE(
    //     party,
    //     io,
    //     8192,
    //     {54, 54, 55, 55},
	// 	536903681
    // );

    this->he_8192_tiny = new HE(
        party,
        io,
        8192,
        {60, 60, 60},
		557057
    );

    this->he_8192_ln = new HE(
        party,
        io,
        8192,
        {54, 54, 55, 55},
		4295049217
    );

    this->p_mod = prime_mod;

	// this->he_4096 = new HE(
	// 	party,
	// 	io,
	// 	4096,
	// 	{54, 55},
	// 	65537
    // );

    pp_1.resize(ATTENTION_LAYERS);
    pp_2.resize(ATTENTION_LAYERS);
    pp_3.resize(ATTENTION_LAYERS);
    pp_4.resize(ATTENTION_LAYERS);

    w_ln_1_pt.resize(ATTENTION_LAYERS);
    w_ln_2_pt.resize(ATTENTION_LAYERS);

    data_lin1_0.filter_h = COMMON_DIM;
    data_lin1_0.filter_w = OUTPUT_DIM;
    data_lin1_0.image_size = INPUT_DIM;
    data_lin1_0.slot_count = 8192;

    int input_dim = INPUT_DIM;
    if(prune){
        input_dim /= 2;
    }

    data_lin1_1.filter_h = COMMON_DIM;
    data_lin1_1.filter_w = OUTPUT_DIM;
    data_lin1_1.image_size = input_dim;
    data_lin1_1.slot_count = 8192;

    data_lin2.filter_h = COMMON_DIM;
    data_lin2.filter_w = COMMON_DIM;
    data_lin2.image_size = input_dim;
    data_lin2.slot_count = 8192;

    data_lin3.filter_h = COMMON_DIM;
    data_lin3.filter_w = INTER_DIM;
    data_lin3.image_size = input_dim;
    data_lin3.slot_count = 8192;

    data_lin4.filter_h = INTER_DIM;
    data_lin4.filter_w = COMMON_DIM;
    data_lin4.image_size = input_dim;
    data_lin4.slot_count = 8192;
}

Linear::~Linear() {

}

vector<Plaintext> Linear::generate_cross_packing_masks(HE* he, const FCMetadata &data) {
    vector<Plaintext> result(data.image_size * 2);
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.slot_count, 0ULL);
        vector<uint64_t> mask2(data.slot_count, 0ULL);
        for (int k = 0; k < data.image_size - i; k++) {
            mask1[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask1[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        for (int k = data.image_size - i; k < data.image_size; k++) {
            mask2[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask2[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        Plaintext pt1, pt2;
        he->encoder->encode(mask1, pt1);
        he->encoder->encode(mask2, pt2);
        result[i] = pt1;
        result[i + data.image_size] = pt2;
    }
    return result;
}

PreprocessParams_1 Linear::params_preprocessing_ct_ct(
    HE* he,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    const FCMetadata &data
){
    PreprocessParams_1 pp;

    uint64_t plain_mod = he->plain_mod;

    vector<vector<vector<uint64_t>>> weights_q(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));
    vector<vector<vector<uint64_t>>> weights_k(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));
    vector<vector<vector<uint64_t>>> weights_v(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));

    vector<vector<uint64_t>> bias_q(12, vector<uint64_t>(data.filter_w));
    vector<vector<uint64_t>> bias_k(12, vector<uint64_t>(data.filter_w));
    vector<vector<uint64_t>> bias_v(12, vector<uint64_t>(data.filter_w));

    for (int packing_index = 0; packing_index < 12; packing_index++) {
		for (int i = 0; i < COMMON_DIM; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                weights_q[packing_index][i][j] = neg_mod((int64_t)w_q[packing_index][i][j], (int64_t)plain_mod);
                weights_k[packing_index][i][j] = neg_mod((int64_t)w_k[packing_index][i][j], (int64_t)plain_mod);
                weights_v[packing_index][i][j] = neg_mod((int64_t)w_v[packing_index][i][j], (int64_t)plain_mod);
            }
        }
        for (int i = 0; i < OUTPUT_DIM; i++) {
            bias_q[packing_index][i] = neg_mod((int64_t)b_q[packing_index][i], (int64_t)plain_mod);
            bias_k[packing_index][i] = neg_mod((int64_t)b_k[packing_index][i], (int64_t)plain_mod);
            bias_v[packing_index][i] = neg_mod((int64_t)b_v[packing_index][i], (int64_t)plain_mod);
        }
    }

    pp.wq_pack = bert_cross_packing_single_matrix(he, weights_q, data);
    pp.wk_pack = bert_cross_packing_single_matrix(he, weights_k, data);
    pp.wv_pack = bert_cross_packing_single_matrix(he, weights_v, data);

    pp.bq_pack = bert_cross_packing_bias(he, bias_q, data);
    pp.bk_pack = bert_cross_packing_bias(he, bias_k, data);
    pp.bv_pack = bert_cross_packing_bias(he, bias_v, data);

	pp.cross_masks = generate_cross_packing_masks(he, data);

    return pp;
}

PreprocessParams_2 Linear::params_preprocessing_ct_pt(
    HE* he,
    int32_t input_dim, 
    int32_t common_dim, 
    int32_t output_dim,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    const FCMetadata &data
){
    PreprocessParams_2 pp;
    uint64_t plain_mod = he->plain_mod;

    vector<uint64_t *> matrix_mod_p1(common_dim);
    vector<uint64_t *> matrix_mod_p2(common_dim);

    vector<uint64_t *> matrix1(common_dim);
    vector<uint64_t *> matrix2(common_dim);
    for (int i = 0; i < common_dim; i++) {
        matrix_mod_p1[i] = new uint64_t[output_dim / 2];
        matrix_mod_p2[i] = new uint64_t[output_dim / 2];

        matrix1[i] = new uint64_t[output_dim / 2];
        matrix2[i] = new uint64_t[output_dim / 2];

        for (int j = 0; j < output_dim / 2; j++) {
            matrix_mod_p1[i][j] = neg_mod((int64_t)w[i][j], (int64_t)plain_mod);
            matrix_mod_p2[i][j] = neg_mod((int64_t)w[i][j + output_dim / 2], (int64_t)plain_mod);
        }
    }
    for (int i = 0; i < output_dim; i++) {
        b[i] = neg_mod((int64_t)b[i], (int64_t)plain_mod);
    }
    pp.cross_mat_single = bert_cross_packing_single_matrix_2(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
    pp.cross_bias_single = bert_cross_packing_bias_2(he, b.data(), data);
    return pp;
}

void Linear::weights_preprocess(BertModel &bm){
    #pragma omp parallel for
    for(int i = 0; i < ATTENTION_LAYERS; i++){
        FCMetadata data = data_lin1_0;
        if(i > 0){
            data = data_lin1_1;
        }
        pp_1[i] = params_preprocessing_ct_ct(
            he_8192,
            bm.w_q[i],
            bm.w_k[i],
            bm.w_v[i],
            bm.b_q[i],
            bm.b_k[i],
            bm.b_v[i],
            data
        );

        pp_2[i] = params_preprocessing_ct_pt(
            he_8192_tiny,
            INPUT_DIM,
            COMMON_DIM,
            COMMON_DIM,
            bm.w_o[i],
            bm.b_o[i],
            data_lin2
        );

        pp_3[i] = params_preprocessing_ct_pt(
            he_8192_tiny,
            INPUT_DIM,
            COMMON_DIM,
            INTER_DIM,
            bm.w_i_1[i],
            bm.b_i_1[i],
            data_lin3
        );

        pp_4[i] = params_preprocessing_ct_pt(
            he_8192_tiny,
            INPUT_DIM,
            INTER_DIM,
            COMMON_DIM,
            bm.w_i_2[i],
            bm.b_i_2[i],
            data_lin4
        );

        int slot_count = he_8192_ln->poly_modulus_degree;
        int w_length = data_lin2.image_size*COMMON_DIM;
        uint64_t* w_1 = new uint64_t[w_length];
        uint64_t* w_2 = new uint64_t[w_length];

        vector<Plaintext> pt_w_1(w_length / slot_count);
        vector<Plaintext> pt_w_2(w_length / slot_count);
        for(int j = 0; j < data_lin2.image_size; j++){
            memcpy(&w_1[j*COMMON_DIM], bm.w_ln_1[i].data(), COMMON_DIM*sizeof(uint64_t));
            memcpy(&w_2[j*COMMON_DIM], bm.w_ln_2[i].data(), COMMON_DIM*sizeof(uint64_t));
        }
        
        for(int i = 0; i < pt_w_1.size(); i++){
            vector<uint64_t> tmp_1(&w_1[i*slot_count], &w_1[(i+1)*slot_count]);
            Plaintext pt_1;
            he_8192_ln->encoder->encode(tmp_1, pt_1);
            pt_w_1[i] = pt_1;

            vector<uint64_t> tmp_2(&w_2[i*slot_count], &w_2[(i+1)*slot_count]);
            Plaintext pt_2;
            he_8192_ln->encoder->encode(tmp_2, pt_2);
            pt_w_2[i] = pt_2;
        }

        w_ln_1_pt[i] = pt_w_1;
        w_ln_2_pt[i] = pt_w_2;
    }

    w_ln_1 = bm.w_ln_1;
    b_ln_1 = bm.b_ln_1;

    w_ln_2 = bm.w_ln_2;
    b_ln_2 = bm.b_ln_2;

    w_c = bm.w_c;
    b_c = bm.b_c;
    w_p = bm.w_p;
    b_p = bm.b_p;
}



vector<Ciphertext> Linear::linear_1(
	HE* he,
	vector<Ciphertext> input_cts, 
	PreprocessParams_1 &pp,
	const FCMetadata &data) 
{


	// Step 1: 分配 Cipher_plain_results
	size_t cipher_plain_size = data.image_size * data.filter_w * 3 * 12 / data.slot_count;

	vector<Ciphertext> Cipher_plain_results(cipher_plain_size);
	
	// Step 2: 调用 bert_cipher_plain_bsgs

        auto t_start = std::chrono::high_resolution_clock::now();
	bert_cipher_plain_bsgs(
		he, 
		input_cts, 
		pp.wq_pack,
		pp.wk_pack,
		pp.wv_pack,
		pp.bq_pack,
		pp.bk_pack,
		pp.bv_pack,
		data, 
		Cipher_plain_results
	);
        auto t_end = std::chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		
        cout << "cipher_plain" << " (" << duration_ms << " ms)" << endl;

	// Step 3: 分配 HE_result
	size_t he_result_size = data.image_size * data.image_size * 12 / data.slot_count 
		+ data.image_size * data.filter_w * 12 / data.slot_count;
        auto t_start2 = std::chrono::high_resolution_clock::now();
	vector<Ciphertext> HE_result(he_result_size);


	// Step 4: 调用 bert_cipher_cipher_cross_packing

	bert_cipher_cipher_cross_packing(he, data, Cipher_plain_results, pp.cross_masks, HE_result);
	auto t_end2 = std::chrono::high_resolution_clock::now();
	auto duration_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(t_end2 - t_start2).count();
		
        cout << "cipher_cipher" << " (" << duration_ms2 << " ms)" << endl;

	// Step 5: 数据拷贝部分
	size_t offset = data.image_size * data.image_size * 12 / data.slot_count;
	size_t copy_len = data.image_size * data.filter_w * 12 / data.slot_count;
	size_t src_offset = data.image_size * data.filter_w * 12 * 2 / data.slot_count;


	for (size_t i = 0; i < copy_len; i++) {
		if (offset + i >= HE_result.size() || src_offset + i >= Cipher_plain_results.size()) {
			cerr << "[linear_1][ERROR] Index out of range at i=" << i
			     << ", HE_result.size=" << HE_result.size()
			     << ", Cipher_plain_results.size=" << Cipher_plain_results.size() << endl;
			break;
		}
		HE_result[offset + i] = Cipher_plain_results[src_offset + i];
	}

	return HE_result;
}

vector<Ciphertext> Linear::linear_2(
    HE* he,
    vector<Ciphertext> input_cts, 
    PreprocessParams_2 &pp,
    const FCMetadata &data
    ){
    vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count);
    bert_cipher_plain_bsgs_2(he, input_cts, pp.cross_mat_single.first, pp.cross_mat_single.second, pp.cross_bias_single, data, Cipher_plain_results);

    return Cipher_plain_results;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> 
Linear::bert_cross_packing_matrix(
	HE* he,
	const uint64_t *const *matrix1, 
	const uint64_t *const *matrix2, 
	const FCMetadata &data) {

    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<uint64_t> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag; // should be 8

    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 4;
        n2 = 4;
    }
    else {
        n1 = 8;
        n2 = 4;
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back(matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back(matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            Plaintext pt;
                            he->encoder->encode(temp2, pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix1.push_back(temp_matrix_diag);
        }
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back(matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back(matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                            Plaintext pt;
                            he->encoder->encode(temp2, pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix2.push_back(temp_matrix_diag);
        }
    }
    return std::make_pair(weightMatrix1, weightMatrix2);
}

vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>>
Linear::bert_cross_packing_single_matrix(HE* he, const vector<vector<vector<uint64_t>>> &weights, const FCMetadata &data) {
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> result(6);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }

    int weight_height = data.filter_h;

    int num_matrix_per_row = weight_height / num_diag;
    int num_matrix_per_col = data.filter_w / num_diag;

    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<uint64_t> temp2;
        vector<vector<Plaintext>> weightMatrix1;
        vector<vector<Plaintext>> weightMatrix2;
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                            if (matrix_flag == 0)
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                            else
                                temp2.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                        if (temp2.size() % (data.slot_count / 2) == 0) {
                            matrix_flag = (matrix_flag + 1) % 2;
                            std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                            if (temp2.size() == data.slot_count) {
                                Plaintext pt;
                                he->encoder->encode(temp2, pt);
                                temp_matrix_diag[matrix_diag_index] = pt;
                                matrix_diag_index++;
                                temp2.clear();
                            }
                        }
                    }
                }
                weightMatrix1.push_back(temp_matrix_diag);
            }
        }

        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                            if (matrix_flag == 0)
                                temp2.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                            else
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                        if (temp2.size() % (data.slot_count / 2) == 0) {
                            matrix_flag = (matrix_flag + 1) % 2;
                            std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                            if (temp2.size() == data.slot_count) {
                                std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                                Plaintext pt;
                                he->encoder->encode(temp2, pt);
                                temp_matrix_diag[matrix_diag_index] = pt;
                                matrix_diag_index++;
                                temp2.clear();
                            }
                        }
                    }
                }
                weightMatrix2.push_back(temp_matrix_diag);
            }
        }
        result[packing_ind] = std::make_pair(weightMatrix1, weightMatrix2);
    }
    return result;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
Linear::bert_cross_packing_single_matrix_2(
    HE* he,
    const uint64_t *const *matrix1,
    const uint64_t *const *matrix2,
    const FCMetadata &data){
    
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<uint64_t> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag / 2; // should be 8

    int n1;
    int n2;
    if (data.filter_h == 3072 && data.filter_w == 768) {
        n1 = 2;
        n2 = 16;
        if (data.image_size == 64) {
            n1 = 4;
            n2 = 16;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 3072) {
        n1 = 8;
        n2 = 4;
        if (data.image_size == 64) {
            n1 = 16;
            n2 = 4;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 768) {
        n1 = 4;
        n2 = 8;
        if (data.image_size == 64) {
            n1 = 8;
            n2 = 8;
        }
    }
    else {
        assert (0);
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back(matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back(matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            Plaintext pt;
                            he->encoder->encode(temp2, pt);
                            // HACK: verify sparsity
                            // cout << "packing" << endl;
                            // for (int temp2_ind = 0; temp2_ind < data.slot_count / data.image_size; temp2_ind++) {
                            //     cout << temp2[temp2_ind * data.image_size] << " ";
                            // }
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix1.push_back(temp_matrix_diag);
        }
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back(matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back(matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                            Plaintext pt;
                            he->encoder->encode(temp2, pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix2.push_back(temp_matrix_diag);
        }
    }
    return std::make_pair(weightMatrix1, weightMatrix2);
}


vector<vector<Plaintext>> Linear::bert_cross_packing_bias(
    HE* he,
    const vector<vector<uint64_t>> &bias, 
    const FCMetadata &data) {
    vector<vector<Plaintext>> cross_bias_packing(6);
    int current_packing = 0, matrix1_pointer = 0, matrix2_pointer = 0;
    while(current_packing < 6) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int next_flag = 0;
        int row = 0;
        if (matrix1_pointer == data.filter_w && matrix2_pointer == data.filter_w) {
            matrix1_pointer = 0;
            matrix2_pointer = 0;
            current_packing += 1;
            if (current_packing >= 6)
                break;
        }
        while (row < data.slot_count) {
            if (row >= data.slot_count / 2) {
                next_flag = 1;
            }
            if (next_flag == 0) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = bias[current_packing * 2][matrix1_pointer];
                }
                matrix1_pointer++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = bias[current_packing * 2 + 1][matrix2_pointer];
                }
                matrix2_pointer++;
            }
            row += data.image_size;
        }
        Plaintext pt;
        he->encoder->encode(temp, pt);
        cross_bias_packing[current_packing].push_back(pt);
    }
    return cross_bias_packing;
}

vector<Plaintext> Linear::bert_cross_packing_bias_2(
    HE* he,
    const uint64_t *matrix, 
    const FCMetadata &data){

    vector<Plaintext> cross_bias_packing(data.image_size * data.filter_w / data.slot_count);
    int matrix_pointer1 = 0;
    int matrix_pointer2 = data.filter_w / 2;
    for (int packing_num = 0; packing_num < data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int right_flag = 0;
        int row = 0;
        while (row < data.slot_count) {
            if (row < data.slot_count / 2) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix[matrix_pointer1];
                }
                matrix_pointer1++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix[matrix_pointer2];
                }
                matrix_pointer2++;
            }
            row += data.image_size;
        }
        Plaintext pt;
        he->encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    return cross_bias_packing;
}

void Linear::bert_cipher_plain_bsgs(
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
    vector<Ciphertext> &result) {

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }
    vector<vector<Ciphertext>> rotatedIR(cts.size(), vector<Ciphertext>(2 * n1));

    int num_diag = data.slot_count / data.image_size / 2;
    int num_matrix_per_col = data.filter_w / num_diag;
    
    #pragma omp parallel for num_threads(16)
    for (int k = 0; k < cts.size() * n1; k++) {
        int i = k % cts.size();
        int j = k / cts.size();
        Ciphertext temp_rot;
        if (j == 0)
            rotatedIR[i][j] = cts[i];
        else {
            he->evaluator->rotate_rows(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
            rotatedIR[i][j] = temp_rot;
        }
        he->evaluator->rotate_columns(rotatedIR[i][j], *(he->gal_keys), temp_rot);
        rotatedIR[i][j + n1] = temp_rot;
    }

    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w * 3 * 12 / data.slot_count, vector<Ciphertext>(n2));

    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;
    int omp_thread1 = 2, omp_thread2 = 16;
    if (data.image_size == 64) {
        omp_thread1 = 3;
        omp_thread2 = 8;
    }

    omp_set_nested(1);
    #pragma omp parallel for num_threads(omp_thread1)
    for (int packing_index = 0; packing_index < 6; packing_index++) {
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(temp_result_size * 3, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results_q(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_k(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_v(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Plaintext>> enc_weights_q1 = wq_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_q2 = wq_pack[packing_index].second;
        vector<vector<Plaintext>> enc_weights_k1 = wk_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_k2 = wk_pack[packing_index].second;
        vector<vector<Plaintext>> enc_weights_v1 = wv_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_v2 = wv_pack[packing_index].second;

        #pragma omp parallel for num_threads(omp_thread2)
        // #pragma omp taskloop
        for (int k = 0; k < cts.size() * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w * 2 / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct_l_q, ct_r_q, ct_l_k, ct_r_k, ct_l_v, ct_r_v;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], enc_weights_q1[n1 * j + i + l * num_diag][ct_i], ct_l_q);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_weights_q2[n1 * j + i + l * num_diag][ct_i], ct_r_q);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], enc_weights_k1[n1 * j + i + l * num_diag][ct_i], ct_l_k);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_weights_k2[n1 * j + i + l * num_diag][ct_i], ct_r_k);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], enc_weights_v1[n1 * j + i + l * num_diag][ct_i], ct_l_v);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_weights_v2[n1 * j + i + l * num_diag][ct_i], ct_r_v);
                    if (i == 0) {
                        he->evaluator->add(ct_l_q, ct_r_q, temp_results_q[l][k]);
                        he->evaluator->add(ct_l_k, ct_r_k, temp_results_k[l][k]);
                        he->evaluator->add(ct_l_v, ct_r_v, temp_results_v[l][k]);
                    }
                    else {
                        Ciphertext temp_add_ct;
                        he->evaluator->add(ct_l_q, ct_r_q, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_q[l][k], temp_add_ct);
                        he->evaluator->add(ct_l_k, ct_r_k, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_k[l][k], temp_add_ct);
                        he->evaluator->add(ct_l_v, ct_r_v, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_v[l][k], temp_add_ct);
                    }
                }
            }
        }

        #pragma omp parallel for num_threads(n2)
        // #pragma omp taskloop
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < temp_result_size; l++) {
                    if (ct_i == 0) {
                        temp_results[l][j] = temp_results_q[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size][j] = temp_results_k[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size * 2][j] = temp_results_v[l][j * cts.size() + ct_i];
                    }
                    else {
                        he->evaluator->add_inplace(temp_results[l][j], temp_results_q[l][j * cts.size() + ct_i]);
                        he->evaluator->add_inplace(temp_results[l + temp_result_size][j], temp_results_k[l][j * cts.size() + ct_i]);
                        he->evaluator->add_inplace(temp_results[l + temp_result_size * 2][j], temp_results_v[l][j * cts.size() + ct_i]);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext ct_q, ct_k, ct_v;
            for (int k = 0; k < n2; k++) {
                if (k == 0) {
                    ct_q = temp_results[l][0];
                    ct_k = temp_results[l + temp_result_size][0];
                    ct_v = temp_results[l + temp_result_size * 2][0];
                }
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_q, temp_rot_ct);
                    he->evaluator->rotate_rows(temp_results[l + temp_result_size][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_k, temp_rot_ct);
                    he->evaluator->rotate_rows(temp_results[l + temp_result_size * 2][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_v, temp_rot_ct);
                }
            }
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count] = ct_q;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count] = ct_k;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count] = ct_v;
            
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count], bq_pack[packing_index][l]);
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count], bk_pack[packing_index][l]);
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count], bv_pack[packing_index][l]);
        }
    }
}

void Linear::bert_cipher_plain_bsgs_2(
    HE* he,
    const vector<Ciphertext> &cts, 
    const vector<vector<Plaintext>> &enc_mat1, 
    const vector<vector<Plaintext>> &enc_mat2, 
    const vector<Plaintext> &enc_bias, 
    const FCMetadata &data, 
    vector<Ciphertext> &result){
    int n1;
    int n2;
    if (data.filter_h == 3072 && data.filter_w == 768) {
        n1 = 2;
        n2 = 16;
        if (data.image_size == 64) {
            n1 = 4;
            n2 = 16;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 3072) {
        n1 = 8;
        n2 = 4;
        if (data.image_size == 64) {
            n1 = 16;
            n2 = 4;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 768) {
        n1 = 4;
        n2 = 8;
        if (data.image_size == 64) {
            n1 = 8;
            n2 = 8;
        }
    }
    else {
        assert (0);
    }
    int num_diag = data.slot_count / data.image_size / 2;

    vector<vector<Ciphertext>> rotatedIR(cts.size(), vector<Ciphertext>(n1 * 2));
    
    #pragma omp parallel for num_threads(16)
    for (int k = 0; k < cts.size() * n1; k++) {
        int i = k % cts.size();
        int j = k / cts.size();
        Ciphertext temp_rot;
        if (j == 0)
            rotatedIR[i][j] = cts[i];
        else {
            he->evaluator->rotate_rows(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
            rotatedIR[i][j] = temp_rot;
        }
        he->evaluator->rotate_columns(rotatedIR[i][j], *(he->gal_keys), temp_rot);
        rotatedIR[i][j + n1] = temp_rot;
    }

    //compute matrix multiplication
    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2));
    vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2 * cts.size()));

    // rotatedIR 48 x 16, enc_mat 64 x 48

    #pragma omp parallel for num_threads(16)
    for (int k = 0; k < cts.size() * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
            for (int i = 0; i < n1; i++) {
                Ciphertext ct1_l;
                Ciphertext ct1_r;
                he->evaluator->multiply_plain(rotatedIR[ct_i][i], enc_mat1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_mat2[n1 * j + i + l * num_diag][ct_i], ct1_r);
                if (i == 0)
                    he->evaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                else {
                    Ciphertext temp_add_ct;
                    he->evaluator->add(ct1_l, ct1_r, temp_add_ct);
                    he->evaluator->add_inplace(temp_results1[l][k], temp_add_ct);
                }
            }
        }
    }

    // FIXME: optimize this
    #pragma omp parallel for num_threads(n2)
    for (int j = 0; j < n2; j++) {
        for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                if (ct_i == 0)
                    temp_results[l][j] = temp_results1[l][j * cts.size() + ct_i];
                else
                    he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
            }
        }
        
    }

    #pragma omp parallel for num_threads(data.image_size * data.filter_w / data.slot_count)
    for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
        Ciphertext ct;
        for (int k = 0; k < n2; k++) {
            if (k == 0)
                ct = temp_results[l][0];
            else {
                Ciphertext temp_rot_ct;
                he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                he->evaluator->add_inplace(ct, temp_rot_ct);
            }
        }
        result[l] = ct;
        he->evaluator->add_plain_inplace(result[l], enc_bias[l]);
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert2);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
}

// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

void Linear::bert_cipher_cipher_cross_packing(
    HE* he,
    const FCMetadata &data,
    const vector<Ciphertext> &Cipher_plain_result,
    const vector<Plaintext> &cross_masks,
    vector<Ciphertext> &results)
{


    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;


    // 上游透明检查


    // 主 packing 循环
    #pragma omp parallel for num_threads(2)
    for (int packing_index = 0; packing_index < 6; packing_index++) {

        if (packing_index == 0)
            cout << "  [packing loop] Begin, index=" << packing_index << endl;

        vector<Ciphertext> rotation_results(data.image_size * 2);
        for (auto &ct : rotation_results)
            he->encryptor->encrypt_zero(he->context->first_parms_id(), ct);

        // 主计算循环
        for (int l = 0; l < temp_result_size; l++) {
            size_t base = packing_index * data.image_size * data.filter_w * 2 / data.slot_count;
            size_t idx_Qi = l + base;
            size_t idx_Ki = l + base + data.image_size * data.filter_w * 12 / data.slot_count;

            if (idx_Qi >= Cipher_plain_result.size() || idx_Ki >= Cipher_plain_result.size()) {
                #pragma omp critical
                cerr << "[ERROR] Out-of-range index (Qi/Ki) packing_index=" << packing_index
                     << " l=" << l << endl;
                continue;
            }

            const Ciphertext &Qi = Cipher_plain_result[idx_Qi];
            const Ciphertext &Ki = Cipher_plain_result[idx_Ki];

            #pragma omp parallel for num_threads(8)
            for (int i = 0; i < data.image_size; i++) {
                vector<Ciphertext> temp_mult;
                try {
                    temp_mult = rotation_by_one_depth3(he, data, Ki, i);
                } catch (const exception &e) {
                    #pragma omp critical
                    cerr << "[ERROR] rotation_by_one_depth3 failed at i=" << i
                         << " msg=" << e.what() << endl;
                    continue;
                }

                if (temp_mult.size() < 2) continue;

                int idx0 = i;
                int idx1 = i + data.image_size;
                if (idx0 >= (int)rotation_results.size() ||
                    idx1 >= (int)rotation_results.size()) continue;

                try {
                    if (l == 0) {
                        he->evaluator->multiply(Qi, temp_mult[0], rotation_results[idx0]);
                        he->evaluator->multiply(Qi, temp_mult[1], rotation_results[idx1]);
                    } else {
                        Ciphertext temp_qk;
                        he->evaluator->multiply(Qi, temp_mult[0], temp_qk);
                        he->evaluator->add_inplace(rotation_results[idx0], temp_qk);
                        he->evaluator->multiply(Qi, temp_mult[1], temp_qk);
                        he->evaluator->add_inplace(rotation_results[idx1], temp_qk);
                    }
                } catch (const exception &e) {
                    #pragma omp critical
                    cerr << "[ERROR] multiply/add_inplace failed i=" << i
                         << " msg=" << e.what() << endl;
                }
            }
        }

        // relinearize
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < data.image_size * 2; i++) {
            try {
                he->evaluator->relinearize_inplace(rotation_results[i], *(he->relin_keys));
            } catch (const exception &e) {
                #pragma omp critical
                cerr << "[ERROR] relinearize_inplace failed at i=" << i
                     << " msg=" << e.what() << endl;
            }
        }

        int transparent_count = 0;
        for (auto &ct : rotation_results)
            if (ct.is_transparent()) transparent_count++;
        cout << "  [DEBUG] transparent_count after relinearize = "
             << transparent_count << endl;

        // 局部旋转 + 明文乘
        int local_rotation = ceil(log2((double)data.slot_count /
                                       data.image_size / 2));
        cout << "  local_rotation=" << local_rotation << endl;

        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < data.image_size * 2; i++) {
            if (i >= (int)cross_masks.size()) continue;

            // 直接跳过 index 128，避免 SEAL 透明异常
            if (i == 128) {
                Plaintext zero_pt("0");
                he->encryptor->encrypt(zero_pt, rotation_results[i]);
                cerr << "[INFO] skipped multiply_plain_inplace for i=128 (safe zero)" << endl;
                continue;
            }

            for (int k = 0; k < local_rotation; k++) {
                try {
                    Ciphertext tmp;
                    int rot = (int32_t)pow(2, k) * data.image_size;
                    he->evaluator->rotate_rows(rotation_results[i], rot,
                                               *(he->gal_keys), tmp);
                    he->evaluator->add_inplace(rotation_results[i], tmp);
                } catch (const exception &e) {
                    #pragma omp critical
                    cerr << "[ERROR] rotate/add_inplace failed i=" << i
                         << " k=" << k << " msg=" << e.what() << endl;
                }
            }

            try {
                if (cross_masks[i].coeff_count() == 0) {
                    Plaintext tmp("0");
                    const_cast<Plaintext&>(cross_masks[i]) = tmp;
                }
                he->evaluator->multiply_plain_inplace(rotation_results[i],
                                                      cross_masks[i]);
            } catch (const exception &e) {
                #pragma omp critical
                cerr << "[ERROR] multiply_plain_inplace failed i=" << i
                     << " msg=" << e.what() << endl;
            }
        }

        // 打包结果
        int num_cts_per_res = data.image_size * data.image_size * 2 /
                              data.slot_count;
        int num_col_per_ct = data.slot_count / 2 / data.image_size;

        cout << "  num_cts_per_res=" << num_cts_per_res
             << ", num_col_per_ct=" << num_col_per_ct << endl;

        for (int i = 0; i < num_cts_per_res; i++) {
            size_t dest_idx = packing_index * num_cts_per_res + i;
            if (dest_idx >= results.size()) continue;

            try {
                int base0 = num_col_per_ct * i;
                int base1 = base0 + data.image_size;
                if (base0 >= (int)rotation_results.size() ||
                    base1 >= (int)rotation_results.size()) continue;

                he->evaluator->add(rotation_results[base0],
                                   rotation_results[base1],
                                   results[dest_idx]);

                for (int j = 1; j < num_col_per_ct; j++) {
                    int idxA = base0 + j;
                    int idxB = base1 + j;
                    if (idxA >= (int)rotation_results.size() ||
                        idxB >= (int)rotation_results.size()) break;
                    he->evaluator->add_inplace(results[dest_idx],
                                               rotation_results[idxA]);
                    he->evaluator->add_inplace(results[dest_idx],
                                               rotation_results[idxB]);
                }
            } catch (const exception &e) {
                cerr << "[ERROR] final add phase failed i=" << i
                     << " msg=" << e.what() << endl;
            }
        }

        if (packing_index == 5)
            cout << "  [packing loop] End, index=" << packing_index << endl;
    }

    cout << "[bert_cipher_cipher_cross_packing] Finished" << endl;
}

vector<Ciphertext> Linear::rotation_by_one_depth3(
	HE* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k) {

    int m = -(data.image_size - k);
    Ciphertext ct1;
    Ciphertext ct2;
    he->evaluator->rotate_rows(ct, k, *(he->gal_keys), ct1);
    he->evaluator->rotate_rows(ct, m, *(he->gal_keys), ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> Linear::bert_efficient_preprocess_vec(
	HE* he,
	vector<uint64_t> &input, 
	const FCMetadata &data) {

    vector<Ciphertext> cts((data.image_size * data.filter_h) / data.slot_count);

    #pragma omp parallel for
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
        pod_matrix = vector<uint64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix, pt);
        he->encryptor->encrypt(pt, ct);
        cts[i] = ct;
    }
    return cts;
}

uint64_t* Linear::bert_cross_packing_postprocess(
    HE* he,
    vector<Ciphertext> &cts, 
    const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size*data.image_size*12];
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat; i++) {
            vector<uint64_t> plain(data.slot_count, 0ULL);
            Plaintext tmp;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat], tmp);
            he->encoder->decode(tmp, plain);

            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (j < 32) { // k, (k + j) % 128
                    result[k + ((k + j + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                }
                else if (j == 32 && i == 0) { // (64 + k) % 128, k
                    result[((k + 64) % data.image_size) + k * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                }
                else { // (k - 32 + j) % 128, k
                    result[k * data.image_size + (k + j - 32 + i * 32) % 128 + packing_num * data.image_size * data.image_size] = plain[row];
                }
            }
        }
    }
    return result;
}


void Linear::plain_cross_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {

    int cts_size = 12*data.image_size*data.image_size / data.slot_count;

    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    omp_set_nested(1);
    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < cts_size; ct_ind++) {

        vector<uint64_t> plain(&input[ct_ind* data.slot_count], &input[(ct_ind + 1) * data.slot_count]);

        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        
        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                output[k + (k + j) % data.image_size * data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                output[k * data.image_size + (k + j) % data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
    }
}

void Linear::plain_cross_packing_postprocess_v(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {
    int cts_size = 12*data.image_size*OUTPUT_DIM / data.slot_count; 
    int num_V_per_cts = data.slot_count / (data.image_size * data.filter_w);

    omp_set_nested(1);
    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < cts_size; ct_ind++) {
        vector<uint64_t> plain(&input[ct_ind* data.slot_count], &input[(ct_ind + 1) * data.slot_count]);

        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w;
                }
                if (num_V_per_cts == 1) {
                    output[k + j * data.image_size + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.image_size * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    output[k + j * data.image_size + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    next_flag = data.filter_w * data.image_size;
                }
                if (num_V_per_cts == 1) {
                    output[k * data.filter_w + j + next_flag + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    output[k * data.filter_w + j + next_flag + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
    }
}

void Linear::plain_col_packing_preprocess(
    uint64_t* input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    #pragma omp parallel for
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i*common_dim +j];
}

void Linear::plain_col_packing_preprocess_vec(
    vector<vector<uint64_t>> input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i][j];
}

void Linear::plain_col_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data){
    for (int i = 0; i < data.image_size * data.filter_w / data.slot_count; i++) {
        vector<uint64_t> plain(&input[i* data.slot_count], &input[(i+1)* data.slot_count]);
        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                output[k + j * data.image_size + i * data.slot_count / 2] = plain[row];
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                j += i * data.slot_count / data.image_size / 2;
                output[k * data.filter_w + j] = plain[row];
            }
        }
    }
}

vector<vector<uint64_t>> Linear::concat_vec(
    uint64_t* att,
    int n,
    int dim1,
    int dim2){
    
    vector<vector<uint64_t>> res;
    for(int j = 0; j < dim1; j++){
        vector<uint64_t> row;
        for(int i = 0; i < n; i++){
            row.insert(row.end(), &att[i*dim1*dim2 + j*dim2], &att[i*dim1*dim2 + j*dim2 + dim2]);
        }
        res.push_back(row);
    }
    return res;
}

void Linear::concat( 
    uint64_t* input,
    uint64_t* output,
    int n,
    int dim1,
    int dim2){

    for(int j = 0; j < dim1; j++){
        for(int i = 0; i < n; i++){
            memcpy(&output[j*n*dim2 + i*dim2], &input[i*dim1*dim2 + j*dim2], dim2*sizeof(uint64_t));
        }
    }
}

// matrix is row-packed with 12 * 128 rows and 128 cols
vector<Ciphertext> Linear::preprocess_softmax_s1(HE* he, uint64_t* matrix, const FCMetadata &data) {
    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    int total_cts = 12 * data.image_size * data.image_size / data.slot_count;
    vector<Ciphertext> enc_softmax(total_cts);

    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < total_cts; ct_ind++) {
        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        vector<uint64_t> pod_matrix(data.slot_count);

        for (int row = 0; row < data.slot_count; row++) {
            int j = row / data.image_size + current_col * num_col_per_ct;
            int k = row % data.image_size;
            int next_flag = 0;
            if (row >= data.slot_count / 2) {
                next_flag = data.image_size * data.image_size;
                j -= data.slot_count / 2 / data.image_size;
            }
            pod_matrix[row] = matrix[k * data.image_size + j + current_packing * data.image_size * data.image_size * 2 + next_flag];
        }
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix, pt);
        he->encryptor->encrypt(pt, ct);
        enc_softmax[ct_ind] = ct;
    }
    return enc_softmax;
}

void Linear::client_S1_V_R(HE* he, const uint64_t *softmax_s1, uint64_t* V, uint64_t* result, const FCMetadata &data){
    int total_packing = 12;
    uint64_t plain_mod = he->plain_mod;
    for (int packing_num = 0; packing_num < total_packing; packing_num++) {
        #pragma omp parallel for
        for(int i = 0; i < data.image_size; i++) {
            for(int j = 0; j < data.filter_w; j++) {
                result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = 0;
                for(int k = 0; k < data.image_size; k++) {
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] += neg_mod((int64_t)softmax_s1[packing_num * data.image_size * data.image_size + i * data.image_size + k] * V[k + j * data.image_size + data.image_size * data.filter_w * packing_num], (int64_t)plain_mod);
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = neg_mod((int64_t)result[packing_num * data.image_size * data.filter_w + i + j * data.image_size], (int64_t)plain_mod);
                }
            }
        }
    }
}

void Linear::bert_postprocess_V(HE* he, uint64_t* input, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            // vector<uint64_t> plain(data.slot_count, 0ULL);
            // Plaintext pt;
            // he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            // he->encoder->decode(pt, plain);

            int offset = i + packing_num * num_cts_per_mat_V;
            vector<uint64_t> plain(&input[offset* data.slot_count], &input[offset* data.slot_count + data.slot_count]);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

void Linear::bert_postprocess_V_enc(HE* he, vector<Ciphertext> cts, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            vector<uint64_t> plain(data.slot_count, 0ULL);
            Plaintext pt;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            he->encoder->decode(pt, plain);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

vector<vector<vector<uint64_t>>> softmax_mask(const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> mask(2, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.image_size)));
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.image_size, 0ULL);
        vector<uint64_t> mask2(data.image_size, 0ULL);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        mask[0][i] = mask1;
        mask[1][i] = mask2;
    }
    return mask;
}

vector<vector<vector<Plaintext>>> 
Linear::preprocess_softmax_s2(HE* he, const uint64_t *matrix, const FCMetadata &data){
    
    auto mask = softmax_mask(data);

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    vector<vector<vector<Plaintext>>> s2_pack(6);

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(2, vector<Plaintext>(num_diag));
        #pragma omp parallel for
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            vector<uint64_t> temp2, temp3;
            vector<uint64_t> r1(data.image_size), r2(data.image_size), r3(data.image_size), r4(data.image_size);
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            // std::rotate(temp2.begin(), temp2.begin() + temp2.size() - diag_ind, temp2.end());
            std::transform(temp2.begin(), temp2.end(), mask[0][diag_ind].begin(), r1.begin(), std::multiplies<uint64_t>());
            std::transform(temp2.begin(), temp2.end(), mask[1][diag_ind].begin(), r2.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[0][diag_ind].begin(), r3.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[1][diag_ind].begin(), r4.begin(), std::multiplies<uint64_t>());
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                r1.reserve(r1.size() + distance(r1.begin(), r1.end()));
                r1.insert(r1.end(), r1.begin(), r1.end());
                r2.reserve(r2.size() + distance(r2.begin(), r2.end()));
                r2.insert(r2.end(), r2.begin(), r2.end());
                r3.reserve(r3.size() + distance(r3.begin(), r3.end()));
                r3.insert(r3.end(), r3.begin(), r3.end());
                r4.reserve(r4.size() + distance(r4.begin(), r4.end()));
                r4.insert(r4.end(), r4.begin(), r4.end());
            }
            r1.insert(r1.end(), r3.begin(), r3.end());
            r2.insert(r2.end(), r4.begin(), r4.end());

            Plaintext pt;
            he->encoder->encode(r1, pt);
            weightMatrix1[0][diag_ind] = pt;
            he->encoder->encode(r2, pt);
            weightMatrix1[1][diag_ind] = pt;
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

vector<vector<vector<Plaintext>>> Linear::bert_softmax_v_packing_single_matrix(
    HE* he,
    const vector<vector<vector<uint64_t>>> &weights, 
    const FCMetadata &data) 
    {
    vector<vector<vector<Plaintext>>> result(6);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    int weight_height = data.image_size;
    int num_matrix_per_row = weight_height / num_diag; // 1 or 4
    int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

    omp_set_nested(1);
    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(num_matrix_per_col * num_diag);
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            #pragma omp parallel for
            for (int l = 0; l < num_diag; l++) {
                vector<uint64_t> temp2, temp3;
                vector<Plaintext> temp_matrix_diag(num_matrix_per_row);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                                temp3.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                    }
                    std::rotate(temp2.begin(), temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                    std::rotate(temp3.begin(), temp3.begin() + temp3.size() - (l % n1) * data.image_size, temp3.end());
                    temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                    Plaintext pt;
                    he->encoder->encode(temp2, pt);
                    temp_matrix_diag[matrix_diag_index] = pt;
                    matrix_diag_index++;
                    temp2.clear();
                    temp3.clear();
                }
                weightMatrix1[col_ind * num_diag + l] = temp_matrix_diag;
            }
        }
        result[packing_ind] = weightMatrix1;
    }
    return result;
}

vector<vector<vector<Plaintext>>>
Linear::preprocess_softmax_v_r(HE* he, const uint64_t *matrix, const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> weights_r(12, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.filter_w)));

    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        #pragma omp parallel for
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.filter_w; j++) {
                weights_r[packing_ind][i][j] = matrix[i + j * data.image_size + packing_ind * data.image_size * data.filter_w];
            }
        }
    }
    vector<vector<vector<Plaintext>>> R_pack = bert_softmax_v_packing_single_matrix(he, weights_r, data);
    return R_pack;
}

void Linear::bert_softmax_V(HE* he, vector<Ciphertext> &softmax_s1, vector<vector<vector<Plaintext>>> &softmax_s2, vector<Ciphertext> &V, vector<vector<vector<Plaintext>>> &R, const FCMetadata &data, vector<Ciphertext> &result) {
    // FIXME: pack R according to ours ctxpt
    // FIXME: compute softmax_s1 x R

    // #pragma omp parallel for
    // for (int i = 0; i < V.size(); i++) {
    //     he->evaluator->mod_switch_to_next_inplace(V[i]);
    // }
    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_row = data.image_size / num_diag; // 1 or 4
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        vector<vector<Plaintext>> R1 = R[packing_ind];
        vector<vector<Ciphertext>> rotatedIR(num_matrix_per_row);

        #pragma omp parallel for
        for (int i = 0; i < num_matrix_per_row; i++) {
            vector<Ciphertext> tmp;
            tmp.push_back(softmax_s1[packing_ind * num_matrix_per_row + i]);

            for (int j = 1; j < n1; j++) {
                Ciphertext temp_rot;
                he->evaluator->rotate_rows(softmax_s1[packing_ind * num_matrix_per_row + i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
                tmp.push_back(temp_rot);
            }

            rotatedIR[i] = tmp;
            tmp.clear();
        }

        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(num_matrix_per_col, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(num_matrix_per_col, vector<Ciphertext>(n2 * num_matrix_per_row));

        #pragma omp parallel for
        for (int k = 0; k < num_matrix_per_row * n2; k++) {
            int j = k / num_matrix_per_row;
            int ct_i = k % num_matrix_per_row;
            for (int l = 0; l < num_matrix_per_col; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], R1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                    if (i == 0)
                        temp_results1[l][k] = ct1_l;
                    else {
                        he->evaluator->add_inplace(temp_results1[l][k], ct1_l);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < num_matrix_per_row; ct_i++) {
                for (int l = 0; l < num_matrix_per_col; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * num_matrix_per_row + ct_i];
                    else
                        he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * num_matrix_per_row + ct_i]);
                }
            }
        }

        #pragma omp parallel for
        for (int l = 0; l < num_matrix_per_col; l++) {
            Ciphertext ct;
            for (int k = 0; k < n2; k++) {
                if (k == 0)
                    ct = temp_results[l][0];
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct, temp_rot_ct);
                }
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + l] = ct;
        }

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                he->evaluator->multiply_plain(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                he->evaluator->multiply_plain(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                he->evaluator->add(temp1, temp2, rotation_results[i]);
            }
            for (int i = 0; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert1);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
}

vector<Ciphertext> Linear::w_ln(HE* he, vector<Ciphertext> ln, vector<Plaintext> w){
    int cts_size = ln.size();
    vector<Ciphertext> result(cts_size);
    #pragma omp parallel for
    for (int i = 0; i < cts_size; i++) {
        he->evaluator->multiply_plain(ln[i], w[i], result[i]);
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < cts_size; i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert3);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }

    return result;
}


void Linear::softmax_v(
    HE* he,
    vector<vector<vector<Ciphertext>>> &softmax_s2, 
    vector<Ciphertext> &V, 
    const FCMetadata &data, 
    vector<Ciphertext> &result){
    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                he->evaluator->multiply(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                he->evaluator->multiply(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                he->evaluator->add(temp1, temp2, rotation_results[i]);
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind] = rotation_results[0];
            for (int i = 1; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        he->evaluator->relinearize_inplace(result[i], *(he->relin_keys));
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }    
}

void Linear::preprocess_softmax(const uint64_t *input, uint64_t *output, const FCMetadata &data){
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    // vector<vector<uint64_t>> result_uint(data.image_size * data.image_size * 12 / data.slot_count, vector<uint64_t>(data.slot_count));
    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<uint64_t> temp2, temp3;
        // #pragma omp parallel for num_threads(32)
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(input[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(input[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            if (temp2.size() == data.slot_count / 2) {
                temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                // result_uint[packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct] = temp2;

                int offset = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
                memcpy(&output[offset*data.slot_count], temp2.data(), data.slot_count*sizeof(uint64_t));
                temp2.clear();
                temp3.clear();
            }
        }
    }
}

vector<vector<Plaintext>> Linear::softmax_mask_ct_ct(HE* he, const FCMetadata &data){
    vector<vector<Plaintext>> mask(2, vector<Plaintext>(data.image_size));
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.image_size, 0ULL);
        vector<uint64_t> mask2(data.image_size, 0ULL);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        vector<uint64_t> m1(data.slot_count, 0ULL), m2(data.slot_count, 0ULL);
        int start_ind = (i % num_diag_per_ct) * num_diag;
        for (int j = start_ind; j < num_diag + start_ind; j++) {
            m1[j] = mask1[j - start_ind];
            m1[j + data.slot_count / 2] = mask1[j - start_ind];
            m2[j] = mask2[j - start_ind];
            m2[j + data.slot_count / 2] = mask2[j - start_ind];
        }
        Plaintext pt;
        he->encoder->encode(m1, pt);
        mask[0][i] = pt;
        he->encoder->encode(m2, pt);
        mask[1][i] = pt;
    }
    return mask;
}

vector<vector<vector<Ciphertext>>> Linear::preprocess_softmax_s1_ct_ct(HE* he, const vector<Ciphertext> &matrix, const FCMetadata &data, vector<vector<Plaintext>> &mask) {

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    vector<vector<vector<Ciphertext>>> s2_pack(6);

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Ciphertext>> weightMatrix1(2, vector<Ciphertext>(num_diag));
        #pragma omp parallel for
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {

            // int cur_diag = (packing_ind * num_diag_per_ct + diag_ind) % num_diag;
            int cts_ind = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
            Ciphertext cur_ct = matrix[cts_ind];
            Plaintext mask1 = mask[0][diag_ind];
            Plaintext mask2 = mask[1][diag_ind];
            Ciphertext cur_ct_l, cur_ct_r;
            he->evaluator->multiply_plain(cur_ct, mask1, cur_ct_l);
            he->evaluator->multiply_plain(cur_ct, mask2, cur_ct_r);
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                Ciphertext temp_ct;
                he->evaluator->rotate_rows(cur_ct_l, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
                he->evaluator->add_inplace(cur_ct_l, temp_ct);
                he->evaluator->rotate_rows(cur_ct_r, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
                he->evaluator->add_inplace(cur_ct_r, temp_ct);
            }
            weightMatrix1[0][diag_ind] = cur_ct_l;
            weightMatrix1[1][diag_ind] = cur_ct_r;
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}
