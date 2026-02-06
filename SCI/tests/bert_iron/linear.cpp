#include "linear.h"


void print_pt_l(HE* he, Plaintext &pt, int len) {
    // vector<int64_t> dest(len, 0ULL);
    // he->encoder->decode(pt, dest);
    // cout << "Decode first 5 rows: ";
    // int non_zero_count;
    // for(int i = 0; i < 10; i++){
    //     cout << dest[i] << " ";
    //     // if(dest[i] != 0){
    //     //     non_zero_count += 1;
    //     // }
    // }
    // // cout << "Non zero count: " << non_zero_count;
    // cout << endl;
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

Linear::Linear(int party, NetIO *io) {
	this->party = party;
	this->io = io;
	this->he_8192 = new HE(
		party,
		io,
		8192,
		// {40, 39, 30},
		{60, 60, 60},
		(uint64_t) pow(2, 37)
    );

    cout << "Set up he done" << endl;

    this->p_mod = prime_mod;

	// this->he_8192 = new HE(
	// 	party,
	// 	io,
	// 	8192,
	// 	{54, 55},
	// 	65537
    // );

    pp_1.resize(ATTENTION_LAYERS);
    pp_2.resize(ATTENTION_LAYERS);
    pp_3.resize(ATTENTION_LAYERS);
    pp_4.resize(ATTENTION_LAYERS);

    data_lin1.filter_h = COMMON_DIM;
    data_lin1.filter_w = OUTPUT_DIM;
    data_lin1.image_size = INPUT_DIM;
    data_lin1.slot_count = 8192;
    // data_lin1.nw = 4;
    // data_lin1.kw = 8;
    data_lin1.nw = 8;
    data_lin1.kw = 8;

    data_lin2.filter_h = COMMON_DIM;
    data_lin2.filter_w = COMMON_DIM;
    data_lin2.image_size = INPUT_DIM;
    data_lin2.slot_count = 8192;
    // data_lin2.nw = 8;
    // data_lin2.kw = 4;
    data_lin2.nw = 8;
    data_lin2.kw = 8;

    data_lin3.filter_h = COMMON_DIM;
    data_lin3.filter_w = INTER_DIM;
    data_lin3.image_size = INPUT_DIM;
    data_lin3.slot_count = 8192;
    // data_lin3.nw = 4;
    // data_lin3.kw = 8;
    data_lin3.nw = 4;
    data_lin3.kw = 16;

    data_lin4.filter_h = INTER_DIM;
    data_lin4.filter_w = COMMON_DIM;
    data_lin4.image_size = INPUT_DIM;
    data_lin4.slot_count = 8192;
    // data_lin4.nw = 8;
    // data_lin4.kw = 4;
    data_lin4.nw = 16;
    data_lin4.kw = 4;
}

Linear::~Linear() {

}

PreprocessParams_1 Linear::params_preprocessing_ct_pt_1(
    HE* he,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    const FCMetadata &data
){
    int input_dim = 128;
    int common_dim = 768;
    int output_dim = 64;

    uint64_t plain_mod = he->plain_mod;

    PreprocessParams_1 pp;
     for (int packing_index = 0; packing_index < 12; packing_index++) {
        vector<uint64_t *> matrix_mod_p1(common_dim);
        vector<uint64_t *> matrix_mod_p2(common_dim);
        vector<uint64_t *> matrix_mod_p3(common_dim);

        vector<uint64_t> bias_mod_p1(output_dim);
        vector<uint64_t> bias_mod_p2(output_dim);
        vector<uint64_t> bias_mod_p3(output_dim);
        for (int i = 0; i < common_dim; i++) {
            matrix_mod_p1[i] = new uint64_t[output_dim];
            matrix_mod_p2[i] = new uint64_t[output_dim];
            matrix_mod_p3[i] = new uint64_t[output_dim];

            for (int j = 0; j < output_dim; j++) {
                matrix_mod_p1[i][j] = neg_mod((int64_t)w_q[packing_index][i][j], (int64_t)plain_mod);
                matrix_mod_p2[i][j] = neg_mod((int64_t)w_k[packing_index][i][j], (int64_t)plain_mod);
                matrix_mod_p3[i][j] = neg_mod((int64_t)w_v[packing_index][i][j], (int64_t)plain_mod);
            }
        }

        for (int i = 0; i < output_dim; i++) {
            bias_mod_p1[i] = neg_mod((int64_t)b_q[packing_index][i], (int64_t)plain_mod);
            bias_mod_p2[i] = neg_mod((int64_t)b_k[packing_index][i], (int64_t)plain_mod);
            bias_mod_p3[i] = neg_mod((int64_t)b_v[packing_index][i], (int64_t)plain_mod);
        }

        auto encoded_mat1 = preprocess_matrix(he, matrix_mod_p1.data(), data);
        auto encoded_mat2 = preprocess_matrix(he, matrix_mod_p2.data(), data);
        auto encoded_mat3 = preprocess_matrix(he, matrix_mod_p3.data(), data);

        auto temp_bias1 = preprocess_bias(he, bias_mod_p1.data(), data);
        auto temp_bias2 = preprocess_bias(he, bias_mod_p2.data(), data);
        auto temp_bias3 = preprocess_bias(he, bias_mod_p3.data(), data);
        pp.encoded_mats1.push_back(encoded_mat1);
        pp.encoded_mats2.push_back(encoded_mat2);
        pp.encoded_mats3.push_back(encoded_mat3);
        pp.encoded_bias1.push_back(temp_bias1);
        pp.encoded_bias2.push_back(temp_bias2);
        pp.encoded_bias3.push_back(temp_bias3);
    }

    return pp;
}

PreprocessParams_2 Linear::params_preprocessing_ct_pt_2(
    HE* he,
    int32_t input_dim, 
    int32_t common_dim, 
    int32_t output_dim,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    const FCMetadata &data
){
    uint64_t plain_mod = he->plain_mod;
    PreprocessParams_2 pp;

    vector<uint64_t *> matrix_mod_p1(common_dim);
    vector<uint64_t *> matrix1(common_dim);
    for (int i = 0; i < common_dim; i++) {
        matrix_mod_p1[i] = new uint64_t[output_dim];
        matrix1[i] = new uint64_t[output_dim];
        for (int j = 0; j < output_dim; j++) {
            matrix_mod_p1[i][j] = neg_mod((int64_t)w[i][j], (int64_t)plain_mod);
        }
    }
    for (int i = 0; i < output_dim; i++) {
        b[i] = neg_mod((int64_t)b[i], (int64_t)plain_mod);
    }

    pp.encoded_mat = preprocess_matrix(he, matrix_mod_p1.data(), data);
    pp.encoded_bias = preprocess_bias(he, b.data(), data);
    return pp;
}

vector<vector<Plaintext>> Linear::preprocess_matrix(HE* he, const uint64_t *const *matrix, const FCMetadata &data){
    vector<vector<Plaintext>> weightMatrix;
    int sub_mat_row = data.filter_h / data.nw; // 48
    int sub_mat_col = data.filter_w / data.kw; // 32
    for (int sub_row_ind = 0; sub_row_ind < sub_mat_row; sub_row_ind++) {
        vector<Plaintext> temp;
        for (int sub_col_ind = 0; sub_col_ind < sub_mat_col; sub_col_ind++) {
            vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
            for (int i = 0; i < data.nw; i++) {
                for (int j = 0; j < data.kw; j++) {
                    pod_matrix[j * data.nw + i] = matrix[i + sub_row_ind * data.nw][j + sub_col_ind * data.kw];
                }
            }
            Plaintext pt = encode_vector(he, pod_matrix.data(), data);
            temp.push_back(pt);
        }
        weightMatrix.push_back(temp);
        temp.clear();
    }
    return weightMatrix;
}

vector<Plaintext> Linear::preprocess_bias(HE* he, const uint64_t *matrix, const FCMetadata &data){
    int res_cts_num = data.filter_w / data.kw;
    vector<Plaintext> packed_bias(res_cts_num);
    for (int ct_ind = 0; ct_ind < res_cts_num; ct_ind++) {
        vector<uint64_t> pt_data(data.slot_count, 0ULL);
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.kw; j++) {
                pt_data[i * data.nw * data.kw + (j + 1) * data.nw - 1] = matrix[(j + ct_ind * data.kw)];
            }
        }
        packed_bias[ct_ind] = encode_vector(he, pt_data.data(), data);
    }
    return packed_bias;
}

Plaintext Linear::encode_vector(HE* he, const uint64_t *vec, const FCMetadata &data) {
    Plaintext pt;
    pt.resize(data.slot_count);
    assert(pt.data() != nullptr);
    seal::util::modulo_poly_coeffs(vec, data.slot_count, he->plain_mod, pt.data());
    return pt;
}


vector<Ciphertext> Linear::linear_1(
HE* he,
vector<Ciphertext> input_cts, 
PreprocessParams_1 &pp,
const FCMetadata &data
){
    vector<Ciphertext> result(data.filter_w / data.kw * 3 * 12);

    for (int packing_index = 0; packing_index < 12; packing_index++) {
        vector<vector<Plaintext>> enc_mat1 = pp.encoded_mats1[packing_index];
        vector<vector<Plaintext>> enc_mat2 = pp.encoded_mats2[packing_index];
        vector<vector<Plaintext>> enc_mat3 = pp.encoded_mats3[packing_index];

        #pragma omp parallel for num_threads(32)
        for (int j = 0; j < data.filter_w / data.kw; j++) {
            for (int i = 0; i < data.filter_h / data.nw; i++) {
                Ciphertext temp_ct1;
                Ciphertext temp_ct2;
                Ciphertext temp_ct3;
                he->evaluator->multiply_plain(input_cts[i], enc_mat1[i][j], temp_ct1);
                he->evaluator->multiply_plain(input_cts[i], enc_mat2[i][j], temp_ct2);
                he->evaluator->multiply_plain(input_cts[i], enc_mat3[i][j], temp_ct3);
                if (i == 0) {
                    result[j + data.filter_w / data.kw * 3 * packing_index] = temp_ct1;
                    result[j + data.filter_w / data.kw + data.filter_w / data.kw * 3 * packing_index] = temp_ct2;
                    result[j + data.filter_w / data.kw * 2 + data.filter_w / data.kw * 3 * packing_index] = temp_ct3;
                }
                else {
                    he->evaluator->add_inplace(result[j + data.filter_w / data.kw * 3 * packing_index], temp_ct1);
                    he->evaluator->add_inplace(result[j + data.filter_w / data.kw + data.filter_w / data.kw * 3 * packing_index], temp_ct2);
                    he->evaluator->add_inplace(result[j + data.filter_w / data.kw * 2 + data.filter_w / data.kw * 3 * packing_index], temp_ct3);
                }
            }

            he->evaluator->add_plain_inplace(result[j + data.filter_w / data.kw * 3 * packing_index], pp.encoded_bias1[packing_index][j]);
            he->evaluator->add_plain_inplace(result[j + data.filter_w / data.kw + data.filter_w / data.kw * 3 * packing_index], pp.encoded_bias2[packing_index][j]);
            he->evaluator->add_plain_inplace(result[j + data.filter_w / data.kw * 2 + data.filter_w / data.kw * 3 * packing_index], pp.encoded_bias3[packing_index][j]);
        }
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);
    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, 100-37);
    }

    int L = result[0].coeff_modulus_size();
    vector<int> used_indices;
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; j < data.kw; j++) {
            used_indices.push_back(i * data.nw * data.kw + (j + 1) * data.nw - 1);
        }
    }
    std::sort(used_indices.begin(), used_indices.end());

    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < data.slot_count; j++) {
            if (std::binary_search(used_indices.cbegin(), used_indices.cend(), j))
                continue;
            auto rns_ptr = result[i].data(0);
            for (int k = 0; k < L; k++) {
                rns_ptr[j] = 0;
                rns_ptr += data.slot_count;
            }
        }
    }
    return result;
}

vector<Ciphertext> Linear::linear_2(
HE* he,
vector<Ciphertext> input_cts, 
PreprocessParams_2 &pp,
const FCMetadata &data
){

    vector<Ciphertext> result(data.filter_w / data.kw);

    #pragma omp parallel for num_threads(32)
    for (int j = 0; j < data.filter_w / data.kw; j++) {
        for (int i = 0; i < data.filter_h / data.nw; i++) {
            Ciphertext temp_ct1;
            he->evaluator->multiply_plain(input_cts[i], pp.encoded_mat[i][j], temp_ct1);
            if (i == 0) {
                result[j] = temp_ct1;
            }
            else {
                he->evaluator->add_inplace(result[j], temp_ct1);
            }
        }
    }

    for (int i = 0; i < result.size(); i++) {
        he->evaluator->add_plain_inplace(result[i], pp.encoded_bias[i]);
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);
    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, 100-37);
    }

    int L = result[0].coeff_modulus_size();
    vector<int> used_indices;
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; j < data.kw; j++) {
            used_indices.push_back(i * data.nw * data.kw + (j + 1) * data.nw - 1);
        }
    }
    std::sort(used_indices.begin(), used_indices.end());

    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < data.slot_count; j++) {
            if (std::binary_search(used_indices.cbegin(), used_indices.cend(), j))
                continue;
            auto rns_ptr = result[i].data(0);
            for (int k = 0; k < L; k++) {
                rns_ptr[j] = 0;
                rns_ptr += data.slot_count;
            }
        }
    }
    return result;
    
}

vector<Ciphertext> Linear::preprocess_vec(HE* he, vector<uint64_t> &input, const FCMetadata &data){
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.filter_h / data.nw); i++) {
        vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
        for (int row_index = 0; row_index < data.image_size; row_index++) {
            for (int col_index = 0; col_index < data.nw; col_index++) {
                pod_matrix[row_index * data.nw * data.kw + (data.nw - 1) - col_index] = input[row_index + (i * data.nw + col_index) * data.image_size];
            }
        }
        Plaintext pt = encode_vector(he, pod_matrix.data(), data);
        Ciphertext ct;
        // encryptor->encrypt(pt, ct);
        he->encryptor->encrypt_symmetric(pt, ct);

        cts.push_back(ct);
    }
    return cts;
}

vector<Ciphertext> Linear::preprocess_ptr(HE* he, uint64_t* input, const FCMetadata &data){
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.filter_h / data.nw); i++) {
        vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
        for (int row_index = 0; row_index < data.image_size; row_index++) {
            for (int col_index = 0; col_index < data.nw; col_index++) {
                pod_matrix[row_index * data.nw * data.kw + (data.nw - 1) - col_index] = input[row_index + (i * data.nw + col_index) * data.image_size];
            }
        }
        Plaintext pt = encode_vector(he, pod_matrix.data(), data);
        Ciphertext ct;
        // encryptor->encrypt(pt, ct);
        he->encryptor->encrypt_symmetric(pt, ct);

        cts.push_back(ct);
    }
    return cts;
}

vector<Plaintext> Linear::preprocess_ptr_plaintext(HE* he, uint64_t* input, const FCMetadata &data){
    vector<Plaintext> pts;
    for (int i = 0; i < (data.filter_h / data.nw); i++) {
        vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
        for (int row_index = 0; row_index < data.image_size; row_index++) {
            for (int col_index = 0; col_index < data.nw; col_index++) {
                pod_matrix[row_index * data.nw * data.kw + (data.nw - 1) - col_index] = input[row_index + (i * data.nw + col_index) * data.image_size];
            }
        }
        Plaintext pt = encode_vector(he, pod_matrix.data(), data);

        pts.push_back(pt);
    }
    return pts;
}

void Linear::weights_preprocess(BertModel &bm){
    #pragma omp parallel for
    for(int i = 0; i < ATTENTION_LAYERS; i++){
        pp_1[i] = params_preprocessing_ct_pt_1(
            he_8192,
            bm.w_q[i],
            bm.w_k[i],
            bm.w_v[i],
            bm.b_q[i],
            bm.b_k[i],
            bm.b_v[i],
            data_lin1
        );

        pp_2[i] = params_preprocessing_ct_pt_2(
            he_8192,
            INPUT_DIM,
            COMMON_DIM,
            COMMON_DIM,
            bm.w_o[i],
            bm.b_o[i],
            data_lin2
        );

        pp_3[i] = params_preprocessing_ct_pt_2(
            he_8192,
            INPUT_DIM,
            COMMON_DIM,
            INTER_DIM,
            bm.w_i_1[i],
            bm.b_i_1[i],
            data_lin3
        );

        pp_4[i] = params_preprocessing_ct_pt_2(
            he_8192,
            INPUT_DIM,
            INTER_DIM,
            COMMON_DIM,
            bm.w_i_2[i],
            bm.b_i_2[i],
            data_lin4
        );
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


vector<Plaintext> Linear::preprocess_noise_1(HE* he, const uint64_t *secret_share, const FCMetadata &data){
    // Sample randomness into vector
    vector<Plaintext> enc_noise(data.filter_w / data.kw * 3 * 12);
    for (int ct_index = 0; ct_index < enc_noise.size(); ct_index++) {
        vector<uint64_t> noise(data.slot_count, 0ULL);
        for (int i = 0; i < data.slot_count; i++)
            noise[i] = secret_share[i + ct_index * data.slot_count];

        Plaintext pt = encode_vector(he, noise.data(), data);
        // Ciphertext ct;
        // encryptor->encrypt(pt, ct);
        enc_noise[ct_index] = pt;
    }
    return enc_noise;
}

vector<Plaintext> Linear::preprocess_noise_2(HE* he, const uint64_t *secret_share, const FCMetadata &data){
    // Sample randomness into vector
    vector<Plaintext> enc_noise(data.filter_w / data.kw);
    for (int ct_index = 0; ct_index < enc_noise.size(); ct_index++) {
        vector<uint64_t> noise(data.slot_count, 0ULL);
        for (int i = 0; i < data.slot_count; i++)
            noise[i] = secret_share[i + ct_index * data.slot_count];

        Plaintext pt = encode_vector(he, noise.data(), data);
        // Ciphertext ct;
        // encryptor->encrypt(pt, ct);
        enc_noise[ct_index] = pt;
    }
    return enc_noise;
}

vector<vector<vector<uint64_t>>> Linear::bert_postprocess_noise(HE* he, vector<Plaintext> &enc_noise, const FCMetadata &data){
    // uint64_t *result = new uint64_t[data.image_size * data.filter_w * 3 * 12];
    vector<vector<vector<uint64_t>>> result(12, vector<vector<uint64_t>>(3, vector<uint64_t>(data.image_size * data.filter_w)));
    uint64_t plain_mod = he->plain_mod;
    for (int packing_index = 0; packing_index < 12; packing_index++) {
        for (int ct_ind = 0; ct_ind < enc_noise.size() / 3 / 12; ct_ind++) {
            Plaintext pt = enc_noise[ct_ind + packing_index * enc_noise.size() / 12];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    result[packing_index][0][i + (j + ct_ind * data.kw) * data.image_size] = plain_mod - pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }

        for (int ct_ind = 0; ct_ind < enc_noise.size() / 3 / 12; ct_ind++) {
            Plaintext pt = enc_noise[ct_ind + enc_noise.size() / 3 / 12 + packing_index * enc_noise.size() / 12];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    result[packing_index][1][i + (j + ct_ind * data.kw) * data.image_size] = plain_mod - pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }

        for (int ct_ind = 0; ct_ind < enc_noise.size() / 3 / 12; ct_ind++) {
            Plaintext pt = enc_noise[ct_ind + enc_noise.size() / 3 / 12 * 2 + packing_index * enc_noise.size() / 12];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    result[packing_index][2][i + (j + ct_ind * data.kw) * data.image_size] = plain_mod - pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }
    }

    return result;
}

vector<vector<vector<uint64_t>>> 
Linear::pt_postprocess_1(
    HE* he,
    vector<Plaintext> &pts, 
    const FCMetadata &data, 
	const bool &col_packing){
    vector<vector<vector<uint64_t>>> result(12, vector<vector<uint64_t>>(3, vector<uint64_t>(data.image_size * data.filter_w)));

    for (int packing_index = 0; packing_index < 12; packing_index++) {
        for (int ct_ind = 0; ct_ind < pts.size() / 3 / 12; ct_ind++) {
            Plaintext pt = pts[ct_ind + packing_index * pts.size() / 12];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    if (col_packing)
                        result[packing_index][0][i + (j + ct_ind * data.kw) * data.image_size] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                    else
                        result[packing_index][0][i * data.filter_w + (j + ct_ind * data.kw)] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }
        for (int ct_ind = 0; ct_ind < pts.size() / 3 / 12; ct_ind++) {
            Plaintext pt = pts[ct_ind + pts.size() / 3 / 12 + packing_index * pts.size() / 12];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    if (col_packing)
                        result[packing_index][1][i + (j + ct_ind * data.kw) * data.image_size] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                    else
                        result[packing_index][1][i * data.filter_w + (j + ct_ind * data.kw)] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }

        for (int ct_ind = 0; ct_ind < pts.size() / 3 / 12; ct_ind++) {
            Plaintext pt = pts[ct_ind + pts.size() / 3 / 12 * 2 + packing_index * pts.size() / 12];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    if (col_packing)
                        result[packing_index][2][i + (j + ct_ind * data.kw) * data.image_size] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                    else
                        result[packing_index][2][i * data.filter_w + (j + ct_ind * data.kw)] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }
    }

    return result;
}

void Linear::pt_postprocess_2(
		HE* he,
        vector<Plaintext> &pts, 
		uint64_t* output,
        const FCMetadata &data, 
		const bool &col_packing){
    
    if (col_packing) {
        for (int ct_ind = 0; ct_ind < pts.size(); ct_ind++) {
            Plaintext pt = pts[ct_ind];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    output[i + (j + ct_ind * data.kw) * data.image_size] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }
    }
    else {
        for (int ct_ind = 0; ct_ind < pts.size(); ct_ind++) {
            Plaintext pt = pts[ct_ind];
            for (int i = 0; i < data.image_size; i++) {
                for (int j = 0; j < data.kw; j++) {
                    output[i * data.filter_w + (j + ct_ind * data.kw)] = pt[i * data.nw * data.kw + (j + 1) * data.nw - 1];
                }
            }
        }
    }
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

void Linear::plain_col_packing_preprocess(
    uint64_t* input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i*common_dim +j];
}
