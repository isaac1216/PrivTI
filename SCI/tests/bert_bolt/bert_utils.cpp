#include "bert_utils.h"

vector<vector<uint64_t>> read_data(const string& filename) {
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    while (getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            row.push_back(stoll(cell));
        }

        data.push_back(row);
    }

    input_file.close();
    return data;
}

vector<uint64_t> read_bias(const string& filename, int output_dim) {
    ifstream input_file(filename);
    vector<uint64_t> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            sub_mat.push_back(stoll(cell));
        }
        row_counting++;
        if (row_counting == output_dim) {
            data = sub_mat;
            sub_mat.clear();
            row_counting -= output_dim;
        }
    }

    input_file.close();
    return data;
}

vector<vector<vector<uint64_t>>> read_qkv_weights(const string& filename) {
    ifstream input_file(filename);
    vector<vector<vector<uint64_t>>> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    vector<vector<uint64_t>> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            row.push_back(stoll(cell));
        }
        sub_mat.push_back(row);
        row_counting++;
        if (row_counting == 768) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= 768;
        }
    }

    input_file.close();
    return data;
}

vector<vector<uint64_t>> read_qkv_bias(const string& filename) {
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            sub_mat.push_back(stoll(cell));
        }
        row_counting++;
        if (row_counting == 64) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= 64;
        }
    }

    input_file.close();
    return data;
}

string replace(string str, string substr1, string substr2) {
    size_t index = str.find(substr1, 0); 
    str.replace(index, substr1.length(), substr2);
    return str;
}

BertModel load_model(string model_dir, int num_class){
    BertModel bm;

    bm.w_q.resize(12);
    bm.w_k.resize(12);
    bm.w_v.resize(12);
    bm.w_o.resize(12);
    bm.w_i_1.resize(12);
    bm.w_i_2.resize(12);

    bm.b_q.resize(12);
    bm.b_k.resize(12);
    bm.b_v.resize(12);
    bm.b_o.resize(12);
    bm.b_i_1.resize(12);
    bm.b_i_2.resize(12);

    bm.w_ln_1.resize(12);
    bm.w_ln_2.resize(12);

    bm.b_ln_1.resize(12);
    bm.b_ln_2.resize(12);

    
    // Attention
    string wq_fname = 
    "bert.encoder.layer.X.attention.self.query.weight.txt";
    string bq_fname = 
    "bert.encoder.layer.X.attention.self.query.bias.txt";

    string wk_fname = 
    "bert.encoder.layer.X.attention.self.key.weight.txt";
    string bk_fname = 
    "bert.encoder.layer.X.attention.self.key.bias.txt";

    string wv_fname = 
    "bert.encoder.layer.X.attention.self.value.weight.txt";
    string bv_fname = 
    "bert.encoder.layer.X.attention.self.value.bias.txt";

    string wo_fname = 
    "bert.encoder.layer.X.attention.output.dense.weight.txt";
    string bo_fname = 
    "bert.encoder.layer.X.attention.output.dense.bias.txt";

    string wi1_fname = 
    "bert.encoder.layer.X.intermediate.dense.weight.txt";
    string bi1_fname = 
    "bert.encoder.layer.X.intermediate.dense.bias.txt";

    string wi2_fname = 
    "bert.encoder.layer.X.output.dense.weight.txt";
    string bi2_fname = 
    "bert.encoder.layer.X.output.dense.bias.txt";

    // Layer Norm
    string wln1_fname = 
    "bert.encoder.layer.X.attention.output.LayerNorm.weight.txt";
    string bln1_fname = 
    "bert.encoder.layer.X.attention.output.LayerNorm.bias.txt";

    string wln2_fname = 
    "bert.encoder.layer.X.output.LayerNorm.weight.txt";
    string bln2_fname = 
    "bert.encoder.layer.X.output.LayerNorm.bias.txt";

    // Pooling
    string wp_fname = 
    "bert.pooler.dense.weight.txt";
    string bp_fname = 
    "bert.pooler.dense.bias.txt";

    // Classification
    string wc_fname = 
    "classifier.weight.txt";
    string bc_fname = 
    "classifier.bias.txt";
    
    // Attention
    #pragma omp parallel for
    for(int i = 0; i < 12; i++){
        string lid = to_string(i);
        vector<vector<vector<uint64_t>>> wq = read_qkv_weights(
            model_dir + replace(wq_fname, "X", lid)
        );
        vector<vector<vector<uint64_t>>> wk = read_qkv_weights(
            model_dir + replace(wk_fname, "X", lid)
        );
        vector<vector<vector<uint64_t>>> wv = read_qkv_weights(
            model_dir + replace(wv_fname, "X", lid)
        );
        
        vector<vector<uint64_t>> wo = read_data(
            model_dir + replace(wo_fname, "X", lid)
        );
        vector<vector<uint64_t>> wi1 = read_data(
            model_dir + replace(wi1_fname, "X", lid)
        );
        vector<vector<uint64_t>> wi2 = read_data(
            model_dir + replace(wi2_fname, "X", lid)
        );

        vector<vector<uint64_t>> bq = read_qkv_bias(
            model_dir + replace(bq_fname, "X", lid)
        );
        vector<vector<uint64_t>> bk = read_qkv_bias(
            model_dir + replace(bk_fname, "X", lid)
        );
        vector<vector<uint64_t>> bv = read_qkv_bias(
            model_dir + replace(bv_fname, "X", lid)
        );

        vector<uint64_t> wln1 = read_bias(
            model_dir + replace(wln1_fname, "X", lid), 
            768
        );
        vector<uint64_t> bln1 = read_bias(
            model_dir + replace(bln1_fname, "X", lid), 
            768
        );
        vector<uint64_t> wln2 = read_bias(
            model_dir + replace(wln2_fname, "X", lid), 
            768
        );
        vector<uint64_t> bln2 = read_bias(
            model_dir + replace(bln2_fname, "X", lid), 
            768
        );

        vector<uint64_t> bo = read_bias(
            model_dir + replace(bo_fname, "X", lid), 
            768
        );
        vector<uint64_t> bi1 = read_bias(
            model_dir + replace(bi1_fname, "X", lid), 
            3072
        );
        vector<uint64_t> bi2 = read_bias(
            model_dir + replace(bi2_fname, "X", lid), 
            768
        );

        bm.w_q[i] =  wq;
        bm.w_k[i] =  wk;
        bm.w_v[i] =  wv;
        bm.w_o[i] =  wo;
        bm.w_i_1[i] =  wi1;
        bm.w_i_2[i] =  wi2;

        bm.w_ln_1[i] =  wln1;
        bm.b_ln_1[i] =  bln1;
        bm.w_ln_2[i] =  wln2;
        bm.b_ln_2[i] =  bln2;

        bm.b_q[i] =  bq;
        bm.b_k[i] =  bk;
        bm.b_v[i] =  bv;
        bm.b_o[i] =  bo;
        bm.b_i_1[i] =  bi1;
        bm.b_i_2[i] =  bi2;
    }

    bm.w_p = read_data(model_dir + wp_fname);
    bm.b_p = read_bias(model_dir + bp_fname, 768);
    
    bm.w_c = read_data(model_dir + wc_fname);
    bm.b_c = read_bias(model_dir + bp_fname, num_class);

    return bm;
}