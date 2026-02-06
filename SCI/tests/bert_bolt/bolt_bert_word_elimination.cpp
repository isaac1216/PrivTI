/*
*/

#include "bert.h"
#include <fstream>

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int port = 8000;
string address = "127.0.0.1";
int num_threads = 4;
int bitlength = 37;

string path = "/home/ubuntu/prune/mrpc/";
string output_file_path = "/home/ubuntu/clive/EzPC/ppnlp_test.txt";
int num_class = 2;
int sample_id = 0;
int num_sample = 1;

bool pruning = true;

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("path", path, "Path of dataset and model");
    amap.arg("num_class", num_class, "Number of classification labels");
    amap.arg("id", sample_id, "Index of first sample");
    amap.arg("num_sample", num_sample, "Length of test data");
    amap.arg("prune", pruning, "Input pruning");
    amap.arg("output", output_file_path, "Path of output");
    amap.parse(argc, argv);

    cout << ">>> Evaluating Bert" << endl;
    cout << "-> Role: " << party << endl;
    cout << "-> Address: " << address << endl;
    cout << "-> Port: " << port << endl;
    cout << "<<<" << endl << endl;

    Bert bt(party, port, address, path + "/weights_txt/", pruning);

    auto start = high_resolution_clock::now();
    
    vector<vector<double>> inference_results;
    vector<int> predicted_labels;

    if(party == ALICE){
        for(int i = sample_id; i < sample_id + num_sample; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run("", "");
        }
    } else{
        ofstream file(output_file_path);
        if (!file) {
            std::cerr << "Could not open the file: " << output_file_path <<std::endl;
            return {};
        }
        for(int i = sample_id; i < sample_id + num_sample; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run(
                path + "/weights_txt/inputs_" + to_string(i) + "_data.txt",
                path + "/weights_txt/inputs_" + to_string(i) +  "_mask.txt"
                );
            if(result.size() == 1){
                file << result[0]<< endl;
            } else if(result.size() == 2){
                // inference_results.push_back(result);
                auto max_ele = max_element(result.begin(), result.end());
                int max_index = distance(result.begin(), max_ele);
                // predicted_labels.push_back(max_index);
                file << max_index << "," 
                        << result[0]<< "," 
                        << result[1] << endl;
            }
        }
        file.close();
    }
    
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    
    cout << "-> End to end takes: " << interval.count() << "sec" << endl;

    return 0;
}
