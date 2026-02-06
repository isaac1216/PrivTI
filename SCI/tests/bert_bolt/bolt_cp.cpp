#include "linear.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

// --- Parties Enum ---
enum Party { ALICE = 1, BOB = 2 };

// --- Mock Network IO for Simulation ---
// This class simulates network communication within a single program
// using a thread-safe queue.
class NetIO {
public:
    void send_data(const void* data, uint64_t size) {
        std::vector<char> buffer(size);
        memcpy(buffer.data(), data, size);
        {
            std::lock_guard<std::mutex> lock(mtx);
            send_queue.push(buffer);
        }
        cv.notify_one();
    }

    void recv_data(void* data, uint64_t size) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return !send_queue.empty(); });
        
        std::vector<char> buffer = send_queue.front();
        send_queue.pop();
        
        if (buffer.size() != size) {
            throw std::runtime_error("NetworkIO error: received data size mismatch.");
        }
        memcpy(data, buffer.data(), size);
    }
private:
    std::queue<std::vector<char>> send_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

// --- Helper Functions ---
// Decrypts a row of data for verification
std::vector<uint64_t> decrypt_row(HE* he, const std::vector<seal::Ciphertext>& ct_row, size_t num_elements) {
    std::vector<uint64_t> result;
    result.reserve(num_elements);
    size_t slot_count = he->encoder->slot_count();

    for (const auto& ct : ct_row) {
        seal::Plaintext pt;
        he->decryptor->decrypt(ct, pt);
        std::vector<uint64_t> decoded_vec;
        he->encoder->decode(pt, decoded_vec);
        size_t elements_to_copy = std::min((size_t)num_elements - result.size(), slot_count);
        result.insert(result.end(), decoded_vec.begin(), decoded_vec.begin() + elements_to_copy);
    }
    result.resize(num_elements);
    return result;
}

// Thread function for BOB (the key generator)
void bob_key_exchange(NetIO* io_channel) {
    std::cout << "[BOB_THREAD] Starting key generation and sending...\n";
    // This constructor will generate keys and send them via the io_channel
    Linear bob_module(BOB, io_channel); 
    std::cout << "[BOB_THREAD] Keys sent. Thread finished.\n";
}


int main() {
    // --- 1. Settings ---
    const int BATCH_SIZE = 128;
    const int INPUT_DIM = 768;
    const int OUTPUT_DIM = 64;
    
    std::cout << "Bolt HE Final Example: Two-Party Ciphertext-Plaintext Multiplication\n";
    std::cout << "Dimensions: (Ciphertext " << BATCH_SIZE << "x" << INPUT_DIM 
              << ") x (Plaintext " << INPUT_DIM << "x" << OUTPUT_DIM << ")\n\n";

    // --- 2. Two-Party Key Exchange Simulation ---
    std::cout << "[Setup] Simulating two-party key exchange...\n";
    NetIO io_channel;

    // Launch BOB in a separate thread to generate and send keys.
    std::thread bob_thread(bob_key_exchange, &io_channel);

    // The main thread will act as ALICE, receiving the keys.
    // The HE constructor for ALICE will block until it receives data from BOB.
    std::cout << "[ALICE] Waiting to receive keys...\n";
    Linear alice_module(ALICE, &io_channel);
    
    // Wait for the BOB thread to complete its task.
    bob_thread.join();
    std::cout << "[Setup] Key exchange complete. ALICE is initialized.\n";

    HE* he = alice_module.he_8192_tiny; // All computations will use ALICE's HE context
    uint64_t plain_mod = he->plain_mod;


    // --- 3. Server (BOB's role) Preprocesses Weights (Offline) ---
    // In this simulation, ALICE will also play the server's role for computation.
    std::cout << "\n--- [Server Role - Offline Phase] ---\n";
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<uint64_t>> W(INPUT_DIM, std::vector<uint64_t>(OUTPUT_DIM));
    std::vector<uint64_t> B(OUTPUT_DIM);
    for(int i = 0; i < INPUT_DIM; ++i)
        for(int j = 0; j < OUTPUT_DIM; ++j) W[i][j] = (i + j) % plain_mod;
    for(int j = 0; j < OUTPUT_DIM; ++j) B[j] = j % plain_mod;
    
    FCMetadata metadata;
    metadata.image_size = BATCH_SIZE; 
    metadata.filter_h = INPUT_DIM;
    metadata.filter_w = OUTPUT_DIM;
    metadata.slot_count = he->encoder->slot_count();

    PreprocessParams_2 pp = alice_module.params_preprocessing_ct_pt(he, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM, W, B, metadata);
    
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    std::cout << "[Server] Weight preprocessing finished in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_preprocess - start_preprocess).count() << " ms.\n";

    // --- 4. Client (ALICE's role) Encrypts Input ---
    std::cout << "\n--- [Client Role - Encryption] ---\n";
    auto start_encrypt = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<uint64_t>> X(BATCH_SIZE, std::vector<uint64_t>(INPUT_DIM));
    for(int i = 0; i < BATCH_SIZE; ++i)
        for(int j = 0; j < INPUT_DIM; ++j) X[i][j] = (i + 1);

    std::vector<std::vector<seal::Ciphertext>> X_encrypted(BATCH_SIZE);
    #pragma omp parallel for
    for(int i = 0; i < BATCH_SIZE; ++i) {
        X_encrypted[i] = alice_module.bert_efficient_preprocess_vec(he, X[i], metadata);
    }
    
    auto end_encrypt = std::chrono::high_resolution_clock::now();
    std::cout << "[Client] Input matrix encrypted in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_encrypt - start_encrypt).count() << " ms.\n";

    // --- 5. Server (BOB's role) Performs Computation (Online) ---
    std::cout << "\n--- [Server Role - Online Phase] ---\n";
    auto start_compute = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<seal::Ciphertext>> Y_encrypted(BATCH_SIZE);
    #pragma omp parallel for
    for(int i = 0; i < BATCH_SIZE; ++i) {
        Y_encrypted[i] = alice_module.linear_2(he, X_encrypted[i], pp, metadata);
    }

    auto end_compute = std::chrono::high_resolution_clock::now();
    std::cout << "[Server] HE computation finished in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count() << " ms.\n";
    
    // --- 6. Client (ALICE's role) Decrypts and Verifies ---
    std::cout << "\n--- [Client Role - Verification] ---\n";

    std::vector<std::vector<uint64_t>> Y_plain(BATCH_SIZE, std::vector<uint64_t>(OUTPUT_DIM, 0));
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            uint64_t sum = 0;
            for (int k = 0; k < INPUT_DIM; ++k) {
                sum += X[i][k] * W[k][j];
            }
            Y_plain[i][j] = (sum + B[j]) % plain_mod;
        }
    }

    std::vector<uint64_t> Y_decrypted_first_row = decrypt_row(he, Y_encrypted[0], OUTPUT_DIM);
    std::vector<uint64_t> Y_decrypted_last_row = decrypt_row(he, Y_encrypted[BATCH_SIZE - 1], OUTPUT_DIM);
    
    std::cout << "\n--- Verification ---\n";
    bool success = true;
    if (Y_plain[0][0] != Y_decrypted_first_row[0]) success = false;
    if (Y_plain[BATCH_SIZE - 1][OUTPUT_DIM - 1] != Y_decrypted_last_row[OUTPUT_DIM - 1]) success = false;

    std::cout << "Expected Y[0][0]: " << Y_plain[0][0] << ", Got: " << Y_decrypted_first_row[0] << std::endl;
    std::cout << "Expected Y[127][63]: " << Y_plain[BATCH_SIZE - 1][OUTPUT_DIM - 1] << ", Got: " << Y_decrypted_last_row[OUTPUT_DIM - 1] << std::endl;

    if (success) {
        std::cout << "\n✅ Verification PASSED!\n";
    } else {
        std::cout << "\n❌ Verification FAILED!\n";
    }

    return 0;
}
