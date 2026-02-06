#include "he.h"

HE::HE(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> coeff_bit_sizes, 
        uint64_t plain_mod){
    this->party = party;
    this->poly_modulus_degree = poly_modulus_degree;
    this->io = io;
	this->plain_mod = plain_mod;
	this->plain_mod_2 = (plain_mod -1) / 2;

    // Generate keys
    EncryptionParameters parms(scheme_type::bfv);

	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(
		CoeffModulus::Create(poly_modulus_degree, coeff_bit_sizes));
	parms.set_plain_modulus(plain_mod);

	context = new SEALContext(parms, true, seal::sec_level_type::tc128);
	encoder = new BatchEncoder(*context);
	evaluator = new Evaluator(*context);

	if (party == BOB) {
		KeyGenerator keygen(*context);
		SecretKey sec_key = keygen.secret_key();
		PublicKey pub_key;
		keygen.create_public_key(pub_key);
		GaloisKeys gal_keys_;
		keygen.create_galois_keys(gal_keys_);
		RelinKeys relin_keys_;
		keygen.create_relin_keys(relin_keys_);

		stringstream os;
		pub_key.save(os);
		uint64_t pk_size = os.tellp();
		gal_keys_.save(os);
		uint64_t gk_size = (uint64_t)os.tellp() - pk_size;
		relin_keys_.save(os);
		uint64_t rk_size = (uint64_t)os.tellp() - pk_size - gk_size;

		string keys_ser = os.str();
		io->send_data(&pk_size, sizeof(uint64_t));
		io->send_data(&gk_size, sizeof(uint64_t));
		io->send_data(&rk_size, sizeof(uint64_t));
		io->send_data(keys_ser.c_str(), pk_size + gk_size + rk_size);

		stringstream os_sk;
		sec_key.save(os_sk);
		uint64_t sk_size = os_sk.tellp();
		string keys_ser_sk = os_sk.str();
		io->send_data(&sk_size, sizeof(uint64_t));
		io->send_data(keys_ser_sk.c_str(), sk_size);

		encryptor = new Encryptor(*context, pub_key);
		decryptor = new Decryptor(*context, sec_key);
	}
	else // party == ALICE
	{
		uint64_t pk_size;
		uint64_t gk_size;
		uint64_t rk_size;
		io->recv_data(&pk_size, sizeof(uint64_t));
		io->recv_data(&gk_size, sizeof(uint64_t));
		io->recv_data(&rk_size, sizeof(uint64_t));
		char *key_share = new char[pk_size + gk_size + rk_size];
		io->recv_data(key_share, pk_size + gk_size + rk_size);
		stringstream is;
		PublicKey pub_key;
		is.write(key_share, pk_size);
		pub_key.load(*context, is);
		gal_keys = new GaloisKeys();
		is.write(key_share + pk_size, gk_size);
		gal_keys->load(*context, is);
		relin_keys = new RelinKeys();
		is.write(key_share + pk_size + gk_size, rk_size);
		relin_keys->load(*context, is);
		delete[] key_share;

		uint64_t sk_size;
		io->recv_data(&sk_size, sizeof(uint64_t));
		char *key_share_sk = new char[sk_size];
		io->recv_data(key_share_sk, sk_size);
		stringstream is_sk;
		SecretKey sec_key;
		is_sk.write(key_share_sk, sk_size);
		sec_key.load(*context, is_sk);
		delete[] key_share_sk;
		decryptor = new Decryptor(*context, sec_key);

		encryptor = new Encryptor(*context, pub_key);
		vector<uint64_t> pod_matrix(poly_modulus_degree, 0ULL);
		Plaintext tmp;
		encoder->encode(pod_matrix, tmp);
		zero = new Ciphertext;
		encryptor->encrypt(tmp, *zero);
	}
    cout << "> HE instance initialized: " << endl;
    cout << "-> Poly Mod Degree: " << poly_modulus_degree << endl;
    cout << "-> Coeff Mod: " ;
	for(auto mod: coeff_bit_sizes){
		cout << mod << " ";
	}
	cout << endl;
    cout << "-> Plaintext Mod: " << plain_mod 
        << "(" << int (log2(plain_mod)) << " bits)" << endl;  
    cout << endl;
}

HE::HE(){
    // TODO: 
}

HE::~HE(){
    // TODO: 
}