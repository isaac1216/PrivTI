/*
HE instance
- Only support BFV for now
*/
#ifndef HE_H__
#define HE_H__

#include "LinearHE/utils-HE.h"
#include "seal/util/polyarithsmallmod.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>

using namespace sci;
using namespace std;
using namespace seal;

class HE
{
public:

    int party;
    NetIO *io;

    size_t poly_modulus_degree;
    uint64_t plain_mod;
    // plain_mod / 2
    uint64_t plain_mod_2;

    SEALContext *context;
	Encryptor *encryptor;
	Decryptor *decryptor;
	Evaluator *evaluator;
	Ciphertext *zero;

    HE();
    HE(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> coeff_bit_sizes, 
        uint64_t plain_mod);

    ~HE();
	
};

#endif