# PrivTI: Efficient End-to-End Privacy-Preserving Inference for Transformer-based Models in MLaaS

This repo contains the source code for the PrivTI implemented in C++. The code has been developed and tested with Ubuntu 22.04.

## Requirements

 - g++ (version >= 8)
 - cmake
 - make
 - libgmp-dev
 - libmpfr-dev
 - libssl-dev  
 - SEAL 3.3.2
 - Eigen 3.3

SEAL and Eigen are included in `extern/` and are automatically compiled and installed if not found. The other packages can be installed directly using `sudo apt-get install <package>` on Linux. 

## Build and Run

To build the library:

```
mkdir build
cd ./build
cmake ..
make
```

On successful compilation, the test and network binaries will be created in `build/bin/`.

Run the tests as follows to make sure everything works as intended:

```
./<test> r=1 [port=port] & ./<test> r=2 [port=port]
```

## Acknowledgements

This library includes code from the following external repositories:

 - [mpc-msri/EzPC](https://github.com/mpc-msri/EzPC) for MPC implementation.
 - [mpc-msri/EzPC/SCI](https://github.com/mpc-msri/EzPC/tree/master/SCI) for MPC implementation.
 - [microsoft/SEAL](https://github.com/microsoft/SEAL) for HE implementation.

 - [emp-toolkit/emp-tool](https://github.com/emp-toolkit/emp-tool/tree/c44566f40690d2f499aba4660f80223dc238eb03/emp-tool) for cryptographic tools and network I/O.
