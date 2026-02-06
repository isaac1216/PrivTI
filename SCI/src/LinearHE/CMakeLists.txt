find_package(OpenMP REQUIRED)
# find_package(OpenFHE)
# set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Wno-error")
find_package(SEAL 4.1.1 EXACT QUIET PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 4.1.1 was not found: clone and install SEAL locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init --recursive extern/SEAL
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        execute_process(COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/")
    execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/")
    find_package(SEAL 4.1.1 EXACT REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
endif()


execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/")
            
# link_directories( ${OpenFHE_LIBDIR} )
# link_directories( ${OPENMP_LIBRARIES} )
# set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
# link_libraries( ${OpenFHE_SHARED_LIBRARIES} )

add_library(SCI-LinearHE
    conv-field.cpp
    fc-field.cpp
    elemwise-prod-field.cpp
    utils-HE.cpp
    # bert-matmul.cpp
    # bert-matmul-cipher-efficient.cpp
    # bert-matmul-cipher-efficient-seal.cpp
    # bert-matmul-cipher-efficient-seal-ckks.cpp
    # iron-seal.cpp
    # iron-seal-inter1.cpp
    # bert-ct-pt.cpp
    # bert-ct-pt-inter1.cpp
    # utils-HE-openfhe.cpp
)

# include_directories( ${OPENMP_INCLUDES} )
# include_directories( ${OpenFHE_INCLUDE} )
# include_directories( ${OpenFHE_INCLUDE}/third-party/include )
# include_directories( ${OpenFHE_INCLUDE}/core )
# include_directories( ${OpenFHE_INCLUDE}/pke )

target_link_libraries(SCI-LinearHE
    PUBLIC
    SCI-utils
    SEAL::seal
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)
