add_library(LIB_BERT_IRON 
    he.cpp
    nonlinear.cpp
    linear.cpp
    bert_utils.cpp
    bert.cpp
    )

target_link_libraries(LIB_BERT_IRON SCI-FloatingPoint)
target_link_libraries(LIB_BERT_IRON SCI-HE)

# add_executable(BERT_NONLINEAR "test_nonlinear.cpp")
# target_link_libraries(BERT_NONLINEAR NONLINEAR)

add_executable(IRON_BERT "iron_bert.cpp")
target_link_libraries(IRON_BERT LIB_BERT_IRON)
