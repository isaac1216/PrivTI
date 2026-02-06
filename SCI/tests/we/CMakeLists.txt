cmake_minimum_required(VERSION 3.10)
project(TestOps)

set(LINK_LIBS SCI-Math SCI-FloatingPoint)

# 定义一个宏，生成 test-xxx 可执行文件
macro (add_test_exec _name)
    add_executable(${_name}-we "${_name}-we.cpp")
    target_link_libraries(${_name}-we ${LINK_LIBS})
endmacro()

# 用宏，一次生成三个
add_test_exec(boltgelu)
add_test_exec(boltsoftmax)
add_test_exec(boltlayernorm)
add_test_exec(testgelu)
add_test_exec(testsoftmax)
add_test_exec(testlayernorm)
