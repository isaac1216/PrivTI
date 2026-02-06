cmake_minimum_required(VERSION 3.10)
project(TestOps)

set(LINK_LIBS SCI-Math SCI-FloatingPoint)

# 定义一个宏，生成 test-xxx 可执行文件
macro (add_test_exec _name)
    add_executable(test-${_name} "test-${_name}.cpp")
    target_link_libraries(test-${_name} ${LINK_LIBS})
endmacro()

# 用宏，一次生成三个
add_test_exec(gelu)
add_test_exec(softmax)
add_test_exec(layernorm)
add_test_exec(silu)
add_test_exec(rms)
add_test_exec(lut)
