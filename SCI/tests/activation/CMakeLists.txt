macro (add_test_IRON _name)
	add_executable(IRON-${_name} "iron_${_name}.cpp")
    target_link_libraries(IRON-${_name} SCI-Math) 
    target_link_libraries(IRON-${_name} SCI-FloatingPoint) 
endmacro()

macro (add_test_BOLT _name)
	add_executable(BOLT-${_name} "bolt_${_name}.cpp")
    target_link_libraries(BOLT-${_name} SCI-FloatingPoint)
    target_link_libraries(BOLT-${_name} SCI-Math) 
endmacro()

add_test_IRON(layer_norm)
add_test_IRON(softmax)
add_test_IRON(gelu)
add_test_IRON(tanh)

add_test_BOLT(softmax)
add_test_BOLT(gelu)
add_test_BOLT(tanh)
add_test_BOLT(layer_norm)

