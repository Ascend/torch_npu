function(add_shared_library name sources)
    add_library(${name} SHARED ${sources})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
endfunction()

set(LIBRARIES
    hccl "hccl.cpp"
    ascendcl "acl.cpp,acl_rt.cpp"
    acl_op_compiler "acl_op_compiler.cpp"
    ge_runner "ge_runner.cpp,ge_api.cpp"
    graph "graph.cpp,operator_factory.cpp,operator.cpp,tensor.cpp"
    acl_tdt_channel "acl_tdt.cpp"
)

set(index 0)
list(LENGTH LIBRARIES num_entries)

while(index LESS num_entries)
    list(GET LIBRARIES ${index} library_name)
    math(EXPR index "${index}+1")
    
    list(GET LIBRARIES ${index} sources)
    string(REPLACE "," ";" sources_list ${sources})
    
    add_shared_library(${library_name} "${sources_list}")
    math(EXPR index "${index}+1")
endwhile()
