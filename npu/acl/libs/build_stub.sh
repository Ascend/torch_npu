#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libhccl.so -I../include hccl.cpp

gcc -fPIC -shared -o libascendcl.so -I../include acl.cpp

gcc -fPIC -shared -o libacl_op_compiler.so -I../include acl_op_compiler.cpp

gcc -fPIC -shared -o libge_runner.so -I../include ge_runner.cpp ge_api.cpp

gcc -fPIC -shared -o libgraph.so -I../include graph.cpp operator_factory.cpp operator.cpp tensor.cpp

gcc -fPIC -shared -o libacl_tdt_channel.so -I../include acl_tdt.cpp

