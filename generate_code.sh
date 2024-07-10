#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"
pytorch_version="$2"

IFS='.' read -ra version_parts <<< "$pytorch_version"

pytorch_dir="v${version_parts[0]}r${version_parts[1]}"

# file-level
find $CDIR/third_party/op-plugin -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | \
xargs -I {} sed -i "s/torch_npu\/csrc\/core\/npu\/NPUStream.h/csrc\/npu\/NPUStream.h/g" {}
find $CDIR/third_party/op-plugin -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | \
xargs -I {} sed -i "s/torch_npu\/csrc\/aten\/mirror\/NPUTypeProperties.h/npu\/aten\/mirror\/NPUTypeProperties.h/g" {}


# derectory-level
find $CDIR/third_party/op-plugin -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | \
xargs -I {} sed -i "s/torch_npu\/csrc\/aten/csrc\/aten\/generated/g" {}
find $CDIR/third_party/op-plugin -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | \
xargs -I {} sed -i "s/torch_npu\/csrc\/core\/npu/npu\/core/g" {}
find $CDIR/third_party/op-plugin -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | \
xargs -I {} sed -i "s/torch_npu\/csrc\/framework/npu\/framework/g" {}
find $CDIR/third_party/op-plugin -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | \
xargs -I {} sed -i "s/\"third_party\/acl\/inc/\"npu\/acl\/include/g" {}

file=$CDIR/third_party/op-plugin/gencode.sh

if [ -f "${file}" ]; then
  bash ${file} ${pytorch_version} ${python_execute}
fi

op_plugin_config_path=$CDIR/third_party/op-plugin/op_plugin/config/$pytorch_dir
source_yaml="$CDIR/csrc/aten/npu_native_functions.yaml"
testing_source_yaml="$CDIR/test/ops_unsupport_list.yaml"

op_plugin_functions_yaml_path="$op_plugin_config_path/npu_native_functions.yaml"

${python_execute} -m codegen.gen_backend_stubs  \
  --output_dir="csrc/aten/generated" \
  --source_yaml="$source_yaml" \
  --impl_path="$CDIR/aten" \
  --op_plugin_impl_path="$CDIR/third_party/op-plugin/op_plugin/ops" \
  --op_plugin_yaml_path="$op_plugin_config_path/op_plugin_functions.yaml"

${python_execute} -m codegen.autograd.gen_autograd \
  --out_dir="$CDIR/csrc/aten/generated" \
  --autograd_dir="$CDIR/codegen/autograd" \
  --npu_native_function_dir="$source_yaml"

${python_execute} -m codegen.codegen_ops_info

if [ -f $CDIR/third_party/op-plugin/codegen/templates/_op_plugin_docs.py ]; then
  if [ -f $CDIR/torch_npu/_op_plugin_docs.py ]; then
      # remove _op_plugin_docs.py
      rm $CDIR/torch_npu/_op_plugin_docs.py
      echo "Existing _op_plugin_docs.py in torch_npu deleted."
  fi
  cp -f $CDIR/third_party/op-plugin/codegen/templates/_op_plugin_docs.py $CDIR/torch_npu/_op_plugin_docs.py
fi
