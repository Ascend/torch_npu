# Directory Structure

```Shell
.
|-- codegen               //代码生成：包括前反向注册，反向实现，反向绑定，自定义算子路由，重定向路由等
|   |-- autograd
|   |   `-- templates     //TODO: will be moved into csrc/aten
|   `-- templates         //TODO: will be moved into csrc/aten
|-- csrc                  //与PyTorch相关的C++实现，不涉及具体后端的实现代码，理论上仅包含后端接口调用
|   |-- api               //libtorch功能
|   |-- aten              //代码生成：仅包含 wrap 以及 PyTorch算子注册；后续考虑把 Tensor & Storage & Serialization 移动到这里，因为这三个都是Tensor的相关逻辑
|   |-- core              //设备无关代码：通用内存池，各种功能模块的基类
|   |-- distributed       //分布式
|   `-- npu               //npu的PyTorch相关实现，理论上仅包含接口调用（后续如果能抽象成功，将改名成通用后端：backend之类）
|-- npu                   //与具体设备强相关，几乎不涉及PyTorch相关概念（除了ATen之外），为csrc提供底层API(后续计划移动到third_party中)
|   |-- acl
|   |   |-- include
|   |   `-- libs
|   |-- aten
|   |   |-- common
|   |   |-- mirror
|   |   `-- ops
|   |-- core
|   |   |-- interface
|   |   |-- register
|   |   `-- sys_ctrl
|   `-- framework
|       |-- aoe
|       |-- autograd
|       |-- contiguous
|       |-- interface
|       `-- utils
|-- third_party
|   |-- googletest
|   `-- op-plugin     //后续计划移动到third_party/npu中，但是意义不大，依然是本体项目的submodule
`-- torch_npu         //Pytorch的Python接口实现
    |-- csrc          //Python&C++绑定
    |   |-- core      //通用能力，仅为Python提供
    |   `-- npu       //面相Python需要暴露的所有底层能力的Python绑定
    |-- meta          //meta算子注册（用来计算自定义算子的输出内存占用）
    |-- npu           //npu设备相关功能模块的封装
    |-- optim         //NPU专属优化器
    |-- testing
    `-- utils         //各种patch
```
