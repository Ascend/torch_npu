#include "torch_npu/csrc/core/TensorType.h"
#include <c10/core/DeviceType.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/python_arg_parser.h>

namespace torch_npu {
namespace utils {

using namespace at;
using namespace torch::autograd;

static Backend backend = c10::Backend::PrivateUse1;
struct PyTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  char name[64];
  int scalar_type;

  DispatchKey get_dispatch_key() const {
    return backendToDispatchKey(backend);
  }

  ScalarType get_scalar_type() const {
    return static_cast<ScalarType>(scalar_type);
  }
};

static_assert(
    std::is_standard_layout<PyTensorType>::value,
    "PyTensorType must be standard layout");

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types);

static PyObject* Tensor_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  static auto warn_once = []() {
    auto name = c10::get_privateuse1_backend();
    std::cout
        << "Warning: The torch." << name
        << ".*DtypeTensor constructors are no longer recommended. "
           "It's best to use methods such as torch.tensor(data, dtype=*, device='"
        << name << "') to create tensors." << std::endl;
    return true;
  }();
  TORCH_CHECK_TYPE(
      c10_npu::device_count() != 0,
      "type ",
      tensor_type.name,
      " not available. Torch not compiled with npu enabled.",
      PTA_ERROR(ErrCode::TYPE))
  torch::utils::device_lazy_init(at::kPrivateUse1);
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(
      tensor_type.get_dispatch_key(),
      tensor_type.get_scalar_type(),
      args,
      kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_instancecheck(PyObject* _self, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto self = (PyTensorType*)_self;
  if (THPVariable_Check(arg)) {
    const auto& var = THPVariable_Unpack(arg);

    if (legacyExtractDispatchKey(var.key_set()) == self->get_dispatch_key() &&
        var.scalar_type() == static_cast<ScalarType>(self->scalar_type)) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

PyObject* Tensor_dtype(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->dtype);
}

PyObject* Tensor_layout(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->layout);
}

PyObject* Tensor_is_sparse(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

static struct PyMethodDef metaclass_methods[] = {
    {"__instancecheck__", Tensor_instancecheck, METH_O, nullptr},
    {nullptr}};

using getter = PyObject* (*)(PyObject*, void*);
static getter always_true = [](auto x, auto y) { Py_RETURN_TRUE; };

static PyGetSetDef* get_metaclass_properties() {
  static std::string is_pv1 = "is_" + c10::get_privateuse1_backend();
  return new PyGetSetDef[5]{
      {"dtype", (getter)Tensor_dtype, nullptr, nullptr, nullptr},
      {"layout", (getter)Tensor_layout, nullptr, nullptr, nullptr},
      {is_pv1.c_str(), always_true, nullptr, nullptr, nullptr},
      {"is_sparse", (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
      {nullptr}};
}

static PyTypeObject metaclass = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.tensortype", /* tp_name */
    sizeof(PyTypeObject) /* tp_basicsize */
};

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = get_metaclass_properties();
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static PyTypeObject tensor_type_prototype = {
    PyVarObject_HEAD_INIT(&metaclass, 0) nullptr, /* tp_name */
    sizeof(PyTensorType) /* tp_basicsize */
};

static void py_initialize_tensor_type(
    PyTypeObject& type,
    const char* name,
    PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // We copy the basic object fields from a prototype definition and
  // initialize the remaining fields below.
  memcpy(&type, &tensor_type_prototype, sizeof(PyTypeObject));
  // Subclassing from torch.<ScalarType>Tensor isn't supported.
  // (Py_TPFLAGS_BASETYPE omitted). Subclassing torch.Tensor still allowed.
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_new = Tensor_new;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static std::string get_name(ScalarType scalarType) {
  std::ostringstream ss;
  ss << "torch." << c10::get_privateuse1_backend() << "."
     << toString(scalarType) << "Tensor";
  return ss.str();
}

static void set_type(PyTensorType& type_obj, ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout = torch::getTHPLayout(c10::layout_from_backend(backend));
  type_obj.dtype = torch::getTHPDtype(scalarType);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_tensor_dict() {
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch) {
    throw python_error();
  }

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class) {
    throw python_error();
  }

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  TORCH_CHECK(
      tensor_type->tp_base,
      "missing base type for Tensor",
      PTA_ERROR(ErrCode::TYPE));

  auto res = THPObjectPtr(PyDict_New());
  if (!res) {
    throw python_error();
  }

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

static std::vector<PyTensorType> tensor_types;

static void initialize_npu_aten_types(
    std::vector<PyTensorType>& tensor_types,
    std::vector<ScalarType>& scalar_types) {
  tensor_types.resize(scalar_types.size());

  for (size_t i = 0, end = scalar_types.size(); i != end; i++) {
    auto& tensor_type = tensor_types[i];
    ScalarType scalar_type = scalar_types[i];
    set_type(tensor_type, scalar_type);
    set_name(tensor_type, get_name(scalar_type));
  }
}

void initialize_python_bindings(std::vector<ScalarType>& scalar_types) {
  // Initialize the at::Type* pointers, name, and properties of the
  // PyTensorType vector. After this call, the vector must not be resized.
  initialize_npu_aten_types(tensor_types, scalar_types);

  // Initialize the Python metaclass for the torch.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype
  // property on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.npu.FloatTensor.add`.
  auto tensor_dict = get_tensor_dict();

  // Initialize each Python type object. e.g. torch.npu.FloatTensor,
  // torch.npu.DoubleTensor, etc.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(
        tensor_type.py_type, tensor_type.name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g.
  // torch.npu.FloatTensor is added to the `torch_npu` module as
  // `FloatTensor`. Also add all the type objects to the set
  // torch_npu._tensor_classes.
  py_bind_tensor_types(tensor_types);
}

static void py_bind_tensor_types(
    const std::vector<PyTensorType>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes)
    throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type.name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj)
      throw python_error();

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* generate_tensor_types(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* p_list;
  if (!PyArg_ParseTuple(args, "O", &p_list)) {
    return nullptr;
  }

  std::vector<ScalarType> scalar_types;
  Py_ssize_t list_len = PyList_Size(p_list);
  for (Py_ssize_t i = 0; i < list_len; ++i) {
    PyObject* item = PyList_GetItem(p_list, i);
    ScalarType scalar_type = torch::toScalarType(item);
    scalar_types.push_back(scalar_type);
  }

  initialize_python_bindings(scalar_types);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef TorchNpuExtensionMethods[] = {
    {"generate_tensor_types",
     (PyCFunction)generate_tensor_types,
     METH_VARARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* npu_extension_functions() {
  return TorchNpuExtensionMethods;
}
} // namespace utils
} // namespace torch_npu
