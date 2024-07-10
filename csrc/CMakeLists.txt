FILE(GLOB _BACKEND_SRCS
    api/*.cpp
    aten/*.cpp
    aten/generated/*.cpp
    core/*.cpp
    npu/*.cpp)

FILE(GLOB _EXCLUDE
    aten/generated/python_functions*
    aten/generated/VariableTypeEverything.cpp
    aten/generated/ADInplaceOrViewTypeEverything.cpp
    aten/generated/RegisterFunctionalizationEverything.cpp)

FOREACH(ITEM ${_EXCLUDE})
    LIST(REMOVE_ITEM _BACKEND_SRCS ${ITEM})
ENDFOREACH()

LIST(APPEND BACKEND_SRCS ${_BACKEND_SRCS})

# Pass to parent
set(BACKEND_SRCS ${BACKEND_SRCS} PARENT_SCOPE)

FILE(GLOB _GENERATED_SRCS
    aten/generated/python_functions_*)

LIST(APPEND GENERATED_SRCS ${_GENERATED_SRCS})
set(GENERATED_SRCS ${GENERATED_SRCS} PARENT_SCOPE)
