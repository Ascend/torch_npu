FILE(GLOB _ATEN_SRCS
    common/*.cpp
    mirror/*.cpp
    ops/*.cpp
    ops/op_api/*.cpp)

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})
# Pass to parent
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)
