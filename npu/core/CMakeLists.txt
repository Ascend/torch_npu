FILE(GLOB _CORE_SRCS *.cpp npu/*.cpp npu/*/*.cpp)
FILE(GLOB _CORE_SRCS
    *.cpp
    interface/*.cpp
    register/*.cpp
    sys_ctrl/*.cpp)

LIST(APPEND CORE_SRCS ${_CORE_SRCS})

# Pass to parent
set(CORE_SRCS ${CORE_SRCS} PARENT_SCOPE)
