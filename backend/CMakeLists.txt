FILE(GLOB _BACKEND_SRCS
    api/*.cpp
    core/*.cpp
    npu/*.cpp)

LIST(APPEND BACKEND_SRCS ${_BACKEND_SRCS})

# Pass to parent
set(BACKEND_SRCS ${BACKEND_SRCS} PARENT_SCOPE)

