#include "hccl/hccl.h"

HcclResult HcclAllReduce(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    HcclReduceOp op,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclBroadcast(
    void* buf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm) {
  return HCCL_SUCCESS;
}

HcclResult HcclReduceScatter(
    void* sendBuf,
    void* recvBuf,
    uint64_t recvCount,
    HcclDataType dataType,
    HcclReduceOp op,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(
    uint32_t nRanks,
    const HcclRootInfo* rootInfo,
    uint32_t rank,
    HcclComm* comm) {
  return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm commHandle, char* commName) {
  return HCCL_SUCCESS;
}

HcclResult HcclAllGather(
    void* sendBuf,
    void* recvBuf,
    uint64_t sendCount,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclRecv(
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t srcRank,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclSend(
    void* sendBuf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t destRank,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo) {
  return HCCL_SUCCESS;
}

HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError) {
  return HCCL_SUCCESS;
}

HcclResult HcclScatter(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclBatchSendRecv(
    HcclSendRecvItemDef* sendRecvInfo,
    uint32_t itemNum,
    HcclComm comm,
    aclrtStream stream) {
  return HCCL_SUCCESS;
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms) {
  return HCCL_SUCCESS;
}
