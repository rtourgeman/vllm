# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import pickle
from typing import Optional, Dict
import torch

try:
    from nixl._api import nixl_agent, nixl_xfer_handle, nixl_agent_config
    from nixl.logging import get_logger
except ImportError:
    from nixl_cu12._api import nixl_agent, nixl_xfer_handle, nixl_agent_config
    from nixl_cu12.logging import get_logger

logger = get_logger(__name__)


class NixlRequestStatus(enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    ERROR = "ERROR"


class NixlRequestType(enum.Enum):
    SEND = "SEND"
    RECV = "RECV"


class NixlRequest:
    def __init__(self, request_type: NixlRequestType, remote_agent: str, tag: str, seq: int,
                 xfer_handle: Optional[nixl_xfer_handle] = None, reg_descs=None, local_descs=None):
        self.type = request_type
        self.remote_agent = remote_agent
        self.tag = tag
        self.seq = seq
        self.xfer_handle = xfer_handle
        self.reg_descs = reg_descs
        self.local_descs = local_descs
        self._completed = False


class NixlCommunicationManager:
    """Provides isend/irecv semantics over NIXL using notifications for coordination."""

    def __init__(self, agent: nixl_agent):
        self.agent = agent
        self._notif_buffer: Dict[str, list[bytes]] = {}
        self._send_seq_counters: Dict[tuple, int] = {}
        self._recv_seq_counters: Dict[tuple, int] = {}

    def _poll_notifications(self):
        notifs = self.agent.get_new_notifs()
        for agent_name, msgs in notifs.items():
            if agent_name not in self._notif_buffer:
                self._notif_buffer[agent_name] = []
            self._notif_buffer[agent_name].extend(msgs)

    def _find_notification(
        self, src_agent: str, msg_type: str, tag: str, seq: int
    ) -> Optional[tuple]:
        self._poll_notifications()
        if src_agent not in self._notif_buffer:
            return None
        for i, raw in enumerate(self._notif_buffer[src_agent]):
            msg = pickle.loads(raw)
            if msg[0] == msg_type and msg[1] == tag and msg[2] == seq:
                self._notif_buffer[src_agent].pop(i)
                return msg
        return None

    def isend(
        self, tensor: torch.Tensor, dst_agent: str, tag: Optional[str] = None
    ) -> NixlRequest:
        """Non-blocking send. Returns Request to track via progress()."""
        if tag is None:
            tag = "default"

        key = (dst_agent, tag)
        seq = self._send_seq_counters.get(key, 0)
        self._send_seq_counters[key] = seq + 1

        # Create local descs for send buffer
        local_descs = self.agent.get_xfer_descs(tensor)

        return NixlRequest(NixlRequestType.SEND, dst_agent, tag, seq, local_descs=local_descs)

    def irecv(
        self, tensor: torch.Tensor, src_agent: str, tag: Optional[str] = None
    ) -> NixlRequest:
        """Non-blocking receive into pre-allocated tensor. Returns Request to track via progress()."""
        if tag is None:
            tag = "default"

        key = (src_agent, tag)
        seq = self._recv_seq_counters.get(key, 0)
        self._recv_seq_counters[key] = seq + 1

        # Send buffer info so sender can write to it
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        device = tensor.get_device()
        if device == -1:
            device = 0
        mem_type = "cuda" if tensor.is_cuda else "cpu"

        msg = pickle.dumps(("READY", tag, seq, ptr, size, device, mem_type))
        self.agent.send_notif(src_agent, msg)

        return NixlRequest(NixlRequestType.RECV, src_agent, tag, seq)

    def _setup_send_transfer(self, request: NixlRequest, msg: tuple) -> NixlRequestStatus:
        # msg = ("READY", tag, seq, ptr, size, device, mem_type)
        _, _, _, ptr, size, device, mem_type = msg

        # Create remote descs from receiver's buffer info
        remote_descs = self.agent.get_xfer_descs([(ptr, size, device)], mem_type=mem_type)

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", request.local_descs, remote_descs, request.remote_agent
        )

        if not xfer_handle:
            logger.error(f"initialize_xfer failed for {request.remote_agent}")
            return NixlRequestStatus.ERROR

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            logger.error(f"transfer initiation failed for {request.remote_agent}")
            return NixlRequestStatus.ERROR

        request.xfer_handle = xfer_handle
        return NixlRequestStatus.IN_PROGRESS

    def progress(self, request: NixlRequest) -> NixlRequestStatus:
        """Call repeatedly until DONE or ERROR."""
        if request._completed:
            return NixlRequestStatus.DONE

        if request.type == NixlRequestType.SEND:
            if request.xfer_handle is None:
                msg = self._find_notification(request.remote_agent, "READY", request.tag, request.seq)
                if msg is None:
                    return NixlRequestStatus.IN_PROGRESS
                status = self._setup_send_transfer(request, msg)
                if status == NixlRequestStatus.ERROR:
                    return NixlRequestStatus.ERROR

            state = self.agent.check_xfer_state(request.xfer_handle)
            if state == "DONE":
                msg = pickle.dumps(("COMPLETE", request.tag, request.seq))
                self.agent.send_notif(request.remote_agent, msg)
                request._completed = True
                return NixlRequestStatus.DONE
            elif state == "ERR":
                logger.error(f"transfer failed for {request.remote_agent}")
                return NixlRequestStatus.ERROR
            return NixlRequestStatus.IN_PROGRESS

        elif request.type == NixlRequestType.RECV:
            if self._find_notification(request.remote_agent, "COMPLETE", request.tag, request.seq):
                request._completed = True
                return NixlRequestStatus.DONE
            return NixlRequestStatus.IN_PROGRESS

        logger.error(f"unknown request type: {request.type}")
        return NixlRequestStatus.ERROR

    def batch_progress(self, requests: list[NixlRequest]) -> Dict[NixlRequest, NixlRequestStatus]:
        return {req: self.progress(req) for req in requests}
