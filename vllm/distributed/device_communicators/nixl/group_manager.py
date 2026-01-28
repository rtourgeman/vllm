# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
import time

try:
    from nixl._api import nixl_agent, nixl_agent_config
    from nixl.logging import get_logger
except ImportError:
    from nixl_cu12._api import nixl_agent, nixl_agent_config
    from nixl_cu12.logging import get_logger

from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.distributed.device_communicators.nixl.communication import NixlCommunicationManager, NixlRequestStatus


class NixlGroupManager:
    """ Manages a process group over NIXL. """
    
    def __init__(self, rank: int, group_prefix: str, agent_config=None):
        self.group_prefix = group_prefix
        self.agent_name = f"{group_prefix}_rank_{rank}"
        self.known_peers: dict[str, Any] = {}
        
        if agent_config is None:
            agent_config = nixl_agent_config(backends=["UCX"])
        
        self.agent = nixl_agent(self.agent_name, agent_config)
        self.comm_manager = NixlCommunicationManager(self.agent)
        
        logger.info(
            f"[NIXL] Created NixlGroupManager: agent_name={self.agent_name}, group_prefix={group_prefix}"
        )
    
    def update_peers(
        self,
        comm_group: StatelessProcessGroup,
        my_rank_in_group: int,
        peer_ranks: list[int]
    ):
        """Update peer list: remove stale peers, exchange metadata, add new peers."""
        logger.info(
            f"[NIXL] update_peers called: agent={self.agent_name}, "
            f"my_rank={my_rank_in_group}, world_size={comm_group.world_size}, "
            f"known_peers={len(self.known_peers)}"
        )
        
        peer_agent_names = {
            rank: f"{self.group_prefix}_rank_{rank}"
            for rank in peer_ranks
        }
        
        # Remove stale peers (scale-down)
        current_peer_set = set(peer_agent_names.values())
        current_peer_set.discard(self.agent_name)
        
        peers_to_remove = set(self.known_peers.keys()) - current_peer_set
        peers_removed = 0
        for peer_name in peers_to_remove:
            logger.debug(f"[NIXL] Removing stale peer: {peer_name}")
            self.agent.remove_remote_agent(peer_name)
            del self.known_peers[peer_name]
            peers_removed += 1
        
        if peers_removed > 0:
            logger.info(
                f"[NIXL] Removed {peers_removed} stale peer(s): {list(peers_to_remove)}"
            )
        
        my_metadata = self.agent.get_agent_metadata()
        logger.debug(f"[NIXL] Got my metadata, length={len(my_metadata)} bytes")
        
        logger.debug(f"[NIXL] Starting all_gather_obj for metadata exchange")
        metadata_list = comm_group.all_gather_obj(my_metadata)
        logger.debug(f"[NIXL] Received metadata from {len(metadata_list)} peers")
        
        # Add new peers
        new_peers_added = 0
        for rank_in_group in peer_ranks:
            if rank_in_group == my_rank_in_group:
                continue
            
            peer_name = peer_agent_names[rank_in_group]
            if peer_name not in self.known_peers:
                peer_metadata = metadata_list[rank_in_group]
                
                logger.debug(
                    f"[NIXL] Adding new peer: {peer_name} (rank {rank_in_group}), "
                    f"metadata_length={len(peer_metadata)}"
                )
                
                self.agent.add_remote_agent(peer_metadata)
                self.known_peers[peer_name] = peer_metadata
                new_peers_added += 1
            else:
                logger.debug(f"[NIXL] Peer {peer_name} already known, skipping")
        
        logger.info(
            f"[NIXL] update_peers completed: removed {peers_removed} stale peer(s), "
            f"added {new_peers_added} new peer(s), total known peers={len(self.known_peers)}"
        )
    
    def register_memory(self, tensors: list):
        """Register tensors with NIXL."""
        logger.debug(f"[NIXL] register_memory called with {len(tensors)} tensor(s)")
        return self.agent.register_memory(tensors)
    
    def deregister_memory(self, reg_descs):
        """Deregister memory from NIXL."""
        logger.debug(f"[NIXL] deregister_memory called")
        self.agent.deregister_memory(reg_descs)
    
    def sync_metadata(
        self,
        comm_group: StatelessProcessGroup,
        my_rank_in_group: int,
        peer_ranks: list[int]
    ):
        """Sync agent metadata across all ranks. Call before batch_isend_irecv."""
        
        my_metadata = self.agent.get_agent_metadata()
        logger.debug(f"[NIXL] Got my metadata, length={len(my_metadata)} bytes")
        
        metadata_list = comm_group.all_gather_obj(my_metadata)
        logger.debug(f"[NIXL] Received metadata from {len(metadata_list)} peers")
        
        peer_agent_names = {rank: f"{self.group_prefix}_rank_{rank}" for rank in peer_ranks}
        
        for rank_in_group in peer_ranks:
            if rank_in_group == my_rank_in_group:
                continue
            peer_name = peer_agent_names[rank_in_group]
            peer_metadata = metadata_list[rank_in_group]
            logger.debug(f"[NIXL] Loading metadata for {peer_name}")
            self.agent.add_remote_agent(peer_metadata)

    def sync_metadata_p2p(
        self,
        comm_group: StatelessProcessGroup,
        my_rank_in_group: int,
        peer_rank: int,
    ):
        """
        Sync agent metadata with a single peer using point-to-point exchange.
        Unlike sync_metadata which uses all_gather (collective), this only
        exchanges metadata between my_rank and peer_rank.
        """
        peer_name = f"{self.group_prefix}_rank_{peer_rank}"
        
        my_metadata = self.agent.get_agent_metadata()
        logger.debug(f"[NIXL] P2P metadata sync: my_rank={my_rank_in_group}, peer_rank={peer_rank}")
        
        # Both sides send their metadata to each other
        comm_group.send_obj(my_metadata, dst=peer_rank)
        
        # Both sides receive metadata from each other
        peer_metadata = comm_group.recv_obj(src=peer_rank)
        
        # Add/update remote agent with fresh metadata
        logger.debug(f"[NIXL] Adding/updating peer {peer_name} from P2P sync")
        self.agent.add_remote_agent(peer_metadata)
        self.known_peers[peer_name] = peer_metadata

    def batch_isend_irecv(self, p2p_ops):
        """Convert P2POp operations to NIXL isend/irecv calls and block until completion."""
        import torch
        
        logger.debug(f"[NIXL] batch_isend_irecv called with {len(p2p_ops)} operations")
        
        requests = []
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                dst_agent_name = f"{self.group_prefix}_rank_{op.group_peer}"
                req = self.comm_manager.isend(op.tensor, dst_agent=dst_agent_name)
                requests.append(req)
            elif op.op is torch.distributed.irecv:
                src_agent_name = f"{self.group_prefix}_rank_{op.group_peer}"
                req = self.comm_manager.irecv(op.tensor, src_agent=src_agent_name)
                requests.append(req)
            else:
                raise ValueError(f"Unsupported operation: {op.op}")
        
        logger.debug(f"[NIXL] Created {len(requests)} NIXL requests, starting progress loop")
        iteration = 0
        while True:
            statuses = self.comm_manager.batch_progress(requests)
            
            if any(status == NixlRequestStatus.ERROR for status in statuses.values()):
                logger.error("[NIXL] Transfer failed with ERROR status")
                raise RuntimeError("NIXL transfer failed")
            
            all_done = all(status == NixlRequestStatus.DONE for status in statuses.values())
            if all_done:
                logger.debug(f"[NIXL] All {len(requests)} requests completed after {iteration} iterations")
                break
            
            iteration += 1
            if iteration % 100 == 0:
                logger.debug(f"[NIXL] Progress iteration {iteration}, still waiting...")
            
            time.sleep(0.001)

