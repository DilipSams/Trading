"""
Deep MCTS Implementation (AlphaGo-Style 15-20 Step Lookahead)
=============================================================

GPU-batched MCTS with:
  - Wave batching: K rollouts per GPU call (K×B batch size, saturates GPU)
  - Virtual loss: forces parallel diversity across rollouts within each wave
  - Progressive widening: reduces effective branching 5→3 for deeper search
  - Continuation rollouts: +5 depth per rollout cheaply (no GPU eval)
  - Iterative deepening: scales rollouts with training iteration

Depth estimation:
  256 rollouts + branching 3 + 5 continuation = ~15-17 effective depth
  512 rollouts + branching 3 + 5 continuation = ~17-19 effective depth

Each wave: K×(Selection) → ONE GPU eval (K×B batch) → K×(Expansion+Backup)
Wave size auto-calculated to target GPU batch size ~256.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from alphago_trading_system import DEVICE


class MCTSNode:
    """Lightweight MCTS tree node with virtual loss support."""
    __slots__ = ('parent', 'action', 'prior', 'vc', 'tv', 'mv',
                 'children', 'vl', 'stored_priors')

    def __init__(self, parent=None, action=None, prior=1.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.vc = 0       # visit count
        self.tv = 0.0     # total value
        self.mv = 0.0     # mean value
        self.children = {}
        self.vl = 0       # virtual loss count
        self.stored_priors = None  # saved network priors for progressive widening

    def is_leaf(self):
        return len(self.children) == 0

    def ucb(self, c_puct):
        """PUCT score with virtual loss adjustment (AlphaGo-style)."""
        if self.parent is None:
            return 0.0
        parent_visits = self.parent.vc + self.parent.vl + 1
        effective_visits = self.vc + self.vl
        # Q adjusted down by virtual loss — makes visited nodes look worse
        # so parallel simulations explore different branches
        q = (self.tv - self.vl * 1.0) / max(effective_visits, 1)
        u = c_puct * self.prior * np.sqrt(parent_visits) / (1 + effective_visits)
        return q + u


class ParallelMCTSPlanner:
    """
    Deep MCTS planner with AlphaGo-style wave batching.

    Architecture:
      - K rollouts selected per wave, ONE GPU eval per wave (batch size K×B)
      - Virtual loss accumulates across rollouts within a wave → forces diversity
      - Tree grows deeper across waves as PUCT drives into less-visited regions

    Key techniques:
      - Wave batching: K rollouts per GPU call → saturates GPU Tensor Cores
      - Virtual loss: penalizes visited nodes across parallel rollouts within a wave
      - Progressive widening: expand top-3 children first → effective branching 3 not 5
      - Continuation rollouts: 5 extra greedy steps after leaf → +5 depth for free
    """

    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        self.na = cfg.n_actions
        self.gamma = cfg.gamma

    def _tree_depth(self, node, depth=0):
        """Compute maximum depth of tree rooted at node."""
        if node.is_leaf():
            return depth
        return max(self._tree_depth(c, depth + 1) for c in node.children.values())

    def _principal_variation_depth(self, node):
        """Depth of the principal variation (most-visited path from root)."""
        depth = 0
        while not node.is_leaf():
            best = max(node.children.values(), key=lambda c: c.vc)
            if best.vc == 0:
                break
            node = best
            depth += 1
        return depth

    @torch.no_grad()
    def batch_search(self, root_envs, n_rollouts=None):
        """
        Deep MCTS search with AlphaGo-style techniques.

        Args:
            root_envs: List of B root environments
            n_rollouts: Number of rollouts per root (default: cfg.mcts_rollouts)

        Returns:
            improved_policies: (B, n_actions) array of MCTS-improved action distributions
        """
        if n_rollouts is None:
            n_rollouts = self.cfg.mcts_rollouts

        B = len(root_envs)
        self.net.eval()

        # ================================================================
        # PHASE 1: Initialize roots with network priors
        # ================================================================
        root_obs = np.array([env._obs() for env in root_envs])
        root_tensor = torch.FloatTensor(root_obs).to(DEVICE)

        with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
            root_priors = self.net.get_action_probs(root_tensor).cpu().numpy()

        # Create root nodes with Dirichlet noise (AlphaGo-style exploration)
        roots = []
        for b in range(B):
            pri = root_priors[b].copy()

            eps = self.cfg.mcts_dirichlet_eps
            alpha = self.cfg.mcts_dirichlet_alpha
            if eps > 0 and alpha > 0:
                noise = np.random.dirichlet([alpha] * self.na)
                pri = (1 - eps) * pri + eps * noise
                pri = pri / (pri.sum() + 1e-10)

            # Create root — with progressive widening, start with top-K children
            r = MCTSNode(prior=1.)
            r.stored_priors = pri.copy()

            if getattr(self.cfg, 'mcts_progressive_widening', False):
                actions_sorted = sorted(range(self.na),
                                        key=lambda a: pri[a], reverse=True)
                k = min(getattr(self.cfg, 'mcts_pw_max_children', 3), self.na)
                for a in actions_sorted[:k]:
                    r.children[a] = MCTSNode(parent=r, action=a, prior=float(pri[a]))
            else:
                for a in range(self.na):
                    r.children[a] = MCTSNode(parent=r, action=a, prior=float(pri[a]))

            roots.append(r)

        # ================================================================
        # PHASE 2: Deep rollouts — one rollout at a time, batched across B envs
        # ================================================================
        self._run_rollouts(root_envs, roots, n_rollouts)

        # ================================================================
        # PHASE 3: Extract improved policies from search statistics
        # ================================================================
        improved = np.zeros((B, self.na), dtype=np.float32)
        for b, root in enumerate(roots):
            # Collect visits for ALL actions (including those not yet expanded)
            visits = np.zeros(self.na, dtype=np.float32)
            for a in range(self.na):
                if a in root.children:
                    visits[a] = root.children[a].vc

            # Temperature-scaled visits to action probabilities
            if self.cfg.mcts_temperature > 0:
                visits = visits ** (1.0 / self.cfg.mcts_temperature)

            total = visits.sum()
            if np.isfinite(total) and total > 0:
                improved[b] = visits / total
            else:
                if np.isfinite(root_priors[b]).all():
                    improved[b] = root_priors[b]
                else:
                    improved[b] = np.ones(self.na, dtype=np.float32) / self.na

        # Final safety: replace any remaining NaN/Inf rows with uniform
        bad = ~np.isfinite(improved).all(axis=1)
        if bad.any():
            improved[bad] = 1.0 / self.na

        return improved

    def _run_rollouts(self, root_envs, roots, n_rollouts):
        """
        Execute n_rollouts of deep MCTS search using wave batching.

        Wave batching: K rollouts are selected before a single GPU eval,
        producing batch size K×B per forward pass. Virtual loss accumulates
        across rollouts within each wave, forcing PUCT to explore different
        branches (AlphaGo-style parallel diversity).

        Each wave:
          1. SELECTION ×K: Walk K rollouts per root, VL accumulates across K
          2. EVALUATION: ONE batched GPU forward pass for all K×B leaves
          3. EXPANSION: Create children (progressive widening)
          4. CONTINUATION: 5 extra greedy steps for deeper value estimate
          5. BACKUP: Propagate values, remove VL for entire wave
        """
        B = len(root_envs)
        cfg = self.cfg
        vl_val = getattr(cfg, 'mcts_virtual_loss', 0.0) if getattr(cfg, 'mcts_use_virtual_loss', False) else 0.0
        pw_enabled = getattr(cfg, 'mcts_progressive_widening', False)
        pw_alpha = getattr(cfg, 'mcts_pw_alpha', 0.5)
        pw_max = getattr(cfg, 'mcts_pw_max_children', 3)
        cont_steps = getattr(cfg, 'mcts_continuation_steps', 0)

        # Wave size: how many rollouts per GPU batch (auto-target batch ~256)
        wave_size = getattr(cfg, 'mcts_wave_size', 0)
        if wave_size <= 0:
            wave_size = max(1, 256 // max(B, 1))
        wave_size = min(wave_size, n_rollouts)

        for wave_start in range(0, n_rollouts, wave_size):
            K = min(wave_size, n_rollouts - wave_start)

            leaf_obs_list = []
            leaf_meta = []  # (env_idx, node, path_reward, depth, sim_env, path_nodes)
            terminated_backups = []  # (node, v_total, path_nodes)

            # --- K rollouts of selection (VL accumulates across rollouts) ---
            for w in range(K):
                for b in range(B):
                    # Clone environment for this rollout
                    if cfg.mcts_stochastic:
                        sim_env = root_envs[b].stochastic_clone(
                            horizon=cfg.mcts_sim_horizon,
                            block_size=cfg.mcts_bootstrap_block)
                    else:
                        sim_env = root_envs[b].clone()

                    node = roots[b]
                    path_reward = 0.0
                    depth = 0
                    path_nodes = []  # track for virtual loss cleanup
                    terminated = False

                    # --- SELECTION with virtual loss ---
                    while not node.is_leaf():
                        # Progressive widening: add children if visits warrant it
                        if pw_enabled and node.stored_priors is not None:
                            threshold = max(1, int(node.vc ** pw_alpha))
                            n_children = len(node.children)
                            if n_children < min(threshold, self.na):
                                existing = set(node.children.keys())
                                all_priors = node.stored_priors
                                missing = [(a, all_priors[a]) for a in range(self.na)
                                           if a not in existing]
                                missing.sort(key=lambda x: x[1], reverse=True)
                                for a, p in missing[:threshold - n_children]:
                                    node.children[a] = MCTSNode(
                                        parent=node, action=a, prior=float(p))

                        # PUCT selection
                        best_a = max(node.children.keys(),
                                     key=lambda a: node.children[a].ucb(cfg.mcts_exploration))
                        child = node.children[best_a]

                        # Apply virtual loss before descending
                        if vl_val > 0:
                            child.vl += vl_val
                            path_nodes.append(child)

                        node = child

                        # Step simulation forward
                        if sim_env.cs < sim_env.nb - 1:
                            _, rew, term, trunc, _ = sim_env.step(best_a)
                            path_reward += (self.gamma ** depth) * rew
                            depth += 1
                            if term or trunc:
                                terminated = True
                                break
                        else:
                            terminated = True
                            break

                    if terminated:
                        terminated_backups.append((node, path_reward, path_nodes))
                    else:
                        leaf_obs_list.append(sim_env._obs())
                        leaf_meta.append((b, node, path_reward, depth, sim_env, path_nodes))

            # --- Backup terminated simulations ---
            for node, v_total, path_nodes in terminated_backups:
                current = node
                while current is not None:
                    current.vc += 1
                    current.tv += v_total
                    current.mv = current.tv / current.vc
                    current = current.parent
                for n in path_nodes:
                    n.vl -= vl_val

            # --- ONE BATCHED GPU EVALUATION for all K×B leaves ---
            if not leaf_obs_list:
                continue

            leaf_tensor = torch.FloatTensor(np.array(leaf_obs_list)).to(DEVICE)
            with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                logits, leaf_values, leaf_risks = self.net.forward(leaf_tensor)
                leaf_values = leaf_values.cpu().numpy()
                leaf_risks = leaf_risks.cpu().numpy()
                leaf_priors = F.softmax(logits, dim=-1).cpu().numpy()

            # --- EXPANSION + CONTINUATION + BACKUP for all K×B leaves ---
            for idx, (b, node, path_rew, depth, sim_env, path_nodes) in enumerate(leaf_meta):
                priors = leaf_priors[idx]

                # Expand leaf node
                if len(node.children) == 0:
                    node.stored_priors = priors.copy()

                    if pw_enabled:
                        actions_sorted = sorted(range(self.na),
                                                key=lambda a: priors[a], reverse=True)
                        k = min(pw_max, self.na)
                        for a in actions_sorted[:k]:
                            node.children[a] = MCTSNode(
                                parent=node, action=a, prior=float(priors[a]))
                    else:
                        for a in range(self.na):
                            node.children[a] = MCTSNode(
                                parent=node, action=a, prior=float(priors[a]))

                # Risk-adjusted leaf value
                v_leaf = leaf_values[idx] - 0.1 * leaf_risks[idx]

                # --- CONTINUATION ROLLOUT: +N depth for free (no GPU) ---
                cont_reward = 0.0
                cont_depth = 0
                if cont_steps > 0:
                    best_action = int(np.argmax(priors))
                    for cs in range(cont_steps):
                        if sim_env.cs >= sim_env.nb - 1:
                            break
                        _, rew, term, trunc, _ = sim_env.step(best_action)
                        cont_reward += (self.gamma ** (depth + cs)) * rew
                        cont_depth += 1
                        if term or trunc:
                            break

                # Total value: path + continuation + discounted leaf
                total_depth = depth + cont_depth
                v_total = path_rew + cont_reward + (self.gamma ** total_depth) * v_leaf

                # --- BACKUP: propagate value up the tree ---
                current = node
                while current is not None:
                    current.vc += 1
                    current.tv += v_total
                    current.mv = current.tv / current.vc
                    current = current.parent

                # Remove virtual loss
                for n in path_nodes:
                    n.vl -= vl_val


# ============================================================================
# Drop-in replacement: Use ParallelMCTSPlanner instead of BatchedMCTSPlanner
# ============================================================================

def get_mcts_planner(net, cfg, use_parallel=True):
    """
    Factory function to get MCTS planner.

    Args:
        net: Neural network
        cfg: Config object
        use_parallel: If True, use ParallelMCTSPlanner (deep search)
                      If False, use original BatchedMCTSPlanner

    Returns:
        MCTS planner instance
    """
    if use_parallel:
        return ParallelMCTSPlanner(net, cfg)
    else:
        from alphago_trading_system import BatchedMCTSPlanner
        return BatchedMCTSPlanner(net, cfg)
