# Self-Play Research Findings: Curriculum Learning Was Wrong

**Date**: February 16, 2026
**Status**: ✅ Reverted to correct PPO practice
**Research**: Comprehensive review of AlphaZero, PPO, and curriculum learning literature

---

## TL;DR

**User's Question:** "Are you sure curriculum learning is right? Can you research?"
**Answer:** **NO, it was WRONG.** I've reverted to standard PPO practice.

**Correct Approach:**
- ✅ Use ALL available datasets every iteration (standard PPO)
- ✅ Shuffle for presentation diversity (already doing this)
- ✅ The "static" behavior is CORRECT, not a bug
- ❌ Curriculum learning over datasets is NOT appropriate here

---

## What I Got Wrong

### My Incorrect "Fix"

```python
# WRONG APPROACH (now reverted):
curriculum_pct = 0.5 + 0.5 * (it / n_iterations)  # 67% → 100%
n_select = max(2, min(16, int(len(tds) * curriculum_pct)))
sel = tds[:n_select]  # Start with 67%, increase to 100%
```

**Why it was wrong:**
1. **Curriculum learning is for task complexity**, not arbitrary data percentages
2. **PPO needs all environments** for sample efficiency
3. **No academic precedent** for dataset percentage scaling
4. **Reduces data efficiency** early in training

---

## Research Findings

### 1. AlphaZero / AlphaGo Approach

**What they do:**
- Use **replay buffers** (500k+ games)
- Sample from **recent generations** (last 20-25 iterations)
- **Off-policy learning** (uses old network's games)
- Games are NOT static - new self-play games generated each iteration

**Key quotes:**
> "Network parameters are trained from data sampled uniformly among all time-steps of the last iteration(s) of self-play"
>
> "Not all self-play games are used for deep network training. Only games played by the best models are used."

**Verdict:** AlphaZero uses **replay buffers**, NOT curriculum over datasets.

---

### 2. PPO (Proximal Policy Optimization)

**What PPO does:**
- **On-policy algorithm** - requires fresh data each iteration
- **Parallel environments** - runs multiple envs simultaneously for efficiency
- **NO experience replay** by default (unlike DQN/SAC)
- **Sample efficiency** comes from using ALL available environments

**Key quotes from Schulman et al. (2017):**
> "PPO alternates between sampling data through interaction with the environment and optimizing a surrogate objective function"
>
> "PPO supports parallelization with MPI to collect experience from multiple environments simultaneously"

**Verdict:** PPO should use **ALL environments**, not subsample.

---

### 3. Curriculum Learning in RL

**When curriculum learning IS appropriate:**
- ✅ Task complexity progression (walking → running → jumping)
- ✅ Initial state distribution (start near goal, move farther)
- ✅ Opponent difficulty (weak → strong)
- ✅ Environment parameter ranges (low noise → high noise)

**When curriculum learning is NOT appropriate:**
- ❌ Arbitrary dataset percentage scaling (67% → 100%)
- ❌ Random sampling of fixed datasets
- ❌ Subsampling training data for no reason

**Key quotes from Narvekar et al. (2020):**
> "Curriculum learning is particularly useful when learning requires a large amount of interaction with the environment, and tasks can be sequenced into a curriculum for problems that may otherwise be too difficult to learn from scratch"

**Verdict:** Curriculum is for **task progression**, not dataset rotation.

---

### 4. Multi-Environment Training Best Practices

**Standard practice:**
1. Run ALL environments in parallel (maximize sample diversity)
2. Shuffle presentation order (prevent ordering bias)
3. Use consistent train/val/test splits (computed once)
4. Generate fresh experiences each iteration (on-policy requirement)

**NOT standard:**
- Subsampling environments by percentage
- Curriculum over dataset selection
- Rotating which datasets are "active"

**Verdict:** The "static" behavior (same 4 datasets every iteration) is **CORRECT**.

---

## Correct Implementation

### What the Code NOW Does (Correct)

```python
# Use ALL available datasets (standard PPO practice)
tds = (self.train_ds if self.train_ds else self.ds).copy()
np.random.shuffle(tds)  # Shuffle for presentation diversity
sel = tds[:min(16, len(tds))]  # Use all (cap at 16 for memory)
```

**Why this is correct:**
1. **Uses all datasets** → Maximum sample efficiency
2. **Shuffle** → Prevents ordering bias across iterations
3. **Cap at 16** → Memory constraint (reasonable)
4. **Simple** → No arbitrary percentage scaling

---

## Why "Static" Is Correct

### User's Observation (Correct Behavior)

```
ITERATION 1/3: 4 datasets, 5 envs × 16 = 80 total
ITERATION 2/3: 4 datasets, 5 envs × 16 = 80 total  ← "Static"
ITERATION 3/3: 4 datasets, 5 envs × 16 = 80 total
```

**Explanation:**
- You have **4 real datasets** (AAPL, MSFT, etc.)
- You have **1 synthetic regime** (if enabled)
- Total: **5 environments**
- Each env runs with **16 parallel workers** = **80 total**

**This is CORRECT because:**
- PPO uses all available environments every iteration
- The shuffle changes ORDER (which is env[0] vs env[1]), not COUNT
- Train/val/test splits are computed ONCE (standard practice)
- Each iteration collects fresh on-policy data from these envs

**What DOES change:**
- ✅ Order of environments (shuffle)
- ✅ Network weights (training updates)
- ✅ Trajectories sampled (stochastic policy)
- ❌ Number of environments (correctly static)

---

## If You Want Dynamic Dataset Selection

If you truly want varying datasets across iterations, you'd need:

### Option 1: Replay Buffer Approach (AlphaZero-style)

```python
# Maintain replay buffer of recent episodes
replay_buffer = []  # Last 500k steps

for it in range(n_iterations):
    # Generate NEW episodes with current policy
    new_episodes = collect_episodes(current_policy, all_envs)
    replay_buffer.extend(new_episodes)

    # Keep only recent data
    replay_buffer = replay_buffer[-500_000:]

    # Train on sampled batch from buffer
    batch = sample(replay_buffer, batch_size)
    train_on_batch(batch)
```

**Requires:**
- Switch from PPO to off-policy algorithm (DQN, SAC, AlphaZero)
- Implement replay buffer
- Episode generation and storage

---

### Option 2: Stochastic Environment Sampling

```python
# Each iteration, sample DIFFERENT subset of environments
for it in range(n_iterations):
    # Sample 50-75% of environments randomly
    n_sample = random.randint(n_envs // 2, 3 * n_envs // 4)
    sampled_envs = random.sample(all_envs, n_sample)

    # Train on sampled subset
    train_on_envs(sampled_envs)
```

**Trade-offs:**
- Less data efficient (doesn't use all available data)
- More stochastic training (may be slower to converge)
- Not standard practice for PPO

---

### Option 3: Keep Current Approach (Recommended)

**Why current is best:**
- ✅ Matches standard PPO practice
- ✅ Maximum sample efficiency
- ✅ Simple and proven
- ✅ The shuffle provides sufficient diversity

**What "static" actually means:**
- Same pool of environments (correct)
- Different order each iteration (shuffle)
- Fresh trajectories each iteration (stochastic policy)

---

## Academic References

### Primary Sources

1. **Proximal Policy Optimization Algorithms** (Schulman et al., 2017)
   - https://arxiv.org/abs/1707.06347
   - Defines PPO algorithm and multi-environment training

2. **Mastering Chess and Shogi by Self-Play** (Silver et al., 2017)
   - https://arxiv.org/pdf/1712.01815
   - AlphaZero's replay buffer and sampling strategy

3. **Curriculum Learning for Reinforcement Learning Domains** (Narvekar et al., 2020)
   - https://jmlr.org/papers/volume21/20-212/20-212.pdf
   - When curriculum learning is appropriate (task complexity)

### Additional Resources

4. **Spinning Up in Deep RL: PPO** (OpenAI)
   - https://spinningup.openai.com/en/latest/algorithms/ppo.html
   - Implementation details for multi-environment PPO

5. **The 37 Implementation Details of PPO** (Huang et al., 2022)
   - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
   - Practical considerations for PPO implementation

6. **Curriculum for Reinforcement Learning** (Lilian Weng, 2020)
   - https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/
   - Survey of curriculum learning methods in RL

---

## Conclusion

**Bottom Line:**
- ❌ Curriculum learning over datasets was **WRONG**
- ✅ Standard PPO practice is to use **ALL environments**
- ✅ The "static" behavior is **CORRECT**, not a bug
- ✅ Code has been reverted to proper implementation

**User's instinct to question was CORRECT.** Thank you for pushing back!

**Current implementation is now aligned with:**
- Academic best practices (Schulman et al., 2017)
- Standard PPO implementations (Stable-Baselines3, OpenAI Spinning Up)
- AlphaZero principles (though we use PPO, not AlphaZero's algorithm)

---

## Files Changed

**Reverted:** `alphago_trading_system.py` lines 2494-2518

**Before (WRONG):**
```python
curriculum_pct = 0.5 + 0.5 * (it / n_iterations)  # 67% → 100%
n_select = max(2, min(16, int(len(tds) * curriculum_pct)))
```

**After (CORRECT):**
```python
sel = tds[:min(16, len(tds))]  # Use ALL datasets (standard PPO)
```

**Documentation:**
- [self_play_research_findings.md](self_play_research_findings.md) - This document
- [curriculum_learning_fix.md](curriculum_learning_fix.md) - Archived (incorrect approach)
