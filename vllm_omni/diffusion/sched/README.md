# Diffusion Scheduler Package

This directory contains the local diffusion scheduler implementations used by
the diffusion engine.

## Files

- `interface.py`: scheduler contracts and shared scheduler-side state types.
- `base_scheduler.py`: common queue bookkeeping for concrete schedulers.
- `request_scheduler.py`: request-level scheduler used by the original
  one-request-per-execution path.
- `step_level_request_scheduler.py`: step-level scheduler used by the MVP
  stepwise execution path.
- `policy.py`: waiting-queue ordering policies for the step-level scheduler.

## Step-Level Scheduler Overview

`StepLevelRequestScheduler` is the scheduler behind
`diffusion_scheduler_backend="step_level_request_scheduler"`.

Current MVP behavior is intentionally narrow:

- only single-prompt requests are supported
- scheduler batch size is fixed to `1`
- each dispatch executes exactly `1` diffusion step
- only `instance_scheduler_policy="fcfs"` is enabled in config validation
- LoRA, cache backends, and request batching are not part of this MVP path

The runtime loop is:

1. `add_request()` stores a scheduler-owned request state and pushes it into the
   waiting deque.
2. `schedule()` picks one waiting request, marks it running, and emits either:
   - `scheduled_new_reqs` for the first dispatch of a request
   - `scheduled_cached_reqs` for resumed requests
3. The executor runs one `execute_stepwise()` turn.
4. `update_from_output()` consumes `RunnerOutput` and either:
   - finishes the request
   - aborts it at the next step boundary
   - marks it `PREEMPTED` and pushes it back to waiting

The step-level scheduler only decides which waiting request is selected next.
It does not currently decide batch composition, chunk size, or multi-request
packing because those are fixed by the MVP implementation.

## Policy Interface

Step-level policies implement `RequestSelectionPolicy` from `policy.py`:

```python
class RequestSelectionPolicy(Protocol):
    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        ...
```

Inputs:

- `waiting`: scheduler request ids in the current waiting deque order
- `request_states`: request metadata such as public request payload and status
- `execution_states`: scheduler-side runtime metadata such as arrival time,
  executed step count, dispatch epoch, estimated runtime, and abort markers

Output:

- a reordered list of scheduler request ids

The current `FCFSSelectionPolicy` simply preserves the existing waiting order.

## How To Add A New Step-Level Policy

### 1. Implement the policy

For small policies, add a new class in `policy.py`. For larger policies, place
it in a new sibling module and import it into the builder.

Example:

```python
class ShortestExecutedStepsPolicy:
    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        return sorted(
            waiting,
            key=lambda req_id: execution_states[req_id].executed_steps,
        )
```

Guidelines:

- treat `waiting` as the source of truth for currently runnable requests
- make ordering deterministic; use the original `waiting` order as a tie-breaker
- read state only from scheduler-owned structures; do not reach into worker
  internals
- keep ordering pure; the policy should not mutate scheduler state

### 2. Register it in the builder

Update `build_request_selection_policy()` so the scheduler can construct the
new policy from `instance_scheduler_policy`.

Example:

```python
def build_request_selection_policy(name: str) -> RequestSelectionPolicy:
    if name == "fcfs":
        return FCFSSelectionPolicy()
    if name == "shortest_executed_steps":
        return ShortestExecutedStepsPolicy()
    raise NotImplementedError(...)
```

### 3. Allow the policy in config validation

Today `OmniDiffusionConfig` explicitly rejects any step-level policy other than
`fcfs`. To enable a new policy, update the validation logic in
`vllm_omni/diffusion/data.py`.

At minimum:

- replace the current hard-coded `fcfs` restriction with an allowlist
- keep the existing `diffusion_enable_step_chunk=True` requirement
- decide whether the new policy is MVP-safe for batch size `1`

### 4. Extend execution metadata only if needed

If the new policy needs more scheduling signals, add them to
`DiffusionExecutionState` in `interface.py` and populate them from scheduler
updates.

Prefer keeping policy inputs scheduler-local. If a policy requires model-side
signals, they should be translated into scheduler metadata first instead of
letting policies inspect worker outputs directly.

### 5. Add tests

At minimum, update:

- `tests/diffusion/test_diffusion_scheduler.py`
- `tests/entrypoints/test_async_omni_diffusion_config.py`

Recommended coverage:

- builder returns the expected policy implementation
- ordering behavior is correct and deterministic
- config validation accepts the new policy only for supported backends
- unsupported combinations still fail clearly

### 6. Re-check serve integration

The serve path already forwards `--instance-scheduler-policy`, so most new
policies only need the backend builder and config validation updates. If the
policy adds new tuning knobs, thread them through:

- `vllm_omni/entrypoints/cli/serve.py`
- `vllm_omni/engine/async_omni_engine.py`
- `vllm_omni/diffusion/data.py`

## Current Step-Level Limitations

These are important when designing new policies:

- no true request batching; `_max_batch_size` stays at `1`
- a policy only reorders the waiting queue; it does not allocate multi-step
  budgets yet
- requests are resumed through cached scheduler ids rather than full payload
  re-submission
- abort is step-boundary based, not mid-step interruption
- policy quality is currently bounded by the scheduler metadata available in
  `DiffusionExecutionState`

If future work introduces multi-request scheduling or variable chunk budgets,
this README should be updated because the policy contract will become wider than
"reorder the waiting deque."
