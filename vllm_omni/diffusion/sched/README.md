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
- config validation currently allows:
  - `fcfs`
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `p95-first`
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

The base class now also exposes optional lifecycle hooks:

- `initialize()`
- `on_request_arrival()`
- `on_request_scheduled()`
- `on_step_complete()`
- `on_request_finished()`

These hooks are how migrated policies keep small amounts of scheduler-local
learning state without pushing that logic back into `StepLevelRequestScheduler`.

## Migrated Policies

### `fcfs`

Preserves the waiting deque order exactly. This remains the safest baseline and
the default config value.

### `sjf`

Orders waiting requests by estimated remaining runtime.

Runtime estimation uses this priority order:

1. request `sampling_params.extra_args["estimated_cost_s"]`
2. `instance_runtime_profile_path` + `instance_runtime_profile_name`
3. scheduler-local heuristic fallback

For resumed requests, remaining cost is scaled by
`(total_steps - executed_steps) / total_steps`.

### `sjf_aging`

Starts from the same remaining-cost estimate as `sjf`, then applies the
`v16-base` aged-cost ranking:

- older requests get an aging discount
- larger requests get a bounded cost-aware aging weight

This keeps the basic SJF throughput bias while reducing starvation.

### `sjf_aging_guarded`

Builds on `sjf_aging` and adds a protected queue for old requests.

A request becomes protected when its wait time exceeds:

- the learned wait guard from completed-request history, or
- `2.0 * estimated_remaining_cost_s`

Protected requests are served ahead of normal requests and ordered by arrival
time within the protected group.

Alias:

- `sjf_aging_guard` is accepted and normalized to `sjf_aging_guarded`

### `sjf_aging_guarded_tail`

Builds on `sjf_aging_guarded` and adds the `v16-base` tail-sink idea:

- only protected, super-heavy requests are eligible
- only a strict 5% global/sliding-window defer budget is available
- at most one request is sunk per reorder pass
- sunk requests stay at the tail across requeues until hard escape releases them

Important limitation in the current `v18-base` landing:

- the migrated policy only controls waiting-queue ordering
- it does not yet migrate the old chunk-budget overrides such as idle-only `3x`
  chunk expansion, because the current step-level backend still dispatches one
  step per turn

### `p95-first`

Implements the normalized tail-pressure ordering path from `v16-base`:

- learns observed service time from completed step runtime
- learns slowdown from end-to-end request latency over cumulative execute time
- greedily orders the waiting queue by normalized predicted tail pressure

The current landing intentionally keeps the algorithm narrow:

- it uses the learned service-rate / slowdown path
- it does not expose the old large CLI tuning surface (`base_ms`, `max_ms`,
  backlog alpha, bucket knobs, fusion knobs, etc.)

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
- prefer keeping policy-owned learning state inside the policy object, driven by
  lifecycle hooks, instead of bloating the scheduler core

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

Today `OmniDiffusionConfig` validates step-level policies against an explicit
allowlist. To enable a new policy, update the validation logic in
`vllm_omni/diffusion/data.py`.

At minimum:

- extend the existing allowlist
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
- the migrated `sjf_aging_guarded_tail` policy does not yet port the old
  chunk-budget override behavior
- requests are resumed through cached scheduler ids rather than full payload
  re-submission
- abort is step-boundary based, not mid-step interruption
- policy quality is currently bounded by the scheduler metadata available in
  `DiffusionExecutionState`

If future work introduces multi-request scheduling or variable chunk budgets,
this README should be updated because the policy contract will become wider than
"reorder the waiting deque."
