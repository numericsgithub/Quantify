# Skill: Live-control command queue (dashboard write endpoints)

**When to use:** you need to mutate a *running* training process from outside
the training thread — an HTTP request, an LLM agent, any code on a different
thread than the training loop. Applies to the dashboard control endpoints in
`training_harness/api/`, but the pattern generalizes.

## The rule

**Never mutate the trainer's shared objects (optimizer, scheduler, model,
callbacks) from the API/background thread.** The training loop reads and
writes those on the main thread mid-step; a concurrent write from another
thread is a race (e.g. the loop reading LR while you overwrite it between
`optimizer.step()` and `scheduler.step()`).

Instead:

1. **Validate + enqueue on the API thread** (`ControlManager.submit`). Bad
   input raises `ControlValidationError` → the route returns HTTP 400. Valid
   input is recorded (`status="pending"`) and pushed onto a thread-safe
   `queue.Queue`. The route returns **202 Accepted** with a command id.
2. **Drain + apply on the training thread** at a safe boundary
   (`ControlManager.drain(boundary)`). The mutation happens here, on the
   thread that owns the object, where nothing is in flight. The command is
   marked `applied` (with a result string) or `failed`.

The client polls `GET /api/v1/commands/<id>` and reflects
`pending → applied/failed` — it must **never** assume instant success.

## Safe boundaries

Both trainers drain the SAME `ControlManager` at two boundaries:

- **`"step"`** — end of a training step, after `self._global_step += 1`. Used
  for LR / hyperparameter changes so they take effect within one step. The
  per-step drain is a single `queue.get_nowait()` that raises `Empty` and
  returns immediately when nothing is queued — no lock contention on the hot
  path. V1: `trainer.py` after the optimizer step. V2: `trainer_v2.py`
  `_run_epoch`, right after `self._global_step += 1`.
- **`"epoch"`** — between epochs. Used for structural mutations: reloading
  model weights, extending the epoch budget. Anything that must not happen
  while batches are in flight goes here. V2 drains it right after
  `console.drain(epoch)`.

## V2 is the control target (migrated from V1)

`QATTrainerV2` always constructs a `ControlManager` (even when `api_port` is
unset), so the **HTTP dashboard and the interactive `TrainingConsole` feed one
queue with one apply path** — the console's `lr` / `load-best` verbs call
`control.submit(...)` instead of mutating the trainer directly. Differences from
V1 the migration had to handle:

- **Two schedulers, not one.** V2 can run a per-step `self.scheduler` AND an
  epoch-stepped `ReduceLROnPlateau` (`_plateau_lr_sched`). A manual-LR override
  sets `_scheduler_suspended`, which guards BOTH `step()` calls. `/status`
  surfaces `scheduler_suspended`.
- **`add-epochs` needs a re-read bound.** V2's loop was `for epoch in
  range(start_epoch, end_epoch)` with a local bound — mutating it does nothing
  (same `range()` gotcha as V1). It was converted to
  `while epoch < self._end_epoch`, and `_apply_add_epochs` extends
  `self._end_epoch`.
- **Callback enable/disable now works on V2.** V2 builds a `CallbackRegistry`
  (`_build_callback_registry`) naming its toggleable loop behaviors —
  checkpointing, scale_factor_tracking, plateau_qat_activation, early_stopping,
  breakdown_recovery, user hooks — each guarded inline by `is_enabled()`
  (default all-enabled = identical behavior). Core behaviors (`optimizer_step`,
  `metrics_logging`) are listed but reject toggles. Note: disabling
  `plateau_qat_activation` only removes the *early* plateau-triggered QAT start;
  QAT still activates at `float_warmup_epochs`. Live `ReduceLROnPlateau`
  patience/factor/min_lr editing is a separate command (`set_scheduler_params`,
  epoch boundary). Still deferred: adding *new* callback logic at runtime and
  swapping the LR-scheduler type (Plateau ↔ Cosine).

## Gotchas this pattern already handles

- **Scheduler vs. manual LR.** A step LR scheduler overwrites the LR every
  step, so a manual override needs `Trainer._scheduler_suspended` (guards the
  `scheduler.step()` call). Setting LR suspends it; `suspend_scheduler=false`
  resumes. Surface `scheduler_suspended` in `/status`.
- **`add-epochs` needs a `while` loop, not `for … range()`.** `range()` is
  evaluated once, so mutating `config.epochs` mid-loop does nothing with a
  `for`. The loop is `while epoch < self.config.epochs:`. Also refresh
  `EpochTimer.total_epochs`.
- **Apply-time failures still happen despite submit-time validation.** e.g.
  reload-best when no checkpoint exists yet — that's a legitimate `failed`,
  not a 400. Keep the failed status.
- **Callbacks are not a real framework here.** `CallbackRegistry` names the
  existing hardcoded loop behaviors and adds one-line `is_enabled()` guards;
  it does not move logic out of the loop. Core behaviors (`optimizer_step`,
  `metrics_logging`) are registered `toggleable=False` and reject toggles.
- **JSONL is written from two threads now** (steps/epochs on the training
  thread, submitted-command events on the API thread) — guard the file write
  with a lock.

## Pause/resume: the two sanctioned off-queue mutations

`pause` and `resume` do **not** go through submit/drain. The training loop has a
**pause gate** — `threading.Event.wait()` at the step boundary (after
`drain("step")`, never mid-backward). `pause()` clears the Event, `resume()`
sets it; both are called directly from the API thread. Only the *blocking* is on
the training thread (at the gate); toggling the Event is a safe cross-thread
signal — that's what `threading.Event` is for.

**Why pause must be direct too (not queued):** if pause were a queued command, a
`resume` could arrive and run *before* the still-queued pause applied; the stale
pause would then fire and re-pause the run. Direct set/clear has no such race.
This was found in live testing and is guarded by a test
(`test_pause_is_not_a_queued_command`).

The still-queued commands are `set_hyperparams` (step), `end_epoch_early` (step),
`add_epochs` / `reload_best` / `halt` (epoch). `halt` and `reload_best` require
`{"confirm": true}`.

## Reload-by-criterion needs a second checkpoint pool

`CheckpointManager` ranks by ONE metric, so to reload "best val_acc" AND "best
train_loss" you keep a **secondary pool** per extra metric
(`TrainerConfigV2.secondary_checkpoint_metrics`, e.g. `[("train_loss","min")]`).
Each secondary pool keeps only its single best file (`top_k=1`, `save_last=False`
→ no last.pt / ONNX) under `checkpoints/by_<metric>/`. The trainer exposes
`_checkpoint_pools` (metric → manager); `reload_best`'s `criterion`
(`"best"` or `"best_<metric>"`) resolves against it, failing loudly if no pool
tracks that metric.

## QAT group controls (Phase 4)

Built on `QuantizerManager.select_quantizers(group)` (group ∈ `weights /
biases / activations / all`). The manager gained group mutators —
`set_group_annealing_alpha` / `_step` / `_ramp`, `disable_group`,
`recalibrate_group`, `set_lsb(quant_id, lsb)`, `describe_quantizers()` — each
returning an accounting dict that **always includes `unknown_role`**, so a
group op that couldn't classify a quantizer says so loudly in the command
result rather than silently skipping it. `ControlManager` wraps them:
`set_annealing` / `set_lsb` (step), `recalibrate` / `disable_quant` (epoch,
`confirm` required).

Two decisions baked in:

- **Annealing modes are NOT conflated** (the semantics trap): `ramp` (α 0→1
  over n), `absolute` (α:=X now, step untouched), `step` (per-forward increment
  only) are separate `mode`s on one endpoint.
- **Disable uses the annealing-α mechanism** (α=0, step=0), NOT Brevitas's
  per-module `disable_quant`. Reason: α is addressable per *registered*
  quantizer by role and is reversible; the Brevitas toggle is per-proxy and
  doesn't map to role groups. Do not mix the two across groups — that leaves the
  model in an incoherent quantization state.

Recalibration is **lazy**: `recalibrate_group` clears `search_done`, so each
quantizer re-runs its LSB search on its next forward, against whatever batch
comes next — surface that, it's a real behavioral consequence. (Uniform
`search_done`-clear for both global and per-group; more robust than the
`force_recalibration` flag, which resets after the first quantizer calibrates.)

## Console as a front-end (not a second path)

`training_harness/console.py` (`TrainingConsole`) is now a **stdin front-end
over the same `ControlManager`**: its `lr` and `load-best` verbs call
`control.submit(...)`, so there is one apply path and one audit log regardless
of input source. It still handles a few local, read-or-"stays-as-is" verbs
directly (`status`, `stop`, and the in-place `ReduceLROnPlateau` `patience` /
`factor` edits, which are deliberately left as-is and out of scope for the
migration).
