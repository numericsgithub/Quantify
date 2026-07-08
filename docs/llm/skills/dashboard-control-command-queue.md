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

## Safe boundaries (V1 `Trainer`)

- **`"step"`** — end of a training step, after `self._global_step += 1`. Used
  for LR / hyperparameter changes so they take effect within one step. The
  per-step drain is a single `queue.get_nowait()` that raises `Empty` and
  returns immediately when nothing is queued — no lock contention on the hot
  path.
- **`"epoch"`** — between epochs. Used for structural mutations: reloading
  model weights, extending the epoch budget, toggling callbacks. Anything
  that must not happen while batches are in flight goes here.

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

## Precedent

`training_harness/console.py` (`TrainingConsole`) is the same pattern driven
by stdin instead of HTTP, for V2: a background reader thread enqueues, the
main loop drains between epochs. The dashboard `ControlManager` mirrors it.
