# Importance Analysis (`importance/`)

A model- and quantizer-agnostic tool that computes, for every Conv1d/2d/3d
and Linear layer (including Brevitas/Quantify quantized variants), how
important each weight/kernel/filter is for each output feature, averaged
over a dataset, and an interactive local web viewer to explore the result.

```python
from importance import analyze, view

result = analyze(model, dataloader)
result.save("runs/imp_mymodel")
view("runs/imp_mymodel")
```

See `new_feature.md` at the repo root for the full original spec. This
document covers the metrics, the BatchNorm caveat, and the on-disk format.

## Metrics

For output feature `j`, sample `n`, parameter `w`:

```
s = w * d(out_j) / d(w)          # first-order Taylor attribution
```

Two metrics are stored at every level (weight / kernel / filter / layer):

- **`mean_abs_s`** (main metric): `mean_n |s|` -- typical importance.
- **`mean_s`** (secondary metric): `mean_n s` -- signed net effect. Cheap:
  it equals `w * mean_n(d out_j / d w)`, computed with a single batched
  `backward()` call per output feature (no per-sample loop needed -- see
  "How it's computed" below).

Aggregation levels sum the *finest computed level*'s per-element scores
upward (not re-differentiated per level):

- `weight`: `[n_out, F, C, *k]`
- `kernel` = weight summed over spatial kernel positions: `[n_out, F, C]`
- `filter` = weight summed over channels + kernel positions (+ bias):
  `[n_out, F]`
- `layer` = filter summed over `F`: `[n_out]`

`n_out = 1 + K (+ 1 if loss_fn given)`: index 0 is always an extra
`"__all__"` row (the mean over the K real output features, computed by
just appending a `mean(out, dim=1)` column to the output tensor before
running the same pipeline -- no special-casing needed), indices `1..K` are
the real output features, and the optional last index is `"loss"`.

### The BatchNorm problem, and why `act_filter` exists

If a conv is immediately followed by BatchNorm, the network's output is
*exactly* invariant to rescaling that conv filter's weights by any positive
constant `c` (BN normalizes by the batch/running standard deviation of that
same filter's output, which scales by `c` too). By Euler's homogeneous
function theorem, a function invariant to `w -> c*w` has `w . grad_w(f) ==
0` at `c=1` -- so the *signed* per-weight scores of a BN-following filter,
summed over the filter, cancel almost exactly. (With `BatchNorm2d`'s
default `track_running_stats=True` and the model in `eval()`, this holds
only approximately, since eval-mode BN normalizes by *fixed* running
statistics rather than the current batch's; `track_running_stats=False`
makes it exact even in eval mode -- see
`tests/test_importance_core.py::test_conv_bn_weight_score_fails_but_act_filter_does_not`
for a constructive proof.)

This means weight-level (and therefore kernel/filter-level) importance is
unreliable for any filter followed directly by BatchNorm -- a large weight
score does not mean "important" and a near-zero one does not mean
"unimportant."

The fix: also compute an **activation-based** filter importance,
`act_filter = mean over samples and spatial positions of |a * d(out_j)/d(a)|`,
where `a` is the *actual* conv/linear output (before BN). This looks at
real activation magnitudes at their trained/calibrated scale, not an
abstract weight-rescale direction, so it does not collapse the same way.
`manifest.json`'s per-layer `has_bn_after` flag tells the viewer (and any
other consumer) when to prefer `act_filter` over `mean_abs_s` as the
default filter score; it uses `act_filter` whenever `has_bn_after` is true.

Only the conv/linear layer's own (pre-BN) output is hooked for `act_filter`
in this version -- not also the post-BN/activation output the original spec
mentions as optional.

### How it's computed (and why the vmap/fallback split exists)

- **`mean_s` and `act_filter`** never need `vmap`. A single forward pass
  plus one plain `backward()` per output feature, with the output column
  summed over the batch before calling `.backward()`, gives the *exact*
  per-sample gradient in each sample's slot of every leaf/activation
  tensor's `.grad` -- because summing a batch of independent (no
  cross-sample coupling, i.e. eval-mode) forward passes and then
  differentiating is linear, so `d(sum_n out_j_n)/d(x_n) == d(out_j_n)/d(x_n)`
  for every `n` individually. This is completely robust: it's just the
  regular autograd machinery every model here already trains through.

- **`mean_abs_s`** genuinely needs *per-sample* gradients of something
  *shared* across the batch (the weight) -- the trick above only gives
  their sum, and `abs` doesn't commute with summing. Two implementations:
  - **primary**: `torch.func.vmap(torch.func.jacrev(...))` over the batch,
    via `functional_call` -- fast, vectorized, no Python loop over samples.
  - **fallback**: a plain per-sample loop (batch size 1), reusing the same
    eager backward-per-output-feature mechanism as the `mean_s` computation
    (which, at batch size 1, *is* already the per-sample gradient -- no
    trick needed).

  `importance.engine` tries the primary path and falls back automatically
  if it raises (logged at `WARNING`, "vmap path failed ... falling back").
  See the new pitfall entries in `docs/llm/pitfalls/brevitas_pitfalls.md`
  for exactly which quantizers trigger this.
  `tests/test_importance_engine.py::test_forced_fallback_matches_vmap_path`
  cross-checks both paths produce the same numbers on a plain model.

### Groups / depthwise convs

Weight-level arrays keep PyTorch's native `[F, C/groups, *k]` shape. The
manifest records `groups` and `in_channels` per layer so a consumer (the
viewer does this) can map a stored channel index `c` back to the real input
channel: `real_channel = (filter_idx // (out_channels // groups)) *
(in_channels // groups) + c`.

## On-disk format

```
result_dir/
  manifest.json
  layers/<layer_id>/<level>_<metric>.npy
  samples/{inputs,outputs}.npy, meta.json
```

- `manifest.json`: model/dataset/settings summary, the layer list (in
  forward-execution order, discovered via a dry-run forward-hook pass) with
  shapes/groups/`has_bn_after`/per-level-per-metric file+dtype+scale
  pointers, output feature names, and a `metadata.size_warning` set when the
  estimated total array size exceeds `size_warning_bytes` (default 2 GiB).
- Arrays are saved as `float16` (kernel/filter/layer) and, by default,
  `int8` + a per-array `scale` float for the weight level
  (`weight_dtype="uint8"` in `analyze()`; also accepts `"float16"`/
  `"float32"`). `Result.load()` memory-maps every array
  (`np.load(mmap_mode="r")`) and dequantizes `int8` lazily on read.
- `torch.fx` graph tracing for branch/residual edges (mentioned as a nice-
  to-have in `new_feature.md`) is **not implemented in v1** -- layer order
  is always the dry-run execution order, which is enough for the linear
  vertical layer stack the viewer renders; storing `graph_edges` for a
  future branch-aware layout is a natural v2 addition.

## Python query API

```python
from importance import load
result = load("runs/imp_mymodel")
result.filter("conv1", output=3)          # [F] mean_abs_s by default
result.filter("conv1", output=3, metric="act_filter")
result.kernel("conv1", output="all")
result.weight("conv1")                     # full [n_out, F, C, *k]
result.to_dataframe(level="filter")        # pandas long-format DataFrame
```

## Web viewer

`view("runs/imp_mymodel")` (or `python -m importance.serve <result_dir>
[--port 8000]`) starts `importance/serve.py`: a stdlib `http.server`
backend (no new dependency -- FastAPI/uvicorn were not already project
dependencies, see `docs/llm/CONVENTIONS.md`) serving the vendored static
app (`importance/static/`, plain ES modules, no build step, no CDN) plus
JSON endpoints (`/api/manifest`, `/api/layer/<id>/<level>/<metric>?output=`,
`/api/samples/...`) that return **one output-row slice at a time** rather
than a whole array, so it stays responsive on large results.

Layout: input sample viewer on top, a scrollable stack of layer cards
(filter bar chart -> click a filter for its kernel grid -> click a kernel
for its weight heatmap) in the middle, output feature chips at the bottom
that filter every chart above to the clicked output. Global controls:
metric (auto/`mean_abs_s`/`mean_s`/`act_filter`), normalization
(raw/per-layer/global/rank), sort, a top-k slider with a live
"retains X% of importance" readout, layer search, and CSV export.
PNG export and keyboard navigation (both called out as nice-to-haves in
`new_feature.md`) are not implemented in v1.

## Known limitations / v1 scope decisions

- Per-sample activation thumbnails in the filter view, and storing
  per-layer activations under `samples/` (both "optional" in the spec) are
  not implemented -- only sample inputs/outputs are stored.
- `output_reduce="flatten_topk"` and the spatial-channel-overflow fallback
  pick their top-k features from the *first* batch only (documented
  approximation, not the whole dataset).
- The optional `loss_fn` row always uses a per-sample loop (loss_fn's
  signature is arbitrary, so it can't generically go through the vmap
  path); if `loss_fn` is only meaningfully defined batched (Ultralytics-
  style), evaluating it at batch size 1 is an approximation.
- `use_quantized_weight=True` reads `module.quant_weight().value` when the
  layer exposes a `quant_weight()` method (Brevitas quantized layers);
  plain `nn.Conv*`/`nn.Linear` fall back to the float weight regardless of
  the flag.
