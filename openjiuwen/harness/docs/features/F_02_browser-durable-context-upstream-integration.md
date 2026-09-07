# F_02 Browser durable context and upstream runtime integration

| Item | Value |
|---|---|
| Date | 2026-09-07 |
| Scope | Browser agent assembly, working context, interactive probes and PageState |
| Test baseline | Browser unit suite and synchronous subagent task-routing tests |
| Refs | Upstream browser runtime and synchronous task-routing changes (#987, #1147) |

## Background

Upstream moved browser task truth into runtime evidence projections and removed
the default working-memory rail. The durable-context feature needs that rail to
retain facts and failures and assess each response's tool batch. The integration
keeps both mechanisms with separate ownership.

## State and lifecycle

The runtime owns requirements, evidence, blockers, completion and replanning.
Durable memory contains only `failures` and `key_facts`. One `batch_intent` covers
all tool calls in a model response; the next response assesses that batch once.
New invocations reset memory, while continuations within an invocation retain it.
Internal memory records are removed from the visible assistant response.

## Decisions

- Register both runtime and working-context rails; use durable mode in the
  browser factory. Explicit projection-only mode remains available and does not
  consume stored memory.
- Render the upstream `request`, `task` and `runtime_directive` projection with
  separate batch bookkeeping and durable memory. Structurally bound the JSON
  payload without dropping the assessment contract or changing persisted state.
- Keep upstream semantic evidence reconciliation, observation windowing and
  synchronous task routing.
- Retain accessibility enrichment and selection provenance together when a
  probe refreshes an existing target. Return enriched `elements` once, alongside
  a compact PageState identity summary.

## Rejected alternatives

Taking upstream's default assembly would disable durable memory. Taking the
feature's complete files would lose upstream evidence and probe fixes. Truncating
serialized JSON could remove batch obligations and produce invalid context.

## Verification

Regression coverage checks batch assessment and retry guards, invocation resets,
checkpoint reconstruction, projection-only isolation, valid bounded JSON with
batch obligations, and simultaneous accessibility/selection refresh. Existing
browser runtime and task-routing tests cover the upstream behavior.

## Known limits

Accessibility enrichment remains best effort. Prompt trimming can omit older
observations and shorten memory values; full durable state stays in the Session.
