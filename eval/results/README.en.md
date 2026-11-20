# Live-model verification records

English · [中文](README.md)

## Phase 6: three long conversations

[Runner](../run_long_conversations.py) uses the configured real DeepSeek API, web search and existing Chinese knowledge index. Each group gets fresh synthetic business data and a copied knowledge catalog; the local E5 encoder and Qdrant index are shared read-only. Everyday business databases are not used.

```bash
.venv/bin/python eval/run_long_conversations.py --report eval/results/long-conversations-new.json
```

This incurs model/search charges. Use `--group 1` to run one group. Failed assertions produce exit code 1. Request replay, order selection, payment-method selection and confirmations use FastAPI endpoints. Reports retain replies, plans, traces, interactions, timings and assertions.

| Group | Turns | Coverage |
|---|---:|---|
| 1 | 21 | System help; three observations, explicit preferences, temporary instructions, shopping preference changes/revocation/reactivation; collaborating order queries; product/policy RAG, overview and missing evidence |
| 2 | 17 | Orders, details, status, tracking; payment/refund queries, selection, cancellation, individual confirmation, replay, restart recovery, dependencies and independent partial failures |
| 3 | 16 | English questions answered from Chinese documents; customer isolation; payment; live search and evidence limits; complaint guidance, formal tickets and follow-up; unsupported actions |

[Final report](long-conversations-final.json): 54 main turns and 11 selection/confirmation submissions passed across all 13 registered capabilities; offline regression: 176 passed. It combines complete passing groups and identifies their sources: [Chinese](long-conversations-final-1b.json), [payments/refunds](long-conversations-final-12.json), [English](long-conversations-final-3b.json). This is not a claim that the first run passed. Earlier results remain available: [initial](long-conversations-initial.json), [retest](long-conversations-retest.json), [final-13](long-conversations-final-13.json), [final-3](long-conversations-final-3.json). Manual review found empty-evidence explanations, a false-positive claim guard and overlapping subquestion answers even after automated assertions passed. These received further fixes; recorded final plans pass the latest schema.

[Supplement](long-conversations-supplement.json) records real-service Streamlit AppTest interactions: initially unselected payment method, selection, English UI switching, cancellation, catalog filtering and clearing. Its three temporary requests for more detail produced no persistent preference candidates; they do not validate three-observation activation. The latest subquestion-focus change was checked with two additional real API calls using actual task facts recorded in the English group; see [synthesis follow-up](long-conversations-synthesis-followup.json). Original long-run responses were not rewritten.

Fixes keep internal format-repair messages out of user input, route explicit refunds without order IDs to selection, strengthen answer-language and complaint-policy routing, retry invalid citation JSON once, block duplicate tickets for progress-only questions, forbid invented currencies/platform restrictions, require synthesis dependencies, focus each subquestion, and avoid displaying intermediate web answers alongside their synthesis. The ticket-claim guard preserves normal negative explanations. Habit observations are distinguished from explicit permanent settings. Invalid memory evidence is still discarded independently of the business plan.

Coverage concerns registered capabilities, business states and interaction paths. Financial actions change local synthetic records only; tickets do not notify humans. Replies were spot-checked, but assertions do not prove every claim is faithful or all phrasings work. Fixed business receipts still contain Chinese. Production payments, deployment, multiple-instance concurrency and all-device visual layouts were not tested here.

## Phase 4 historical records

These earlier tests used isolated synthetic databases and the configured DeepSeek model, with fresh data for each scenario/repetition.

| Report | Result | Description |
|---|---|---|
| [initial.json](initial.json) | 14/16 | Initial scenarios; policy routing and task-success value errors |
| [boundary-initial.json](boundary-initial.json) | 35/36 | Initial combinations; invalid unpaid condition value |
| [baseline-final.json](baseline-final.json) | 22/22 | Fixed basic scenarios and six generalization cases |
| [boundary-final.json](boundary-final.json) | 36/36 | Three independent runs per boundary scenario |

Fixes used general policy routing, enumerated task/business states and evidence-based explanations. No scenario-text business branches were added. Historical offline regression: 141 passed. Reports retain original failures, plans, tool traces, selections, actions, timings and errors. Finite samples do not establish correctness for arbitrary inputs.
