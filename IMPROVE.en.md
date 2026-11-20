# Improvements

English · [中文](IMPROVE.md)

Phase 6 is current. Earlier phases are historical; current implementation takes precedence where behavior changed.

## Phase 6: current changes

1. **Shopping memory.** Brand, category and budget candidates share the existing planning call and quoted evidence. Explicit settings activate immediately; observations need three distinct requests. Active shopping preferences require explicit changes.
2. **Modular interface.** Chat, memory, catalog and guide are separated into presentation modules. Light-theme configuration fixes sidebar contrast under dark system themes. Task summaries reflect actual state without implying a serial order for parallel reads.
3. **Catalog interaction.** Filter/document columns, real catalog categories, pagination, filter clearing and distinct empty-library/no-result states. No product brand/price matching is claimed.
4. **Earlier verification.** Earlier offline regression: 166 passed, including language switching and English payment controls. Three real DeepSeek turns verified budget saving, cross-session recall and one-off budget isolation using temporary data. Desktop empty chat, memory and empty catalog were inspected. Mobile visual checks and the entire 12-turn shopping script remain unverified; newer long-conversation records are below.
5. **Documentation and localization.** README and architecture checked against code; verbose validation, API and internal-field explanations reduced. IMPROVE filename casing standardized. English documentation and Chinese/English UI controls added. UI language does not overwrite stored response-language preferences.

6. **Cross-language knowledge answers.** One Chinese corpus supports both languages. English overviews and optional product articles are handled; planning can split compound questions and produce equivalent Chinese queries. Generated answers follow current instructions, active language preferences, then question language, retaining source text.

7. **Live long conversations and fixes.** Added a three-group real-API runner; 54 turns and 11 interaction submissions passed across all 13 registered capabilities using isolated synthetic databases. Fixed internal repair errors treated as user input, missing refund selection, answer language and complaint-policy routing, citation JSON retry, duplicate tickets on progress questions, invented currencies, empty-evidence synthesis, false-positive claim guards and repeated answers. Habit observations are distinguished from permanent settings. Offline regression: 176 passed. Initial and repeated runs are retained; see [results and limits](eval/results/README.en.md).

## Phase 5: historical changes

1. Response preference extraction was added to the existing planning call, originally with two candidates and four categories. Quotes must occur in the current input; invalid candidates do not invalidate the business plan.
2. Explicit preferences activate directly; repeated same-value observations activate after three distinct requests. Evidence is deduplicated by customer/request/category. No semantic clustering or full profile history was introduced.
3. Revocation preserves state. Completed-request memory failures stay pending for replay without rolling back business actions. Financial flows finish before memory writes.
4. Memory UI shows real candidate progress, evidence, active/revoked counts and recent extraction records; empty states contain no fabricated preferences.
5. Local multilingual E5 Small and Qdrant provide CPU embeddings, filtered retrieval and cached models; DeepSeek remains online for answers.
6. Manifest import supports TXT, Markdown, text PDF and DOCX; uses exclusive import/shared app locks, synchronizes removed files and skips unchanged encoding. No OCR or historical rule isolation.
7. Knowledge search and dependent synthesis were registered in the existing graph. Exact product aliases and upstream order products identify targets. Ordinary retrieval keeps six deduplicated results from twelve candidates within an evidence budget.
8. Citation schema and source-ID checks were added. They do not prove every claim is grounded. Platform policy gaps do not silently fall back to FAQ/web knowledge or alter refund eligibility.
9. Added 200 fictional policies and 200 fictional products, full-text browsing and read/revoke/catalog APIs. No authentication, upload editor or real orders were added.
10. Replaced the old 16 UI examples with five long scenarios totaling 77 turns. Consolidated import/cache/reset guidance and distinguished Qdrant service from the app image.
11. Offline regression: 161 passed. Historical live E5/DeepSeek answers and local Qdrant import/search checks were recorded. Full manual execution, app-image deployment and 100,000-chunk performance were not verified.

## Phase 4: historical changes

1. Structured capability-based plans support owned-order lists, upstream targets, dependencies and conditions; format errors can be repaired three times.
2. UI/API use one persistent graph; independent reads run in parallel and action preparation/execution are serial.
3. Full order details combine order and payment executors. Placed status does not imply unpaid.
4. Payment methods became a shared UI/backend registry with explicit selection before confirmation.
5. Mixed operations bind tasks and inputs separately; only the current action can execute. Cancellation preserves independent work.
6. Conditional tasks resume after confirmed actions; unmet/unknown conditions and failed prerequisites skip dependent work.
7. Grounded explanation and search preserve evidence boundaries; status enums prevent invented planning values.
8. Responses expose actual tasks, agents, inputs and tool traces for inspection.
9. SQLite checkpoints, stable action positions, transactional action results and replay protect recovery. Unfinished reads/model calls may repeat.
10. Fresh-store initialization was simplified without introducing migration, ownership backfill or archive machinery.
11. Offline regression: 141 passed. Historical real-model checks passed 22 scenarios and 36 repeated runs; initial failures were retained. These samples do not guarantee arbitrary language correctness.
12. README, architecture, scenarios and evaluator were synchronized. Legacy Word/images are not current implementation specifications.

## Retained early results

- Phase 1: immediate user-message display, stable request replay and local-only tickets.
- Phase 2: customer/order isolation, independent conversations, owned-order refund selection, shared eligibility checks, manual preferences, 50 displayed turns and six completed context turns.
- Phase 3: shared DeepSeek credentials for model/search, bounded search with sources, and explicit-only human/complaint escalation. Technical errors and ordinary dissatisfaction do not create tickets.
