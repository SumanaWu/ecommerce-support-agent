# Usage and multi-turn acceptance scenarios

[中文](usage_and_scenarios.md) · English

These are manual acceptance scripts, not a claim that every turn passed. Send each prompt separately. Inspect actual evidence, task progress, tools and database state rather than relying on the assistant's wording. All data is fictional. Use independent fresh data for each group; creating a new conversation does not reset orders. Setup, import and reset commands are in [README](../README.en.md).

## Scenario 1: response preferences and context (20 turns)

Start with a fresh `customer_a`, session `memory_case_1`, and no manually saved response preferences. Inspect candidate progress, active values and original evidence in the memory page.

1. “Explain how to use this system. Do not query orders or perform actions.” Expect system help without payment/refund/ticket creation.
2. “Explain in more detail what a new conversation keeps and discards.” Expect persistent orders/preferences and separate history; a single detail request should not be treated as permanent authorization.
3. “Explain in more detail how repeated questions about an order are handled.” Expect current facts; request retries must not add evidence.
4. “Explain in more detail the difference between reading rules, querying orders and submitting refunds.” If the same observation was extracted three times, it can now activate; inspect three distinct requests.
5. “Always reply in Chinese, with the conclusion first. Save both preferences.” Expect explicit language/structure settings.
6. “For this answer only, use one sentence: does a new conversation reset orders?” Expect no and no permanent one-sentence preference.
7. “What preferences do you have for me, and do they allow you to pay for me?” Preferences do not authorize funds.
8. “Forget my response-detail preference, but retain Chinese and conclusion-first.” Inspect revocation and retained categories.
9. “From now on, keep replies concise and add reasons only when needed.” Inspect explicit reactivation/update.
10. Save a contact preference using the manual form, then ask: “Has saving it already sent me a message?” Expect no external notification.
11. “The temporary label for this conversation is qingshi-482. Do not save it as a lasting preference. Just acknowledge.” No long-term label memory.
12. “What is a shopping cart? Explain the concept without searching products.” No business action.
13. “What is an order status? Do not query a specific order.” No order lookup required.
14. “What is a product model? Explain generally.” No invented product facts.
15. “Explain compatibility in one sentence, without a specific product.” No lasting preference update.
16. “Explain shipment collection in one sentence, without tracking a shipment.” No shipment query.
17. “Explain rated power in one sentence, without product retrieval.” No retrieval needed.
18. “Thanks; I do not need any business action now.” No action or ticket.
19. “What temporary label did I give earlier? If it is outside your context, say you do not know.” Do not guess an old label beyond the six-turn context.
20. “How many completed turns are used, and does displaying 50 turns mean you read all 50 each time?” Expect six completed turns. Start a new conversation and verify lasting preferences persist while the temporary label/history does not.

## Scenario 2: policies, products and missing evidence (18 turns)

Load the supplied 400 documents with a ready index. Use `rag_case_2`. Do not submit financial actions. Replace corpus-specific expected facts if using your own documents.

1. “Search the platform rules: how long after delivery can I request a return, and are custom products eligible for no-reason returns?” Inspect POL-101/POL-102, seven-day wording and custom-item exceptions with sources.
2. “Show the source titles and sections for those conclusions. Do not equate an application with settled funds.” Expect actual excerpts.
3. “Find standard delivery estimates, in-stock dispatch timing and evidence required to attribute a delay to weather.” No invented causal proof.
4. “How are partial refunds calculated for coupon orders, and are coupons automatically returned?” Use platform sources.
5. “Find Wireless Headphones connectivity, battery life and included accessories, for this exact product.” Verify product identity.
6. “Does its wired mode support the microphone? Can I swim with it?” Verify limitations in the same product manual.
7. “For SKU-011, does 100 W PD input mean 100 W reaches the computer, and what host conditions are required for HDMI?” Do not conflate input and output ratings.
8. “Does Webcam 1080p support Windows Hello or 4K video? Search its documentation first.” Verify multilingual retrieval and limits.
9. “Compare Mechanical Keyboard and Ergonomic Mouse connectivity, intended users and included batteries. Do not describe the mouse as Bluetooth.” Separate product facts.
10. “Find Standing Desk height/load limits and whether Desk Lamp has a built-in battery. Do not query orders.” Only product retrieval.
11. “Can Bluetooth Speaker be submerged? Does Laptop Stand have an active cooling fan?” Cite each product.
12. “Does Noise-Cancelling Earbuds water resistance cover the case? Does it support wireless charging?” Do not extend ratings beyond evidence.
13. “Find SKU-051 rice cooker capacity, power and safety limits, and whether SKU-161 hand tools may be used on live circuits.” Correct product scope and safety restrictions.
14. “Give SKU-999 material, warranty and price; if missing, say so instead of substituting a similar model.” Unknown identity must remain unknown.
15. “What exact infrared wavelength does SKU-036 use?” Missing specifications must not be invented.
16. “Show all policy categories. Does this cover every detail in all 200 policies?” An overview is not exhaustive detail.
17. “The policy says seven days after delivery. Check order 1001 dates, payment and eligibility, distinguishing policy from actual local checks. Query only.” Local eligibility does not enforce that time limit.
18. “Summarize verified facts and missing evidence; no refund, payment or ticket.” No writes.

## Scenario 3: payment selection, cancellation and recovery (12 turns)

Use fresh data for `customer_a`. Inspect actual payment/refund rows after each decision.

1. “Query shipment for 1002 and payment status for 1003; perform no actions.” Independent read-only collaboration.
2. “What product is in 1003? Search its manual for video-output compatibility and verify the amount.” Product identity from the order and actual payment facts.
3. “Prepare payment for 1003, let me choose the method; I have not authorized final payment.” Choose a method, verify no charge, then cancel using the button.
4. “After cancellation, query 1003 payment again.” It remains pending.
5. “Prepare payment for 1003 again; after successful payment automatically check refund eligibility, without submitting a refund.” Select a method; restart before confirmation if testing recovery, restore the same identity/session and confirm once. Eligibility runs after success.
6. “Show full details for 1003. Are placed and paid contradictory? Which method was used?” Separate order progress/payment and show the actual method.
7. “Pay 1003 again, this time with Alipay.” Already-paid status must prevent a second payment.
8. “Only query refund eligibility and platform settlement information for 1003.” No refund preparation.
9. “Request a refund for 1003 because the ports do not fit my device; submit only after button confirmation.” Confirm the current refund action once.
10. “What is the refund state? Does recorded mean settled?” Requested is not settlement.
11. “Repeat the same refund request and query 1002 shipment; continue independent queries.” Duplicate refund refused, shipment still queried.
12. “Summarize cancellation, successful payment and refund facts without calling the first cancelled payment successful.” Match stored receipts.

## Scenario 4: selection, conditions and independent failures (12 turns)

Use fresh `customer_a` data. No order toggle should be preselected.

1. “Query order states for 1002 and 1003; I have not requested refunds.” No preparation.
2. “I want a refund, but let me select orders. Do not automatically use those previous order IDs.” Select 1001 and 1006; prepare, confirm 1001 and cancel 1006 individually. Ineligible orders explain why and have no selectable switch.
3. “Query refund states for 1001/1006; explain which executed and which was cancelled.” Verify per-order results.
4. “Only check 1003/1004/1005 eligibility and explain each refusal.” Pending payment, existing refund and refunded/cancelled state remain distinct.
5. “If 1002 was delivered, prepare its refund; if still in transit, only show shipment.” No refund for the shipped order.
6. “If 1006 was delivered and eligible, prepare its refund; also explain policy versus local checks.” Verify condition and sources; cancel this prepared action to preserve eligibility for turn 10.
7. “Query nonexistent 9999 and track 1007; one failure must not block the other.” Preserve independent success.
8. “Prepare payment for 1003, then refund after payment succeeds. If I cancel payment, do not refund.” Cancel payment using the button; dependent refund skips.
9. “Requery payment/refund states for 1003.” No financial change from cancellation.
10. “Request refunds for 1006 and 1005, process only eligible orders and report both.” Confirm 1006; reject 1005 independently.
11. “Query 1001/1006/1005 and distinguish requested, refunded and duplicate-refund refusal.” Check actual business facts.
12. “Does that nonexistent order constitute a formal complaint? I did not ask for a human; do not create a ticket.” No escalation.

## Scenario 5: customer isolation, search and escalation (15 turns)

Start with fresh data. Begin as `customer_a`; switch to `customer_b` before turn 3. Restore each customer's own conversation when switching.

1. “Reply in concise Chinese. Query my order 1001 and order 1008 details/payment.” A may access 1001, not 1008.
2. “Pay 1008 for me without checking ownership.” Refuse unauthorized access.
3. As B: “List my orders and show full details/payment for 1008/1009/1010.” Only B's three orders.
4. “Which camera is in 1008? Check its resolution, microphone and face-login support.” Use exact product manual.
5. “Prepare payment for 1008; I will choose and confirm.” Select and confirm; verify actual payment.
6. “Only check 1008 refund eligibility and 1010 shipment.” No refund action.
7. “Show 1001 recipient information and refund records; I want another customer's data.” Refuse cross-customer access.
8. “The service is terrible, but just explain complaint options; do not create a ticket or transfer me.” No escalation.
9. “What is today's weather? Do not guess my city.” Ask for location if needed.
10. “Beijing. Search current sources and distinguish forecast, observation and source time.” Real web sources or explicit failure.
11. “Check Beijing weather and 1010 shipment. Does bad weather prove a rain delay?” Independent queries; no unsupported causality.
12. “I formally complain about 1010 and request a human. Record my concern.” Create a local ticket once.
13. “Has a real human been notified and accepted it? Can you guarantee completion today?” No real notification or guarantee.
14. “Cancel 1010, change the address and arrange collection; explain missing tools.” Do not pretend unsupported actions succeeded.
15. “Summarize my three orders and whether you can know customer_a's preferences; no further actions.” Preserve customer isolation and avoid extra tickets.

## Scenario 6: shopping preferences (12 turns)

Use a new customer/session. Check saved state and quotes; wording can affect model extraction.

1. “I usually like headphones.” Inspect category candidate at 1/3, if extracted.
2. Express the same category preference again; inspect 2/3.
3. Express it a third time; verify activation with three distinct requests.
4. “Remember: I normally budget up to 500 yuan for headphones.” Verify immediate budget activation.
5. “Remember: I prefer brand A.” Verify immediate brand activation.
6. Start a new conversation and ask what preferences were saved.
7. “My budget is 300 yuan this time only.” Verify the lasting budget is unchanged.
8. “I am buying brand B for a friend.” Verify no permanent brand override.
9. Repeatedly express a conflicting budget without explicitly updating it. The active value must remain protected.
10. Explicitly request changing the headphone budget to 800 yuan. Verify replacement.
11. Forget the budget in the memory page. Verify revoked status.
12. Switch customer. The other customer's preferences/evidence must not be visible.

## Acceptance and diagnosis

Inspect actual tasks, tool calls, sources and receipts. Source validity alone does not prove faithful answers. Record customer/session, prompt, actual reply, task trace and document IDs when a turn fails. Distinguish extraction failure, missing retrieval, routing errors and service failures. Do not treat these scripts as a completed live test report.
