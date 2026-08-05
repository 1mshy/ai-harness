"""Harder prompts for stress-testing decode under real work, not just short replies.

Each entry embeds its own synthetic data (transcript, log, invoice, etc.) so the
prompts share no long common prefix -- important on a server doing automatic
prefix caching (e.g. vLLM), where a shared prefix across concurrent requests
would make throughput numbers reflect cache hits, not generation. Each entry
also bounds its own output (JSON with a fixed key set, a table with a row cap,
a bullet list with an item cap) so requests across the rotation produce roughly
the same number of output tokens -- otherwise a ramp level that happens to
sample the short entries looks faster than one that samples the long ones for
reasons that have nothing to do with concurrency.

Used via `stress_test.py --hard` (or the "use hard prompts" toggle in the
interactive Settings menu) in place of the short PROMPTS rotation.
"""

from __future__ import annotations

HARD_PROMPTS: list[str] = [
    """Read the following customer support call transcript and extract structured data.

Transcript:
Agent: Thanks for calling Brightline Internet, this is Dana, how can I help?
Customer: Hi, my connection has been dropping every twenty minutes since yesterday morning. I work from home so this is costing me actual money.
Agent: I'm sorry to hear that. Can I get your account number?
Customer: It's 88214-B.
Agent: Thanks. I see a firmware update pushed to your modem two days ago. I'm going to schedule a technician for tomorrow between 9 and 11 AM, and I'll also credit your account fifteen dollars for the inconvenience.
Customer: Tomorrow works. Can the tech call before arriving?
Agent: Yes, they'll call thirty minutes ahead. Anything else?
Customer: No, that covers it. Thanks Dana.
Agent: You're welcome, have a good day.

Respond with only a JSON object with exactly these keys: account_number, issue_summary, root_cause, resolution_action, credit_amount_usd, appointment_window, customer_sentiment (one word). No other text.""",

    """Extract the line items from this OCR'd invoice text (formatting is imperfect on purpose).

INVOICE #4471   Vendor: Marlowe Office Supply
Bill To: Cascade Dental Group

Qty  Item                       UnitPrice   Total
 3   A4 Copier Paper (ream)       $4.25      $12.75
12   Blue Ballpoint Pens (box)    $1.10      $13.20
 1   Standing Desk Riser        $89.99       $89.99
 5   USB-C Charging Cable         $6.50      $32.50

Subtotal: $148.44
Tax (7%): $10.39
Total Due: $158.83
Due Date: 2026-08-15

Respond with only a JSON object: {"invoice_number": ..., "vendor": ..., "line_items": [{"item":..., "qty":..., "unit_price":..., "total":...}], "subtotal":..., "tax":..., "total_due":..., "due_date":...}. No other text.""",

    """Below is a snippet of application logs. Identify what went wrong and when it started.

09:14:02 INFO  order-service: healthy, 120 req/s
09:14:47 WARN  db-pool: connection wait time 850ms (threshold 500ms)
09:15:03 WARN  db-pool: connection wait time 1400ms
09:15:19 ERROR order-service: timeout acquiring DB connection after 2000ms
09:15:20 ERROR order-service: timeout acquiring DB connection after 2000ms
09:15:41 INFO  autoscaler: scaling db-pool max_connections 20 -> 40
09:16:10 INFO  db-pool: connection wait time 90ms
09:16:12 INFO  order-service: healthy, 118 req/s

Respond with only a JSON object with keys: first_anomalous_timestamp, root_cause (one sentence), resolution_action, total_duration_seconds. No other text.""",

    """Read this excerpt from a vendor services agreement and extract the parties' obligations.

Section 4.2: Vendor shall deliver monthly usage reports to Client no later than the 5th business day of each month, subject to a $500 credit per day late.
Section 4.3: Client shall provide payment within 30 days of invoice receipt; late payments accrue 1.5% monthly interest.
Section 4.4: Vendor shall maintain 99.5% uptime measured monthly; failure to meet this triggers a service credit of 5% of that month's fees per 0.1% shortfall.
Section 4.5: Either party may terminate for convenience with 60 days written notice.

Respond with only a markdown table (columns: Party, Obligation, Deadline/Metric, Penalty) with at most 4 rows, one per obligation with a concrete penalty. No other text.""",

    """Extract action items from this product planning meeting excerpt.

Priya: Okay, so for the Q3 launch we need the billing migration done first. Sam, can you own that?
Sam: Yeah, I'll have the migration plan drafted by next Wednesday.
Priya: Great. Jordan, what about the pricing page copy?
Jordan: I can get a draft to marketing by Friday, but I need the final tier names from Priya by Monday.
Priya: I'll send those Monday morning.
Sam: One more thing -- we still need someone to own the rollback plan in case the migration fails.
Priya: Let's have Jordan pick that up too, due end of next week.
Jordan: Works for me.

Respond with only a JSON array of objects, each with keys: owner, task, deadline. At most 5 items. No other text.""",

    """Classify the sentiment and primary intent of each of these five product reviews.

1. "Battery died after two weeks. Returning it."
2. "Works exactly as described, setup took five minutes. Very happy."
3. "Shipping was slow but the product itself is fine, no complaints there."
4. "Does anyone know if this is compatible with the 2019 model? Instructions don't say."
5. "Overpriced for what it is, cheaper alternatives exist that do the same job."

Respond with only a JSON array of 5 objects, each with keys: review_number, sentiment (positive/negative/neutral), intent (complaint/praise/question/comparison). No other text.""",

    """Reformat this misaligned, tab-broken data dump into a clean markdown table.

Name    Dept  Start
Alicia Chen ,Engineering,2023-01-09
Marcus Boone,   Sales   , 2021-11-02
Priya Nair,Engineering ,2024-03-15
Devon Ashworth , Support,2022-07-30

Respond with only a markdown table with columns Name, Department, Start Date, sorted alphabetically by Name. No other text.""",

    """Extract the defined terms from this contract excerpt.

"Confidential Information" means any non-public business, technical, or financial information disclosed by either party. "Effective Date" means the date this Agreement is signed by both parties. "Term" means the period beginning on the Effective Date and continuing for twelve (12) months, unless earlier terminated. "Affiliate" means any entity that directly or indirectly controls, is controlled by, or is under common control with a party.

Respond with only a JSON object mapping each defined term to its one-sentence definition. No other text.""",

    """Extract the key metrics mentioned in this earnings call excerpt.

"Revenue for the quarter came in at $42.3 million, up 18% year over year. Gross margin held steady at 71%. We ended the quarter with 1,240 enterprise customers, a net add of 85. Churn ticked up slightly to 2.1% monthly, which we're watching closely. Free cash flow was positive for the third consecutive quarter at $3.1 million."

Respond with only a JSON object with keys: revenue_usd_millions, yoy_growth_pct, gross_margin_pct, enterprise_customers, net_adds, monthly_churn_pct, free_cash_flow_usd_millions. No other text.""",

    """Extract a structured ingredient list from this recipe paragraph.

"Start by whisking two large eggs with a quarter cup of whole milk. In a separate bowl, combine one and a half cups of all-purpose flour with a teaspoon of baking powder and a pinch of salt. Fold in three tablespoons of melted butter and two tablespoons of sugar. Gently mix the wet and dry ingredients until just combined -- don't overmix."

Respond with only a JSON array of objects, each with keys: ingredient, quantity, unit. No other text.""",

    """Extract structured details from this stack trace.

Traceback (most recent call last):
  File "app/billing/processor.py", line 142, in charge_customer
    result = gateway.submit(payload)
  File "app/billing/gateway.py", line 58, in submit
    raise PaymentGatewayError(f"declined: {reason}")
app.billing.gateway.PaymentGatewayError: declined: insufficient_funds

Respond with only a JSON object with keys: exception_type, failing_file, failing_line, call_chain (array of "file:line" strings, outermost first), root_cause. No other text.""",

    """Read this negotiation chat and extract the final agreed terms.

Buyer: We can do $18,000 for the equipment but we need delivery by the 20th.
Seller: $18,000 is below our floor. I can do $19,500 with delivery by the 20th, or $18,000 with delivery by the 30th.
Buyer: Let's split the difference -- $18,750, delivery by the 25th?
Seller: I can accept $18,750 if you also cover the $400 freight cost.
Buyer: Deal, we'll cover freight.

Respond with only a JSON object with keys: final_price_usd, freight_cost_usd, delivery_date_or_deadline, who_pays_freight. No other text.""",

    """Extract structured requirements from this job posting excerpt.

"We're looking for a Senior Backend Engineer with 5+ years of experience in distributed systems. Must be proficient in Go or Rust; Python experience is a plus. This role requires experience with Kafka or similar message queues, and familiarity with Kubernetes is expected. Salary range is $150,000-$190,000 depending on experience, plus equity. Remote within US time zones."

Respond with only a JSON object with keys: min_years_experience, required_skills (array), nice_to_have_skills (array), salary_min_usd, salary_max_usd, location_requirement. No other text.""",

    """This is a synthetic training note (not real patient data). Convert the free-text note below into structured SOAP note fields.

"Patient reports intermittent lower back pain for two weeks, worse after long car rides, rates it 5/10. No numbness or tingling. On exam, mild tenderness over L4-L5, full range of motion, negative straight-leg raise. Plan: recommend ibuprofen 400mg as needed, stretching routine, follow up in two weeks if not improved."

Respond with only a JSON object with keys: subjective, objective, assessment, plan (each a short phrase, not a full sentence). No other text.""",

    """Read this interview debrief and extract a structured summary.

"Candidate had strong system design instincts, walked through the sharding tradeoffs clearly. Communication was excellent, and she asked good clarifying questions before jumping into a solution. Weak spot: her coding exercise had a bug she didn't catch until prompted twice, and she seemed unfamiliar with our stack's testing conventions. Overall the panel leaned toward a hire for a mid-level role rather than the senior role she interviewed for."

Respond with only a JSON object with keys: strengths (array, max 3), weaknesses (array, max 3), recommended_level, decision. No other text.""",

    """Extract structured details from these two voicemail transcriptions.

Message 1: "Hi, this is Karen from Redwood Dental calling to confirm your appointment tomorrow at 2 PM. If you need to reschedule, call us back at 555-0142. Thanks!"

Message 2: "Hey it's Tom, I need to talk about the contract ASAP, it's pretty urgent, call me back tonight if you can at 555-0198."

Respond with only a JSON array of 2 objects, each with keys: caller_name, callback_number, urgency (low/medium/high), purpose. No other text.""",

    """Extract structured package data from this shipping manifest.

MANIFEST #7734
PKG-001 | 12.4kg | Denver, CO | Fragile
PKG-002 | 3.1kg  | Austin, TX | Standard
PKG-003 | 27.0kg | Denver, CO | Standard
PKG-004 | 1.8kg  | Miami, FL  | Fragile

Respond with only a JSON array of objects, each with keys: package_id, weight_kg, destination, handling. No other text.""",

    """Extract structured data from this bug report thread.

"App crashes when uploading a file over 10MB on iOS 17.2, iPhone 13. Doesn't happen on Android. Steps: open app, go to Attachments, select a file over 10MB, tap Upload -- app force-closes within 2 seconds. Started after the 3.4.1 release. This is blocking our biggest customer's onboarding, need a fix this week."

Respond with only a JSON object with keys: platform, os_version, device, trigger_steps (array), first_broken_version, severity (low/medium/high/critical), business_impact. No other text.""",

    """Extract all deadlines and their associated obligations from this compliance memo.

"Per the updated data retention policy, all customer PII must be purged within 90 days of account closure. Security incident reports must be filed with the compliance team within 72 hours of detection. Annual access reviews are due every March 1st. Vendor risk assessments must be renewed 30 days before each vendor contract's anniversary date."

Respond with only a JSON array of objects, each with keys: obligation, deadline_description. No other text.""",

    """Read this internal incident chat and extract a timeline.

[14:02] Alex: getting 500s on checkout, anyone else seeing this?
[14:03] Priya: yep, confirmed, looks like it started right at 14:00
[14:05] Alex: rolling back the payments deploy from 13:55 now
[14:09] Alex: rollback done, 500s stopped
[14:11] Priya: confirmed clean, closing the incident

Respond with only a JSON object with keys: start_time, detected_by, likely_cause, resolution_action, resolution_time, total_duration_minutes. No other text.""",

    """Extract structured data from this restaurant order called in over the phone.

"Yeah hi, I'd like to order two large pepperoni pizzas, one medium veggie with extra olives, and a dozen garlic knots. Also a two-liter Coke. This is for pickup, name's Renata, and I'll be there in about forty minutes."

Respond with only a JSON object with keys: items (array of {name, size, qty, notes}), order_type, customer_name, pickup_eta_minutes. No other text.""",

    """Read this real estate listing description and extract structured facts.

"Charming 3-bed, 2-bath craftsman in the Elmhurst district, built 1948, 1,620 sq ft on a 5,200 sq ft lot. Updated kitchen with quartz counters, original hardwood floors throughout, detached one-car garage. HOA-free. Listed at $612,000. Property taxes approx $6,100/year."

Respond with only a JSON object with keys: bedrooms, bathrooms, year_built, sqft, lot_sqft, list_price_usd, annual_taxes_usd, has_hoa (boolean). No other text.""",

    """Extract structured data from this flight itinerary confirmation text.

"Confirmation ABC123. Passenger: J. Whitfield. Flight AA 2210 departs ORD 08:15 AM, arrives LAX 10:42 AM, Terminal 3, Gate B12, Seat 14C, Economy. Return flight AA 2215 departs LAX 6:05 PM arrives ORD 12:10 AM (+1 day), Seat 9A."

Respond with only a JSON object with keys: confirmation_code, passenger, outbound (object: flight_no, departure_airport, arrival_airport, departure_time, arrival_time, seat), return (same shape). No other text.""",

    """Extract structured findings from this code review comment thread.

Reviewer: This function mutates the input array in place, which surprised me given the name `sorted_copy`. Can we deep-copy first?
Author: Good catch, fixed in the next push.
Reviewer: Also line 84 has a bare `except:` swallowing all errors, that should at least log.
Reviewer: And the retry loop has no backoff, could hammer the upstream on failure.
Author: Added exponential backoff and logging, PTAL.
Reviewer: LGTM now, approving.

Respond with only a JSON array of objects, each with keys: issue, location_hint, status (open/resolved). At most 3 items. No other text.""",

    """Extract structured data from this warranty claim form text.

"Product: ThermoBrew Coffee Maker Model TB-400. Purchase date: March 3, 2025. Issue: unit stopped heating water after six weeks of normal use, no visible damage. Customer requests replacement, not refund. Serial number CB-9981234. Retailer: HomeGoods Plus."

Respond with only a JSON object with keys: product, model, purchase_date, issue_description, requested_remedy, serial_number, retailer. No other text.""",

    """Read this excerpt from a lease agreement and extract the key terms.

"Tenant shall pay $2,150 per month, due on the 1st, with a $100 late fee after the 5th. Security deposit of $2,150 is required at signing. Lease term is 12 months beginning June 1, 2026. No pets permitted without written consent. Tenant responsible for utilities except water and trash, which are included."

Respond with only a JSON object with keys: monthly_rent_usd, late_fee_usd, security_deposit_usd, lease_start_date, lease_term_months, pets_allowed (boolean), utilities_included (array). No other text.""",

    """Extract structured data from this customer churn exit-survey response.

"We're canceling mainly because the reporting dashboard never loaded the custom date ranges we needed -- support said it was a known bug for over three months. Pricing was fine, actually a bit better than competitors. The onboarding was smooth. We might come back if the dashboard issue gets fixed."

Respond with only a JSON object with keys: primary_cancellation_reason, secondary_factors (array), pricing_sentiment, onboarding_sentiment, likely_to_return (boolean). No other text.""",

    """Extract structured data from this parking ticket / citation text.

"Citation #A88213. Vehicle: blue Honda Civic, plate 7XKD192. Violation: expired meter, exceeded by 47 minutes. Location: 4th Ave & Pine St. Fine: $53, due within 21 days or increases to $78. Issued 2026-06-14 at 2:47 PM."

Respond with only a JSON object with keys: citation_number, vehicle_description, plate, violation, fine_amount_usd, late_fine_amount_usd, due_days, issued_datetime. No other text.""",

    """Read this excerpt from a research paper's methods section and extract the study design.

"We conducted a randomized controlled trial with 240 participants (120 per arm) over 8 weeks. The treatment group received the intervention twice weekly; controls received standard care. Primary outcome was change in symptom score, measured at baseline, week 4, and week 8. Attrition was 6% in the treatment arm and 9% in controls."

Respond with only a JSON object with keys: study_type, total_participants, arms (array of {name, n}), duration_weeks, primary_outcome, measurement_timepoints (array), attrition_pct_by_arm (object). No other text.""",

    """Extract structured data from this multi-turn tech support chat.

User: my printer says offline but it's plugged in and turned on
Agent: are you connected to wifi or usb?
User: wifi
Agent: can you check if the printer's ip address changed? go to settings > network
User: oh yeah it's different now, was .105 now it's .112
Agent: that's the issue, your router probably reassigned it. i'll walk you through updating it in your printer driver settings
User: ok did that, printing a test page now... it worked!

Respond with only a JSON object with keys: reported_symptom, diagnosis, fix_applied, resolved (boolean), connection_type. No other text.""",

    """Extract structured data from this expense reimbursement report line.

"Trip to Denver, June 10-12, 2026 for the Northwind client kickoff. Flight: $412.50. Hotel (2 nights @ $189): $378.00. Meals: $94.20 total across 3 days. Rideshare to/from airport: $61.10. All receipts attached. Client: Northwind Logistics."

Respond with only a JSON object with keys: purpose, client, trip_dates, line_items (array of {category, amount_usd}), total_usd. No other text.""",

    """Read this excerpt from a software changelog and extract structured release notes.

"v3.4.0 (2026-05-22): Added dark mode support across all screens. Fixed a crash on startup affecting users on iOS 16. Improved sync latency by ~40% for large workspaces. Deprecated the legacy export format, will be removed in v4.0. Known issue: search occasionally returns stale results after a bulk import."

Respond with only a JSON object with keys: version, release_date, added (array), fixed (array), improved (array), deprecated (array), known_issues (array). No other text.""",

    """Extract structured data from this court docket summary excerpt.

"Case No. 2026-CV-04471, Filed 2026-02-10. Plaintiff: Reeves Manufacturing Co. Defendant: Alden Supply Partners. Claim: breach of contract, damages sought $340,000. Motion to dismiss filed by defendant on 2026-03-15, denied 2026-04-02. Trial date set for 2026-11-09."

Respond with only a JSON object with keys: case_number, filed_date, plaintiff, defendant, claim_type, damages_sought_usd, trial_date, motions (array of {type, filed_date, outcome}). No other text.""",

    """Extract structured data from this HR onboarding checklist status email.

"Update on new hire Marcus Fielding, starting July 6th: laptop ordered and expected to arrive July 3rd. Badge access request submitted, pending security approval. Benefits enrollment link sent, not yet completed by Marcus. Manager intro meeting scheduled for July 6th at 10 AM. Still need: parking permit request."

Respond with only a JSON object with keys: employee_name, start_date, tasks (array of {task, status (done/pending/not_started), note}). No other text.""",

    """Read this excerpt from a survey of employee satisfaction free-text comments and extract themes.

1. "Management communicates changes way too late, always feels like a surprise."
2. "I love the flexibility of remote work, wouldn't want to lose that."
3. "Pay is below market for my role, I've checked."
4. "My manager is genuinely supportive and gives useful feedback."
5. "The tools we use are outdated and slow us down daily."

Respond with only a JSON object with keys: positive_themes (array), negative_themes (array), comment_count. No other text.""",

    """Extract structured data from this vehicle accident report narrative.

"Driver 1 (blue sedan) was stopped at a red light on Main St. Driver 2 (silver truck) failed to stop in time and rear-ended Driver 1 at approximately 15 mph. No injuries reported. Driver 2's front bumper sustained damage; Driver 1's rear bumper was dented. Police report filed, Driver 2 cited for following too closely. Weather was clear, road dry."

Respond with only a JSON object with keys: at_fault_driver, impact_type, estimated_speed_mph, injuries (boolean), damage_summary (object: driver1, driver2), citation_issued_to, weather_conditions. No other text.""",

    """Extract structured data from this social media crisis/complaint post.

"Ordered a birthday cake for pickup at 3pm for my daughter's party. Showed up and they hadn't even started it, said the order got lost in their system. Party started at 4, we had no cake. Manager offered a refund and a $25 gift card but honestly the whole day was ruined. This is the third issue I've had with this bakery."

Respond with only a JSON object with keys: complaint_summary, promised_time, actual_outcome, compensation_offered, is_repeat_issue (boolean), sentiment. No other text.""",

    """Read this excerpt from a podcast transcript and extract the key claims made.

Host: So you're saying remote work actually increased productivity at your company?
Guest: Yeah, we measured it two ways -- output per engineer went up about 12%, and voluntary attrition dropped from 18% to 11% annually. The surprising part was that collaboration scores, measured through our internal survey, stayed flat, they didn't drop like we feared.
Host: And you attribute that to what exactly?
Guest: Mostly async documentation habits that got forced into place. People write things down now instead of relying on hallway conversations.

Respond with only a JSON array of objects, each with keys: claim, supporting_metric (or null). At most 4 items. No other text.""",

    """Extract structured data from this used car listing description.

"2019 Toyota RAV4 XLE, 47,200 miles, one owner, clean title, no accidents per Carfax. Recent service: new brakes and tires installed April 2026. Minor cosmetic scratch on rear bumper. Asking $21,500, price is firm. Located in Sacramento, CA."

Respond with only a JSON object with keys: year, make, model, trim, mileage, owners, accident_history, price_usd, price_negotiable (boolean), location. No other text.""",

    """Extract structured data from this two-person customer onboarding call transcript.

Rep: So walk me through your current setup -- what CRM are you migrating from?
Client: We're on Spreadsheets, honestly. About 1,400 contacts, mostly B2B.
Rep: Got it, we can import that via CSV. Any integrations you rely on, like email or calendar?
Client: We use Google Workspace for everything.
Rep: Perfect, that's a native integration. I'll have our onboarding team reach out within 2 business days to schedule the data import.

Respond with only a JSON object with keys: current_system, contact_count, integrations_needed (array), next_step, next_step_timeline. No other text.""",

    """Extract structured data from this internal Slack thread about a security incident.

[09:02] sec-oncall: seeing unusual login attempts from a new IP range against the admin panel, ~200 attempts in 5 min
[09:04] sec-oncall: IP blocked at the WAF level, no successful logins detected
[09:06] eng-lead: should we force a password reset for admin accounts as a precaution?
[09:07] sec-oncall: yes, doing that now, also enabling rate limiting on the login endpoint
[09:15] sec-oncall: reset emails sent to all 12 admin accounts, rate limiting live

Respond with only a JSON object with keys: detected_at, attack_type, attempts_count, blocked (boolean), mitigations (array), affected_account_count. No other text.""",

    """Read this excerpt from a customer's renewal negotiation email and extract the outcome.

"Given our usage has dropped by about 30% this year, we were hoping for a corresponding reduction in the renewal price, or at minimum keep it flat rather than the proposed 8% increase. We'd also like to move from annual to quarterly billing if possible. We're prepared to sign a 2-year term if you can meet us on price."

Respond with only a JSON object with keys: usage_change_pct, requested_price_change, proposed_increase_pct, requested_billing_frequency, term_offered_years, willing_to_commit_longer (boolean). No other text.""",

    """Extract structured data from this excerpt of a building inspection report.

"Electrical panel is outdated (Federal Pacific brand), recommend replacement -- safety concern. Roof shows moderate wear, approximately 8-10 years of remaining life. Plumbing is copper throughout, no visible leaks. HVAC system installed 2015, functioning normally, due for servicing. Foundation shows minor hairline cracking, not considered structural."

Respond with only a JSON array of objects, each with keys: system, condition, concern_level (none/low/medium/high), recommendation. No other text.""",

    """Extract structured data from this text message exchange about a delivery.

Driver: Hey, I'm outside but the gate code isn't working, can you buzz me in?
Customer: sorry! try 4471# instead
Driver: that worked, thanks. leaving the package by the door
Customer: perfect thank you so much
Driver: no problem, have a good one

Respond with only a JSON object with keys: issue, resolution, delivery_location, resolved (boolean). No other text.""",

    """Read this excerpt from a teacher's parent-conference notes and extract a structured summary.

"Emma is excelling in reading comprehension, consistently above grade level, but struggles with math word problems specifically -- computation itself is fine. Socially she's well-adjusted, has a solid friend group. Recommend extra practice with word problems at home, maybe 10 minutes, 3x a week. No concerns about behavior or attendance."

Respond with only a JSON object with keys: strengths (array), areas_for_improvement (array), social_notes, recommended_action, concerns (array, empty if none). No other text.""",

    """Extract structured data from this excerpt of a product return/RMA request.

"Item: Wireless Noise-Cancelling Headphones, Order #58821. Reason for return: right earcup stopped producing sound after 3 weeks of normal use. Not damaged, no signs of misuse. Customer prefers a replacement over refund, and needs it before a trip in 10 days if possible."

Respond with only a JSON object with keys: item, order_number, defect_description, damage_or_misuse (boolean), requested_remedy, urgency_note. No other text.""",

    """Extract structured data from this excerpt of a conference talk abstract.

"In this talk we present a caching layer that reduced p99 latency from 340ms to 60ms for a read-heavy workload serving 50,000 requests per second. We'll cover the cache invalidation strategy, which uses versioned keys rather than TTLs, and share three production incidents caused by cache stampedes and how we fixed them."

Respond with only a JSON object with keys: p99_before_ms, p99_after_ms, workload_rps, invalidation_strategy, incident_count_mentioned. No other text.""",

    """Read this excerpt from a restaurant health inspection report and extract structured findings.

"Violation 1 (critical): cold-holding unit measured at 48F, above the required 41F max -- corrected on site by technician call. Violation 2 (non-critical): hand-washing sign missing in kitchen restroom. Violation 3 (critical): raw chicken stored above ready-to-eat vegetables in walk-in cooler -- corrected immediately by staff. Overall score: 82/100."

Respond with only a JSON array of objects, each with keys: violation_number, severity (critical/non_critical), description, corrected_on_site (boolean). Plus note the overall_score separately is not needed -- just the array. No other text.""",

    """Extract structured data from this excerpt of a customer's product feature request submitted via support ticket.

"It would be great if we could export reports directly to Google Sheets instead of just CSV -- our whole team lives in Sheets and the manual import step is annoying, we do it maybe 15 times a week across the team. Also, a smaller ask: can the default date range remember our last selection instead of resetting to 'last 7 days' every time?"

Respond with only a JSON array of objects, each with keys: request, frequency_or_impact, priority_hint (high/low based on stated frequency). No other text.""",

    """Extract structured data from this excerpt of a gym membership cancellation call transcript.

Rep: I see you're looking to cancel your membership, can I ask why?
Member: I moved to a different city three weeks ago, this location just isn't accessible anymore.
Rep: Understood, that's a valid reason. I can process the cancellation with no fee since it's a relocation. Do you want to keep your account active until the end of the current billing cycle on the 18th, or cancel immediately?
Member: End of the cycle is fine.
Rep: Done, you're set to cancel effective the 18th, no further charges after that.

Respond with only a JSON object with keys: cancellation_reason, fee_waived (boolean), effective_date, further_charges (boolean). No other text.""",
]
