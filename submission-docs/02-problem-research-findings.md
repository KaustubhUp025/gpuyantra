# Problem Research Findings — Taskmaster Track Candidate Sourcing

**Status:** Research complete. Input to the problem-selection decision. Not yet a decision.
**Date:** 2026-08-17
**Method:** Four independent research agents, parallel, non-overlapping source domains.
**Evaluated against:** the decision gates in `01-hackathon-rules-and-evaluation-baseline.md` §10.

| Agent | Source domain | Primary access method |
|---|---|---|
| A | Reddit + open communities | Arctic Shift archive API (Reddit blocks direct fetch) |
| B | Hacker News + YC + startup news | HN Algolia API, ycombinator.com/rfs |
| C | Vertical/professional communities | Trade press, associations, Amazon Seller Forums, LinkedIn |
| D | Automation whitespace | n8n/UiPath Discourse APIs, arXiv, dated job postings |

---

## 0. Read this first — evidence quality

Three of the four agents independently reported the same contamination problem, and it
changes how much weight each finding carries.

- **Agent C:** *"roughly 70% of what surfaces on these queries in 2026 is vendor marketing
  content engineered to look like practitioner commentary."* Statistics like "340 hours per
  firm" or "85% of delays" are uncited seller-side claims.
- **Agent D:** *"A large fraction of '2026 AI agent failure statistics' on the open web is SEO
  content-farm material recycling each other with unverifiable numbers like '88% of agents
  fail.' I have not used any of it."*
- **Agent A:** many 2026 "what's your most tedious task?" threads are founder market-research
  bait, and Redditors say so: *"New accounts trying to make/market a SaaS for this are
  relentless, eh? This exact thread is posted every few days here."*
  (r/sysadmin, 2026-07-13)

**Access failures worth knowing:** Reddit blocks automated fetching (Agent A routed around it
via the Pushshift successor; Agent D could not and rerouted entirely). AccountingWEB and
TruckersReport returned HTTP 403. Agent B could not reach the 2025 YC RFS pages (404 live,
archive.org blocked) so its 2025 RFS material is secondary-sourced. Agent B also flagged some
of its own quotes as paraphrases rather than verbatim.

**Consequence:** evidence strength is uneven and is noted per candidate below. Reddit and
forum-API sources (Agents A, D) are the strongest — real people, hard timestamps, unprompted.
Vertical trade sources (Agent C) are the weakest. **Re-verify any quote before it enters a
submission narrative.**

---

## 1. The meta-finding — the chase loop

Four agents searched four disjoint source domains and converged on the same structural shape.
This is the most reliable signal in the entire study, because nothing coordinated it.

Agent D named it directly:

> "The 2026 whitespace is not 'agents can't read documents' — that threshold is crossed. It is
> **the chase loop**: noticing something is missing, deciding it matters, going and getting it
> from a human, and verifying it landed."

Every top-ranked candidate across all four reports is an instance of it:

| Candidate | What gets chased | Agent |
|---|---|---|
| FOIA response | Custodian departments holding records | C |
| Bookkeeping close | Clients who won't explain transactions | A, B |
| IT offboarding | App owners for non-SSO accounts | A |
| AP exceptions | Buyers/warehouse on failed 3-way match | D |
| Marketplace compliance | Overseas suppliers for certificates | C |
| Recruiting coordination | Interviewers who owe feedback | D |
| Clinical binder currency | Staff with expired CVs/licenses | C |
| Construction A/P | Supervisors who owe approvals | A |
| Subcontractor COI | Subs and their insurance agents | C |

**Why it is unbuilt:** Agent D found the benchmark for this task class — *SentinelBench*
(arXiv 2606.05342, submitted 2026-06-03) — was introduced only in **June 2026**:

> "the default model of agent behavior is continuous action... **This is the wrong approach for
> many long-running tasks**, which are better served by a strategy of sustained attention.
> Instead, agents should monitor an environment, notice when an external event makes progress
> possible, then respond promptly without wasting resources while waiting."

A task class that only got a benchmark two months ago is barely measured and therefore barely
productized. **Whatever we build, the chase loop should be the core of it.**

---

## 2. Capability calibration — the hard ceiling

This determines how ambitious the architecture can be, and it is not negotiable.

**OSWorld 2.0** (arXiv 2606.29537, submitted 2026-06-28) — 108 long-horizon workflows,
median 1.6 human-hours each, averaging **318 tool calls** vs ~30 in OSWorld 1.0:

> "Claude Opus 4.8 with maximum thinking and batched tool calls scores best but still completes
> only **20.6% of tasks** at a 54.8% partial score; GPT-5.5 is far more token-efficient yet
> plateaus near 13%."

Named failure modes:

> "rather than stumbling on basic GUI control or coding, they **lose track of constraints, miss
> information that arrives mid-task, guess rather than ask the user, and skip verification**,
> struggling most when a task hinges on hidden state they must recover."

**Design consequence:** decompose into short, individually-verified steps, each producing a
visible artifact. Never build a multi-hour unattended GUI marathon — it fails on stage.

### Thresholds that HAVE crossed (all verified 2026)

1. **Multimodal document reading is no longer the bottleneck.** n8n Community, 2026-06-12:
   *"The gap really isn't OCR anymore. Multimodal models read documents fine. The gap is that
   'raw text → generic JSON' leaves all the business logic to you."* Every PDF-reading
   candidate below was infeasible in the RPA era and is now viable.
2. **Cross-model disagreement as a free confidence score.** Same source: run a document through
   two models, diff field-by-field, agreement → auto-accept, disagreement → route to human.
   A principled escalation gate with no training required. **Directly serviceable to the
   Architectural Discipline criterion (30%).**
3. **Action-capable agent tools rose from 24% → 65% of usage in 16 months** (UK AI Safety
   Institute, 177,000 tools, reported 2026-07-07).

### Silent failure is the dominant operational risk

n8n Community, 2026-08-05: *"A step ran. It returned successfully. It returned nothing... Eight
rows became zero rows. The agent got empty context, answered from its own priors, and the
execution finished green. Nothing threw, so nothing alerted. **It was wrong for two days before
a human noticed.** Empty is not an error, and that is exactly the problem."*

Both the OSWorld 2.0 authors ("skip verification") and this practitioner independently identify
the same gap. **Read-back verification must be built in, and showing it on camera is free
points under Architectural Discipline.**

---

## 3. Cross-agent corroboration

A problem found independently in multiple source domains is far stronger evidence than one
found in a single thread.

| Problem | A (Reddit) | B (HN/YC) | C (Vertical) | D (Whitespace) | Verdict |
|---|---|---|---|---|---|
| **FOIA / public records** | — | #7 | **#1** | — | Corroborated from both sides — requester (B) and agency (C) |
| **Bookkeeping uncategorized txns** | **#2** | #5 | — | — | Pain corroborated; **saturation also corroborated** |
| **Freight carrier vetting** | **#1** | — | #7 SATURATED | #8 partial | **Direct conflict — see §5** |
| **AP / invoice exception chase** | #4 | — | — | **#4** | Corroborated |
| **Medical claim denial appeals** | — | #6 | #5 | — | Pain corroborated; both flag closed systems + funding |
| **Security questionnaires** | — | **#2** | — | REJECTED saturated | Conflict — see §5 |
| **Subcontractor COI chasing** | **UNVERIFIABLE** | — | #2 (evidence gap admitted) | REJECTED saturated | **Killed — see §4** |

---

## 4. Consensus kill list

Do not spend design time on any of these.

### Killed — saturated
- **Dev-tooling agents** — coding agents, sandboxes, agent observability, prod-alert triage,
  code review, meeting notes. ~50% of YC S25 was AI agents; W26/S26 Launch HNs are dominated
  by dev infra. (B)
- **Regulatory-change monitoring for finance** — Norm AI raised a **$120M Series C in July
  2026** (Khosla, Blackstone, Vanguard, BCV, Coatue); Dili $15M; Iridius $8.6M. (B)
- **Construction submittal-register generation** — Procore ships "Submittal Builder", Autodesk
  ships "AutoSpecs". The two dominant platforms own it. (C)
- **Accounting client-document chasing** — Thomson Reuters acquired SafeSend for a reported
  $600M (Jan 2025); Bright already ships a "Document Collection agent". (C)
- **Property-management maintenance triage** — Latchel has a *free* core tier. (A)
- **Chargeback representment** — Chargeflow installs free, takes 25% of recovered funds only. (A)
- **GovCon SAM.gov opportunity matching** — four separate hobbyists launched this exact tool
  within two months of 2026. (A)
- **Restaurant multi-platform menu sync** — Deliverect, Otter, Chowly; also mechanical, not
  judgment. (A)

### Killed — closed systems (fails G1/feasibility)
- **Prior authorization submission** — deliberately phone/fax-gated. One practitioner:
  *"Prior authorization needing to be done on the phone is a feature, not a bug."* (B, C)
- **Provider credentialing / CAQH** — every write is behind MFA'd payer portals. (C)
- **Research admin / IRB / grant effort reporting** — lives inside Workday, Kuali, Cayuse. (C)
- **K-12 intervention documentation** — PowerSchool / Infinite Campus, district-gated. (A, C)
- **Customs ISF filing** — ABI filing is a licensed, bonded, regulated write. (C)
- **Insurance subrogation** — requires the carrier claim file. (C)

### Killed — fails the task test
- **Long-horizon autonomous GUI operation** — the trap. 20.6% completion at 500 steps. (D)
- **Reference checks** — being *abandoned as a practice*: *"That is an antiquated practice"*,
  *"Reference checks are bogus."* Don't automate a task people are deleting. (A)
- **SSL certificate renewal** — ACME solved it; *"There's no reason to be manually updating SSL
  certs in 2026."* Also rule-based, not judgment. (A)
- **PDF bank statement rekeying** — format conversion, not multi-step judgment. Solved
  in-thread with Power Query. (A)
- **Subcontractor COI chasing** — **three-way kill.** Agent A actively searched for it and
  found nothing organic: *"Could not verify... I will not manufacture evidence for it."*
  Agent C ranked it #2 but admitted *"I was unable to find a dated 2025/2026 first-person
  practitioner complaint... outside Reddit."* Agent D rejected it as saturated (myCOI, BCS,
  Jones, SmartCompliance). Pain unproven, market crowded.

---

## 5. Genuine disagreements between agents

Surfaced rather than resolved, because both sides have real evidence.

### Freight carrier vetting — A ranked #1, C called it the most-solved on its list
**A's case:** On **2026-05-14** the Supreme Court decided *Montgomery v. Caribe Transport II*,
holding unanimously that brokers can be sued under state law for negligent carrier selection
and that FAAAA does not preempt. The vetting record became litigation evidence overnight — a
dated legal catalyst creating fresh demand. The incumbent is quoted on Reddit at **$1,500/mo**
(*"pretty significant expense for a new brokerage"*) and **rejects small applicants**
(*"Highway rejected our request. I am shocked!"*). Brokers call it *"security theater"* and
fall back to hand-made fillable PDFs. Data sits behind the **free FMCSA QCMobile API**.
**C's and D's case:** Highway, MyCarrierPortal, RMIS, Carrier411, DAT CarrierWatch, plus 2026
entrants CarrierOwl, DOTScreener, VettedHaul, FleetGen. *"A hackathon judge in logistics will
name three incumbents instantly."*
**Resolution:** Both are right about different things. The *lookup* is thoroughly solved; the
*judgment + timestamped evidence artifact* is not. A also flags a real hazard: FMCSA moved to a
new registration system ("Motus") in May 2026 with randomized 8-digit numbers, breaking tools
still reading old SAFER data.

### Security questionnaire response — B ranked #2, D rejected it outright
**B's case:** a 173-point, 138-comment HN thread (2026-05-15) of small vendors priced out of
Vanta (~$10k/yr entry, $30–65k all-in first year). Differentiator would be answering from
*live* infrastructure reads rather than a document library.
**D's case:** Loopio, Responsive, AutoRFP, SiftHub, ResponseHub, AnswerPath all shipping in
2026. Plus an open-source entrant (CAOS) landed 2026-08-10.
**Resolution:** D is right that the category is crowded. B's wedge (live evidence vs. answer
recycling) is real but narrow, and it would have to *be* the entire demo.

---

## 6. Finalists

Scored against the §10 gates. G1 (compliance) is neutral across all — every candidate can be
built on Gemini + a Google agent framework + Cloud Run/Firestore.

### F1 — Public-records / FOIA request handling *(Agent C #1, Agent B #7)*
**Workflow:** request arrives by email → classify scope, flag ambiguity → look up jurisdiction
statute, compute exact statutory deadline → identify custodian departments → **send** tailored
routing emails with due dates → **chase** non-responders → on receipt, run exemption analysis
per document → produce **redacted PDFs plus a redaction log citing statute per redaction** →
draft and **send** the response letter with fee estimate → update the request register.

| Gate | Assessment |
|---|---|
| G2 Taskmaster fit | **Excellent** — chase loop is the core; 5+ real side effects |
| G3 Deliverable in 14d | Good — no integrations to negotiate |
| G4 Result visible | **Excellent** — routing emails, redacted PDF, redaction log, response letter, register |
| G5 Architecture | **Excellent** — natural multi-stage decomposition, explicit state, human sign-off gate |
| G6 Cost | Good — document-bounded, no hot polling |

**Why unsolved:** GovQA, NextRequest, JustFOIA, FOIAXpress are **tracking and portal systems**.
They log, ticket, and alert. Scoping, custodian identification, exemption analysis, redaction
decisions, and drafting the legally-defensible letter remain human in all of them. Small
agencies run on Outlook plus a spreadsheet.
**Evidence:** Poynter, 2026-07-09 (verified): *"There's always been an average of at least a
three-month (waiting period). It's gotten way worse."* — Jason Leopold, Bloomberg.
**Weaknesses:** Agent C's clerk-side evidence is thinner than its requester-side evidence.
Wrongful redaction/release is a legal liability, so this must be framed as drafting with human
sign-off. Government sales cycles are irrelevant to a hackathon but relevant to the pitch.

### F2 — Freight carrier vetting + evidence dossier *(Agent A #1)*
**Workflow:** carrier email + COI arrives → query free FMCSA QCMobile API for authority, safety
rating, insurance history → parse COI PDF, cross-check insurer/policy/limits/dates against
FMCSA's L&I record → compare contact email/phone/domain against FMCSA-listed contacts, flag
double-brokering tells → apply the broker's written risk policy → **approve/hold/reject with
stated reasons** → write a timestamped signed dossier PDF → file under load number → send rate
confirmation or a rejection/more-info request.

| Gate | Assessment |
|---|---|
| G2 Taskmaster fit | Very good — clear judgment step, real side effects |
| G3 Deliverable in 14d | **Excellent** — free public API, best data access in the study |
| G4 Result visible | **Excellent** — dossier PDF, sent email, tracker row; plant a bad carrier to prove judgment |
| G5 Architecture | Good |
| G6 Cost | **Excellent** — free API, low token volume |

**Fresh catalyst:** SCOTUS, 2026-05-14 (see §5).
**Weaknesses:** Most contested candidate. Five startups entered in summer 2026. FMCSA "Motus"
migration is a live correctness hazard. Two of Agent A's supporting threads were themselves
founder research.

### F3 — Construction bid-invitation intake and triage *(Agent D #1)*
**Workflow:** high-volume inbox → download plan/spec sets → extract bid dates, scope, equipment
schedules, engineer of record → decide bid/no-bid against a written policy → write tracker row
→ create calendar event at deadline minus lead time → file PDFs to a named folder → send
go/no-go summary to the estimator.

**Evidence is unusually clean:** a *dated job posting* — Brady Services "Bid Coordinator",
Greenhouse, first published **2026-07-02** — whose listed duties are literally the workflow:
*"Manage a high volume Microsoft Outlook inbox... Download, organize, and review mechanical
drawings, specifications... Extract relevant project information including bid dates, project
scope, equipment schedules, engineer of record, contractors... Track bid deadlines."*
Someone is paying a salary for exactly this, right now.

| Gate | Assessment |
|---|---|
| G2 Taskmaster fit | Very good |
| G3 Deliverable in 14d | **Excellent** — email + PDF + sheets + calendar |
| G4 Result visible | **Excellent** — named files, tracker row, calendar entry |
| G5 Architecture | Good |
| G6 Cost | Moderate — large PDFs, needs chunking discipline |

**Why unsolved:** vendors in construction sell takeoff software, not intake triage. Extraction
targets are semantic, not positional — no template exists.
**Weakness:** bidding platforms (BuildingConnected, SmartBid, iSqFt) are login-gated and
ToS-hostile. Demo on emailed attachments only.

### F4 — Marketplace compliance-document gap closure *(Agent C #3)*
**Least-solved candidate found.** Amazon's dashboard tells you a document is missing and gives
you no help getting or validating one. Served today by testing labs and hourly consultants.
**Best single practitioner quote in the entire study** — Amazon Seller Forums, ~April 2026,
verified by direct fetch: *"We have an ASIN that has been stuck in compliance document review
for over 5 weeks... We currently have approximately 800 units of inventory at risk of removal."*
**Fatal-ish gate:** SP-API does not expose compliance-document upload — that is Seller Central
UI only. The agent assembles and validates the bundle but **cannot submit**. Ends one click
short of a real side effect, which directly weakens the 40% criterion.

### F5 — IT offboarding across the non-SSO long tail *(Agent A #3)*
Discovers unfederated accounts from OAuth grants + expense data, reasons about likely owners,
revokes what it can via API, chases the rest by email, closes with a written evidence record.
Incumbents (BetterCloud, Torii, Zluri, Nudge Security) are all quote-based enterprise pricing —
wrong for the 1–3 person IT team with the worst version of the problem.
**Weakness:** the community's stock answer is "fix it upstream with SSO everywhere," so you're
building for messy reality rather than correct architecture. Destructive actions need guardrails.

### F6 — Bookkeeping uncategorized-transaction resolution *(Agent A #2, Agent B #5)*
**Strongest and most repeated human evidence in the study** — May 2025 through July 2026 across
r/Bookkeeping and r/Accounting, with practitioners spontaneously describing the exact reasoning
step: *"Hardware Store transactions in the context of a Landscaping Business is most likely
COGS. But Hardware Store transactions in the context of an Ice Cream shop is most likely an
Expense."* (2026-04-08)
The unbuilt step is **research-before-asking** — look the merchant up, reason about it in the
context of *this* business, and shrink the client question list to near zero. Existing tools
optimize the asking rather than eliminating it.
**Weakness:** most commercially crowded candidate. Uncat is $9/client/mo — "cheaper" is not a
wedge. Auto-posting entries carries real accuracy/liability concerns, and bookkeepers argue
openly about whether educated assumptions are acceptable practice at all.

---

## 7. Recommendation

**F1 (public records) is the strongest fit for this specific competition**, for reasons that
are about the judging rubric rather than about market size:

1. **It is the only finalist with no gated system anywhere in the loop.** Every other candidate
   has a wall — Amazon's upload UI, bidding-platform ToS, QBO liability, FMCSA's migration.
   Under a 14-day clock, an unblocked path is worth more than a bigger market.
2. **It produces the most distinct verifiable artifacts** — routing emails, redacted PDFs, a
   redaction log citing statute per redaction, a response letter, an updated register. The 30%
   Demo criterion is scored on exactly this.
3. **It decomposes naturally into the short verified steps** the OSWorld 2.0 ceiling demands,
   with an obvious human sign-off gate that demonstrates the "escalate rather than guess"
   behavior the rubric rewards under Architectural Discipline.
4. **It was corroborated from two independent directions** — requester-side (B) and
   agency-side (C).

**F2 is the strongest alternative** if we want a sharper "why now" story: a dated Supreme Court
decision three months ago created the demand, and the data is a free government API. The cost
is walking into the most contested space in the study.

**F3 is the safest build** — cleanest feasibility, no incumbent, and a dated job posting as
proof someone pays a salary for the work. The cost is the least compelling narrative.

### Cross-cutting design commitments (apply to whichever we pick)
- Build the **chase loop** as the core, not as a feature (§1).
- Decompose into **short, individually-verified steps** (§2).
- **Read-back verification** on every write — guard against the green-but-empty failure.
- **Cross-model disagreement** as the human-escalation gate.
- Agent produces a **reviewable artifact with per-claim provenance**; escalates rather than
  guesses. HN consensus in 2026 is that agents are not trusted to act unsupervised — design
  with that, not against it.

---

## 8. Open decision

- [ ] Pick the problem: F1, F2, F3, or another from §6
- [ ] Re-verify the key quotes for the chosen candidate before they enter any narrative (§0)
- [ ] Then: architecture and code design — **Opus 5, high effort** per §11 rule 4
