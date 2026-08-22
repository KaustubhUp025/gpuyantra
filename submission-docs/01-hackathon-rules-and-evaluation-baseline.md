# All Things Agentic Hackathon — Rules, Constraints & Evaluation Baseline

**Status:** Authoritative reference document. Every product, architecture, and code decision for this
submission is evaluated against this file.
**Organizer:** Google (administered by Devpost)
**Sources:**
- Overview — https://allthingsagentichackathon.devpost.com/
- Rules — https://allthingsagentichackathon.devpost.com/rules
- Resources — https://allthingsagentichackathon.devpost.com/resources
- Organizer clarification, demo video speed-up —
  https://allthingsagentichackathon.devpost.com/forum_topics/44809-demo-video-is-speeding-up-the-whole-recording-allowed-under-unedited

**Captured:** 2026-08-17
**Chosen track:** The Taskmaster
**Entry type:** Solo individual entrant

> ⚠️ This document is a working transcription of the official rules for internal decision-making.
> The Devpost pages are the legally binding source. Re-verify before final submission.

---

## 1. Timeline (all times PT unless noted)

| Milestone | Date/Time | Days from 2026-08-17 |
|---|---|---|
| Contest / Submission period opens | Aug 3, 2026, 9:00 AM PT | — (already open) |
| ~~Google Cloud credits request deadline~~ | ~~Aug 28, 2026, 12:00 PM PT~~ | ✅ **Already requested** |
| **Submission deadline (hard)** | **Aug 31, 2026, 5:00 PM PDT** | **14 days** |
| Judging period | Sep 1 – Oct 1, 2026 | — |
| Winners announced | Oct 8, 2026, 10:00 AM PT | — |

**Implications:**
- Effective build window is **~14 calendar days**. Scope must be sized to finish, deploy, and
  record a live demo with buffer — not sized to be maximally ambitious.
- Credits form **already submitted** (confirmed 2026-08-17) — no longer a blocker.
- **No edits are possible after the submission period closes.** Everything — repo, README,
  architecture diagram, video, hosted URL — must be final by Aug 31, 5:00 PM PDT.
- Drafts can be saved on Devpost before submitting. Submit early, refine the draft, don't
  race the clock.

---

## 2. Mandatory technical stack (pass/fail gate)

A submission that misses **any** of these three fails Stage One judging outright.

1. **Gemini 3.5 or newer**, accessed via the **Gemini API** or **Vertex AI**.
2. **At least one Google Agent Framework:** Google ADK, GenAI SDK, Antigravity SDK, or Genkit.
3. **At least one Google Cloud infrastructure service:** e.g. Cloud Run, Cloud SQL, Firestore,
   GKE, Pub/Sub.

**Decision rule:** Any proposed component that would replace one of these three with a
non-Google equivalent is automatically rejected. Non-Google services may be used *in addition*
(as tools/integrations the agent acts upon), never *instead of*.

---

## 3. Track definitions

### 3.1 The Taskmaster — **our track**
> "Build complete workflows beyond chatbots. Make one that takes action. Find a messy,
> multi-step chore… Build an agent that handles the details, sends the right info to the
> right places."

Judged specifically on: **multi-step background workflow completion without human
intervention.**

**Decision rule (the Taskmaster test):** For any feature we consider, ask —
*"Does the agent complete this chore end-to-end on its own, and write the result somewhere real?"*
If the answer is "it produces text for the user to then act on," it is a chatbot feature and
does not count toward the 40% criterion. Prefer autonomous execution + real side effects
(writes to systems, sends messages, files records) over conversation.

### 3.2 The Collaborative Partner (not chosen)
Agent leads step-by-step, captures feedback, adapts. Judged on active data
synthesis/mutation and ingestion of complex unstructured data.

### 3.3 The Fortified Enterprise Fleet (not chosen)
Scalable multi-agent institutional systems: Discovery & Lifecycle, Core Execution & State,
Security & Governance, Telemetry. Judged on whether multi-agent complexity is *justified*
and delegation is intelligent.

> The Sponsor reserves the right to **reassign a submission's category** at its discretion.
> Our positioning must be unambiguously Taskmaster so we are judged where we intend.

---

## 4. Judging model

**Final score range: 1 to 6** (Stage Two max 5, plus up to +1.0 of Stage Three bonus).

### Stage One — Viability (pass/fail)
- Includes all required submission elements.
- Reasonably addresses a challenge.
- Properly applies the specified required technologies.

### Stage Two — Weighted criteria (each scored 1–5)

| Weight | Criterion | What judges look for |
|---:|---|---|
| **40%** | **Innovation & Operational Utility** | How much real-world friction the agent removes *on its own*. Autonomous, high-value action is rewarded over simple chat. For Taskmaster: multi-step background workflow completion without intervention. |
| **30%** | **Architectural Discipline & Tech Stack** | System decoupling; state management; failure tolerance & recovery logic; tool isolation and security scoping of credentials; data architecture efficiency for large contexts; separation of concerns across agent workflow. |
| **30%** | **Demo & Production Readiness** | "Proof of Action" — an **unedited, live execution** of the agent performing its task on video. Architecture clarity, documentation/repo quality, reproducible setup, visual proof of Google Cloud deployment. |

### Stage Three — Bonus (max **+1.0**)

| Bonus | Value | Action |
|---|---:|---|
| Public content (blog / podcast / video) about building the project | +0.2 | Write a build blog post; publish publicly. |
| Social post with **#AllThingsAgenticHackathon** on X, LinkedIn, Instagram, or Facebook | +0.2 | Post with the hashtag before deadline. |
| Integrate additional Google AI models (Gemma, Veo, Lyria) | +0.2 each, **+0.6 max** | Only if genuinely useful to the workflow. |

**Decision rule:** The bonus is worth up to **1 full point out of 6 (~17%)** for a few hours of
work. All three are cheap relative to their weight and must be treated as required
deliverables, not optional extras — *provided* the extra models serve the product rather than
being bolted on.

### Tie-breaking
Compared criterion-by-criterion in the listed order (Innovation first, then Architecture,
then Demo), then judge vote. Judges' determinations are final and binding.
**Implication:** when trading off, protect the Innovation & Operational Utility score first.

---

## 5. Prizes

Total pool **$180,000**. Each project is eligible for a **maximum of one prize**.

| Category | Qty | Cash | GCP credits |
|---|---:|---:|---:|
| Grand Prize | 1 | $50,000 | $5,000 |
| The Taskmaster | 1 | $20,000 | $2,000 |
| The Collaborative Partner | 1 | $20,000 | $2,000 |
| The Fortified Enterprise Fleet | 1 | $20,000 | $2,000 |
| Startup Excellence (incorporated orgs only) | 1 | $20,000 | $5,000 |
| Individual / Hobbyist — Best Build | 2 | $10,000 | $1,000 |
| Best Architectural Design | 2 | $5,000 | $1,000 |
| Best Multimodal UX | 2 | $5,000 | $1,000 |
| Honorable Mentions | 5 | $2,000 | $500 |

**Relevant to us as a solo entrant:** Grand Prize, Taskmaster category, **Individual/Hobbyist
Best Build** (2 awarded — strong realistic target), Best Architectural Design, Best Multimodal
UX, Honorable Mentions. Startup Excellence is not available (requires an incorporated
organization).

**Decision rule:** Two of the reachable prizes are *Best Architectural Design* and *Best
Multimodal UX*. A clean, decoupled architecture and a genuinely multimodal interface are not
just criteria points — they are separate prize surfaces. Weight design work accordingly.

---

## 6. Submission checklist (all mandatory unless noted)

- [ ] Project built with the required Google tools (§2)
- [ ] **One** category selected — Taskmaster
- [ ] Hosted project URL / working demo (*highly encouraged*; if private, supply credentials in
      testing instructions)
- [ ] Text description: features, technologies used, data sources, findings/learnings
- [ ] Public code repository (GitHub / GitLab / Bitbucket)
      - If private, grant access to **testing@devpost.com** and **cloudhackathons@google.com**
- [ ] **README.md with spin-up instructions** and reproducibility steps
- [ ] **Architecture diagram** showing system design
- [ ] **Demo video, ≤ 4 minutes**, public on **YouTube or Vimeo**, in English (or English subtitles)
  - [ ] Short overview of the problem being solved
  - [ ] Value proposition
  - [ ] Live demo of the application in action
  - [ ] **Visual proof the backend runs on Google Cloud** — Cloud Console, Cloud Run dashboard,
        Vertex AI logs, a `*.run.app` URL, etc.
- [ ] Application supports English at minimum
- [ ] Bonus: public build content (+0.2)
- [ ] Bonus: social post with `#AllThingsAgenticHackathon` (+0.2)
- [ ] Bonus: additional Google AI model integration (+0.2 each, +0.6 max)

### 6.1 Demo video — "unedited" is not a scope constraint

Organizer confirmed
([thread 44809](https://allthingsagentichackathon.devpost.com/forum_topics/44809-demo-video-is-speeding-up-the-whole-recording-allowed-under-unedited),
Shawni / Devpost Manager) that a **uniform speed-up of one continuous live run** counts as
"unedited"; cuts, splicing, and trimming do not. Record a single take, apply one speed
multiplier, note the speed on screen.

Video production and editing are handled by the entrant and are **not** a factor in problem
selection or architecture. Runtime does not meaningfully limit workflow ambition.

**Video content restrictions:** nothing derogatory, offensive, threatening, defamatory,
disparaging, libelous, inappropriate, indecent, sexual, profane, tortious, slanderous, or
discriminatory. No third-party advertising, logos, or endorsements outside the spirit of the
contest. No violation of third-party publicity, privacy, or IP rights.

**Functionality requirement:** *"The Project must be capable of being successfully installed
and run consistently on the platform for which it is intended, and must function as depicted
in the video and/or expressed in the text description."*
→ The demo must show what the system actually does. No mocked or staged capability.

---

## 7. Code provenance & IP rules

- **Projects must be newly created during the Submission Period (Aug 3 – Aug 31, 2026).**
  *"The work described and submitted must have been built during the Submission Period."*
- **Allowed without disclosure:** standard development tools — frameworks, libraries, starter
  templates, and **AI coding assistants**.
- **Must be disclosed:** any *other* pre-existing code or work incorporated into the project.
- **Ownership:** submissions remain the IP of the entrant. Entrant warrants the work is
  original, solely owned, with no third-party rights violated (copyright, trademark, patent,
  contract, privacy).
- **Third-party technical assistance** is permitted only if the submission components are
  solely the entrant's work product, the result of the entrant's ideas and creativity, and the
  entrant owns all rights.
- **Open source** may be used if the entrant complies with the applicable licenses **and**
  creates software that *enhances and builds upon* the underlying open source product.
- **Third-party SDKs/APIs/data** require the entrant to be authorized under those tools' terms
  and licensing requirements.
- Project must **not** have been developed with financial or preferential support from the
  Sponsor or Administrator.
- **License to Google:** entering grants Google a perpetual, irrevocable, worldwide,
  royalty-free, non-exclusive license to use, reproduce, adapt, modify, publish, distribute,
  publicly perform, create derivative works from, and publicly display the project, for
  evaluation and promotion. (Generally commercially available software not owned by the
  entrant is excluded from this grant.)
- **Multiple submissions** allowed, but each must be unique and substantially different.

### 7.1 Working rules derived from this section
1. **All code in the submission repo must be authored fresh within the submission window.**
   No lifting from prior personal projects unless explicitly disclosed.
2. **AI coding assistants are explicitly permitted and require no disclosure.** Use of Claude
   Code is compliant with the rules as written.
3. **The repository is pushed with the entrant as sole contributor.** No AI co-author trailers,
   no assistant attribution in commits, README, or docs.
4. Every third-party dependency must have a license compatible with our use, and its terms must
   permit hackathon/commercial-style use. Verify before adopting.

---

## 8. Eligibility & disqualification

**Eligible:** above the age of majority in country of residence (min. 20 in Taiwan). Individual,
team, or organization entries.

**Ineligible:** residents of Italy, Quebec, Crimea, Cuba, Iran, Syria, North Korea, Sudan,
Belarus, Russia, or US-sanctioned countries; anyone under US export controls/sanctions;
employees, interns, contractors, and office-holders of Google, Devpost, or organizations
involved in the contest (and their families); government-agency employees with a conflict of
interest.

**Employment clause:** if entering in connection with employment, the employer must have full
knowledge and consent, and the entrant warrants compliance with employer policies.

**Disqualification triggers:** false identity/contact/rights information; cheating, deception,
or unfair play; harassing or abusing entrants, Google, or judges; tampering with the submission
process or contest site; failing to respond to winner verification within **two days**; missing
the **10 business day** Required Forms deadline; missing mandatory registration data.

**Winner verification:** identity, qualifications, and the winner's role in creating the
submission are all verified. Prize affidavits must be completed and verified before anyone is
declared a winner. Prizes delivered within 60 days of receipt of completed forms. Winner bears
all fees and taxes.

**Legal:** governed by California law; disputes resolved by binding arbitration via JAMS in
San Jose, CA. Entrants consent to promotional use of their name, likeness, photograph, voice,
comments, hometown, and country.

> **Note:** solo entry means the "role of the potential winner in the creation of the
> Submission" is verified against a single person. Keep commit history clean, consistent, and
> genuinely representative of the build.

---

## 9. Official resources & free-tier levers

| Resource | Where | Notes |
|---|---|---|
| **$150 Google Cloud credits** | Credit form on hackathon site | **Request by Aug 28, 12:00 PM PT** |
| Google Cloud free trial | cloud.google.com/free | Separate from hackathon credits |
| **Agent Development Kit (ADK)** | github.com/google/adk-python · google.github.io/adk-docs | "Fastest way to build, evaluate, and deploy agents" |
| Gemini API / AI Studio | ai.google.dev · aistudio.google.com | Models, quickstarts, multimodal guides |
| Antigravity SDK | — | Pre-packaged agent runtime |
| Genkit | — | Open-source; JS/Go/Python |
| Cloud Run | — | Serverless deployment target |
| Firestore | — | NoSQL datastore for state/memory |
| GEAP (Gemini Enterprise Agent Platform) | — | Agent Registry, Runtime, Memory Bank, identity, gateway, guardrails, observability. Aimed at the Enterprise Fleet track. |
| GEAR program | Google Skills | Free skilling, 35 monthly learning credits |
| Webinars | Aug 11, 13, 20, 27 | Multi-agent orchestration, persistent workflows, self-evolving agents, memory architecture |
| Community | Devpost Discord, hackathon discussion forum | — |

**Cost decision rule:** the budget is $150 in credits plus free tiers. Architecture must be
**cost-efficient under free-tier / low-quota model access**: batch and cache aggressively,
prefer cheaper models for high-volume sub-tasks, reserve the strongest model for reasoning
steps, and never design a hot loop that burns tokens per poll. This directly reinforces the
Architectural Discipline criterion (data architecture efficiency for large contexts).

---

## 10. Decision framework — how this document gets used

Every proposal (feature, library, service, architectural choice) is checked against these gates
**in order**. A failure at any gate kills the proposal.

| # | Gate | Question |
|---|---|---|
| G1 | **Compliance** | Does it keep Gemini 3.5+, a Google Agent Framework, and a Google Cloud service in place? Does it respect licensing and code-provenance rules? |
| G2 | **Taskmaster fit** | Does it increase *autonomous multi-step completion with real side effects*? Or is it chat surface? |
| G3 | **Deliverability by Aug 31** | Can it be built, deployed, and demoed live within the remaining window with buffer? |
| G4 | **Demonstrability** | Can the workflow be shown actually *working* on real or realistic data in one live run? Runtime is not the constraint (§6.1) — the constraint is whether the result is **visible and verifiable** on screen. Weak if the payoff is invisible or unfalsifiable. |
| G5 | **Architectural discipline** | Does it preserve decoupling, explicit state, failure/recovery handling, and scoped credentials? |
| G6 | **Cost** | Does it stay viable on $150 credits + free tiers? |

### Scoring-weight priority when trading off
1. Autonomous operational utility (40%) — protect first, also the first tie-breaker.
2. Architectural discipline (30%) — also a separate prize surface.
3. Demo & production readiness (30%) — also a separate prize surface via multimodal UX.
4. Bonus items (+1.0 of 6) — cheap; treat as required.

---

## 11. Standing operating rules for this build

These govern how work is executed on this project.

1. **Attribution:** the repository is pushed with the entrant (Kaustubh) as the **sole
   contributor**. No AI co-authorship, co-author trailers, or assistant attribution anywhere in
   commits, README, or documentation.
2. **Verify before assuming:** before adopting any library, SDK, model version, or third-party
   service, a subagent verifies against current web sources that the choice is current, correct,
   and compatible with the plan. No decisions from stale assumptions.
3. **Review before merge:** before writing code, a background senior-engineer code-review
   subagent (Opus 5) is started to check for redundancy and to confirm the code is optimized —
   specifically that performance holds up under free-tier / low-quota agent and model API usage.
4. **Model assignment:**
   - Architecture, system design, and code-design decisions → **Opus 5, high effort**
   - Code writing and testing → **Opus 4.6, medium effort**

---

## 12. Open items

- [x] Submit the Google Cloud credits request — **done 2026-08-17**
- [ ] Register on Devpost and create a draft submission early
- [ ] Confirm the exact current Gemini model ID that satisfies "3.5 or newer" at build time
- [ ] Choose the specific Taskmaster problem domain
- [ ] Decide the agent framework: ADK vs. GenAI SDK vs. Genkit vs. Antigravity
- [ ] Decide the Google Cloud service mix (Cloud Run + Firestore is the likely baseline)
- [ ] Re-read the official rules page before final submission to catch any amendments
