# How We Shipped 3 Full-Stack Data Platforms in One Session Using Microsoft Fabric + Azure AI Foundry

## The problem: analytical platforms are slow to build. Here's how we changed that.

---

Building data-intensive analytical platforms — whether for equity research, battery supply chain management, or mining intelligence — traditionally means weeks of infrastructure work before you write a single line of actual business logic. Data pipelines, ETL orchestration, LLM integrations, lakehouse schemas, cost benchmarking engines — each one a project in itself.

Over the past few sessions, we shipped **three complete platforms** — a SEC earnings workbench with peer batch processing, a full battery value chain ERP, and a mining intelligence platform — all on the same stack: **Microsoft Fabric + Azure AI Foundry**.

The result wasn't just speed. It was a repeatable pattern that let us go from zero to production-grade code consistently. Here's how.

---

## The Stack: Why Fabric + AI Foundry Changes Everything

Most analytical platforms cobble together infrastructure from a dozen different services: S3 buckets for storage, Airflow for orchestration, LangChain for LLM calls, a separate BI tool for dashboards, another service for scheduling. Each integration is a potential failure point, each authentication layer a maintenance burden.

**Fabric collapses this into a single plane.** Delta Lake tables live in the Lakehouse. PySpark notebooks run directly against those tables. The AI Foundry endpoint speaks the OpenAI protocol, so any `openai`-compatible client works. There's no separate ETL layer — the notebook *is* the pipeline. There's no separate model deployment — the AI endpoint *is* the deployment.

The stack looks like this:

```
┌─────────────────────────────────┐
│         Fabric Lakehouse        │
│   (Delta Tables — 11 to 14)     │
├─────────────────────────────────┤
│       Fabric Notebooks          │
│   (PySpark + Python, # %% cells)│
├─────────────────────────────────┤
│       Azure AI Foundry          │
│   (GPT-4o / Kimi K2.6, OpenAI)  │
├─────────────────────────────────┤
│       External APIs             │
│   (AlphaVantage, FRED, SEC EDGAR)│
└─────────────────────────────────┘
```

Three layers. One authentication model. Zero infrastructure glue code.

---

## Platform 1: SEC Earnings Workbench + Peer Batch Processing

**What it does:** Automated equity research pipeline that ingests SEC filings (10-K, 10-Q), runs three AI agents (Fundamentals, Diligence, Markets), synthesizes findings through a CHP adjudicator, and outputs structured research artifacts.

**Then we added peer batch processing** — running the entire pipeline across a primary company and N peers, with rate-limit enforcement for free-tier APIs (AlphaVantage: 5 req/min), exponential backoff on AI calls, and a Comparative Analysis Agent that produces cross-company intelligence.

**Key metrics:**
- 8 Delta tables (sec_filings, company_fundamentals, macro_indicators, research_sessions, agent_outputs, research_artifacts, audit_trail, peer_comparisons)
- 3 AI agents + 1 comparative agent
- Rate limiting: 15-second delays between AlphaVantage calls, 3-attempt retry with backoff
- Full pipeline runs per company: data ingestion → AI analysis → CHP hardening → artifact generation → comparative synthesis

**The efficiency:** The peer batch notebook (`fabric_peer_batch.py`) wraps the single-company pipeline into a reusable `process_company()` function. Adding a new peer to the batch is one line in a config dict. The Comparative Analysis Agent at the end automatically generates side-by-side metrics, relative value rankings, and pair trade suggestions.

---

## Platform 2: Battery ERP

**What it does:** End-to-end battery value chain management — from raw materials (lithium carbonate, nickel, cobalt, graphite) through cell chemistry (NMC-111, NMC-811, NMC-622, NCA, LFP, LMO) to battery packs.

**The domain model covers 11 entities:** RawMaterial, CellChemistry, BatteryCell, BatteryPack, BOMItem, Supplier, InventoryRecord, PurchaseOrder, ManufacturingBatch, PriceHistory, CostScenario.

**Business logic includes:**
- BOM cost rollups from materials → cells → packs
- Supplier scoring (composite 0-100 with A/B/C/D grades)
- Dual-sourcing strategy recommendations
- Cost scenario modeling (lithium shock, cobalt restriction, nickel recovery)
- Inventory status and reorder logic
- Batch yield calculations

**Key metrics:**
- 11 Delta tables with realistic seed data
- 20+ materials tracked across 6 chemistries
- 32 unit tests, all passing
- Two Fabric notebooks: lakehouse setup (13 cells) + cost dashboard (14 cells)

**The efficiency:** The pricing engine pulls live commodity prices from AlphaVantage and FRED, feeds them into the BOM rollup, and the cost dashboard notebook automatically produces margin analysis per chemistry and sensitivity tables per scenario. What would normally require a team of analysts with spreadsheets happens in a single notebook run.

---

## Platform 3: Minescope.Signal

**What it does:** Mining intelligence platform that extracts actionable signals from commodity pricing, mineral reserves, production data, and AISC (All-In Sustaining Cost) benchmarks. Includes AI-powered comparative analysis across mining companies.

**The signal scoring system** is the core innovation — a weighted composite score (0-100) across five dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Grade | 25% | Reserve quality (avg grade, proven/probable ratio) |
| Cost | 25% | AISC percentile rank (lower = cheaper = better) |
| Production | 20% | Guidance beat rate, recovery efficiency |
| Growth | 15% | Reserve-to-resource conversion potential |
| ESG | 15% | Environmental, social, governance score |

Scores map to ratings: **Strong Buy** (80+) → **Buy** (65-80) → **Hold** (50-65) → **Underperform** (35-50) → **Sell** (<35).

**Key metrics:**
- 6 domain models with NI 43-101 / JORC reporting compliance
- 5 services: Pricing, Reserves, Production, AISC, Intelligence Orchestrator
- 14 Delta tables seeded with 8 mining companies, 14 mine sites, 18 reserve estimates
- 64 tests, all passing
- AI Foundry agents: per-company intelligence briefing + cross-company comparative ranking

**The efficiency:** The reserve service automatically aggregates tonnage and contained metal across classification tiers (Proven/Probable/Measured/Indicated/Inferred), calculates NPV sensitivity across commodity price scenarios, and compares reserve profiles across companies — all from the same Delta table schema.

---

## The Patterns That Made It Fast

### 1. Notebook-as-Pipeline

Fabric notebooks aren't just for exploration. With `# %%` cell markers, they become version-controllable, reproducible pipeline definitions. Each notebook has a clear sequence: config → data load → business logic → AI inference → Delta write → display. No Airflow DAGs, no Kubernetes pods, no YAML configuration hell.

### 2. Delta Tables as the Integration Layer

Instead of passing data between microservices via API calls or message queues, everything writes to Delta tables. The reserve service writes to `reserve_estimates`. The AISC service writes to `aisc_metrics`. The intelligence orchestrator reads from all of them. The AI agent reads the joined view. No serialization, no schema negotiation, no version drift.

### 3. AI Foundry as a Utility

The OpenAI-compatible client means swapping between GPT-4o, Kimi K2.6, or any other deployed model is a single config change. No separate model serving infrastructure. No cold starts. The AI is just another function call in the notebook — `client.chat.completions.create()` with a system prompt, and you get structured analytical narrative back.

### 4. Domain Models First, Infrastructure Second

Every platform started with the domain model — the dataclasses that represent the actual business entities. MiningCompany, ReserveEstimate, AiscMetric. BatteryCell, BOMItem, Supplier. These are pure Python with zero framework dependencies. They can be tested in isolation, reasoned about locally, and then plugged into the Fabric pipeline. The infrastructure exists to serve the domain model, not the other way around.

### 5. Seed Data in the Setup Notebook

Every setup notebook includes realistic seed data — actual mining company financials, real commodity prices, real battery chemistry BOMs. This means the moment the Lakehouse is initialized, you have a working demo. No "please configure your data source" blank-screen experience.

---

## What This Means for Analytical Teams

The traditional model for building analytical platforms is: **months of infrastructure, then weeks of business logic, then maybe some AI integration if there's budget left.**

The Fabric + AI Foundry model inverts this: **minutes of infrastructure (Delta tables), days of domain modeling, and AI is baked in from the start.**

For teams building equity research tools, supply chain analytics, commodity intelligence, or any domain where structured data meets narrative analysis, this stack eliminates the friction between "having the idea" and "shipping the product."

The three platforms we built — SEC research, battery ERP, mining intelligence — share almost no business logic. They're in completely different domains. But they share the same architectural DNA: Fabric Lakehouse for storage, PySpark notebooks for orchestration, AI Foundry for intelligence, and clean Python domain models for business logic.

**That's the pattern. And it's repeatable.**

---

*All code is open source:*
- [sec-earnings-workbench](https://codeberg.org/cubiczan/sec-earnings-workbench) — SEC equity research pipeline + peer batch
- [battery-erp](https://codeberg.org/cubiczan/battery-erp) — Battery value chain ERP
- [minescope-signal](https://codeberg.org/cubiczan/minescope-signal) — Mining intelligence platform

*Stack: Microsoft Fabric · Azure AI Foundry · Python · Delta Lake · AlphaVantage · FRED*
