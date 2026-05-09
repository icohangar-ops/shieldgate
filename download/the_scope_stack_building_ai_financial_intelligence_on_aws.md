# Building an AI-Powered Financial Intelligence Suite on AWS: Three Platforms, One Architecture

## How I built REIT analytics, supply chain intelligence, and energy market platforms — all on S3 + Iceberg + Glue + Athena + Bedrock — for under $10/month each.

---

In institutional finance, the edge isn't found in a single model or dataset. It's found at the intersection of domain expertise, diverse data sources, and the ability to synthesize signals faster than the market. After years working across real estate, commodities, and supply chain analytics, I kept running into the same friction: every new analytical project required stitching together data pipelines, storage layers, compute engines, and inference systems from scratch. The overhead of infrastructure was eating the time I should have spent on analysis.

So I built a modular architecture — a pattern I call the **Scope Stack** — that could be adapted across fundamentally different financial domains: commercial real estate, global supply chains, and energy markets. Three platforms. Three domains. One repeatable AWS-native architecture.

This is the story of how Scope.Sentinel, Scope.Vantage, and Scope.Glacier came together, the engineering decisions that shaped them, and why Amazon Bedrock's Converse API became the glue between raw data and actionable intelligence.

---

## The Architecture: Why This Stack, and Why Not Something Else

Before diving into the individual platforms, let me explain the architectural decisions that underpin all three. The goal was to build production-grade analytical systems that could run on a shoestring budget while remaining extensible enough for serious institutional use.

### The Data Lake: S3 + Apache Iceberg

Every financial analytics system needs a foundational data store. The question is what properties matter most. For analytical workloads, the priorities are:

- **ACID compliance** — You need to know your data is consistent, especially when multiple ETL jobs write simultaneously.
- **Time-travel** — Financial analysis often requires looking at data as it existed at a point in time. Regulatory audits demand it.
- **Schema evolution** — Markets change. New data fields appear. Your storage format needs to accommodate this without breaking existing queries.
- **Cost efficiency** — Query engines should only read the data they need, not entire partitions.

Apache Iceberg on S3 delivers all four. Unlike raw Parquet files managed manually, Iceberg provides a metadata layer that tracks snapshots, handles schema changes gracefully, and enables efficient partition pruning. Combined with S3's tiered storage (Standard → IA → Glacier), we get hot analytics on recent data and cold storage for historical records at a fraction of traditional data warehouse costs.

### The Compute Layer: AWS Glue + Athena + Lambda

The processing pipeline follows a three-tier model:

1. **AWS Glue** handles the heavy ETL — ingesting raw data from external APIs, cleaning, normalizing, and writing to Iceberg tables. Glue jobs run on Spark under the hood, but the service is fully managed. You pay per job execution, not for idle clusters.

2. **Amazon Athena** provides the query engine. It reads directly from Iceberg tables on S3, which means there's no data duplication and no cluster to manage. Every analytical view — whether it's a REIT FFO ranking or a supply chain concentration index — is a SQL query executed on demand.

3. **AWS Lambda** provides the orchestration layer. Each platform has Lambda handlers that respond to scheduled EventBridge triggers, fetch fresh data from external APIs, and write raw records to S3. The intelligence pipeline — score computation, AI analysis, and signal persistence — runs as a Step Functions state machine that chains Lambda invocations.

### The Intelligence Layer: Bedrock Converse API

This is where the systems generate alpha. Rather than using generic `InvokeModel` calls with raw JSON payloads, all three platforms use the Bedrock **Converse API** — the native multi-turn conversation interface. The advantages are significant:

- **System prompts** are first-class citizens, not string concatenation hacks.
- **Tool use** and structured output can be layered in without switching API surfaces.
- **Multi-turn context** allows for follow-up refinement without prompt engineering workarounds.
- **Token tracking** via the `usage` response field enables cost monitoring per inference call.

The model of choice across all platforms is Claude 3 Haiku. For analytical intelligence — where the task is synthesizing structured data into narrative insights — Haiku hits the sweet spot of speed, cost, and quality. At $0.25 per 100K input tokens, a platform generating 50 intelligence briefings per day costs pennies.

### The Orchestration: EventBridge + Step Functions

EventBridge provides the scheduling heartbeat. Each platform has its own cadence:

- **Sentinel**: Daily SEC filing check (6 AM ET), weekly full analysis (Sunday 8 AM ET)
- **Vantage**: Weekly Comtrade sync (Monday 2 AM UTC), daily intelligence (6 AM UTC)
- **Glacier**: Daily EIA price sync (8 PM ET previous day), daily signal generation (2 AM ET)

Step Functions orchestrates the analysis pipeline for each platform as a deterministic state machine: ingest → compute scores → invoke Bedrock → persist signals. If any step fails, the execution is visible in the Step Functions console with full input/output tracing — critical for debugging production analytical pipelines.

---

## Scope.Sentinel: REIT Analytics Intelligence

### The Problem

Commercial real estate analysis is drowning in data but starving for synthesis. A single REIT like Prologis (PLD) generates quarterly 10-Q filings with hundreds of data points — FFO, AFFO, same-store NOI growth, weighted average cap rates, debt maturities, occupancy rates, development pipeline. Multiply that across 200+ public REITs, and you have a dataset that no human can process comprehensively on a quarterly cycle.

The existing tools fall into two categories: expensive institutional platforms (Bloomberg Terminal, Capital IQ) that cost five to six figures annually, and free tools (Yahoo Finance, SEC EDGAR directly) that provide raw data without analytical context. There's a gap in the middle — a system that ingests raw data and produces structured, scored, AI-analyzed signals.

### The Data Model

Scope.Sentinel tracks six core entity types:

1. **Reit** — The core entity: ticker, sector (Equity, Mortgage, Specialty, Infrastructure), sub-sector (Retail, Industrial, Data Center, Cell Tower, Healthcare), dividend yield, payout ratio.

2. **FinancialMetric** — Quarterly financials: FFO per share, AFFO per share, same-store NOI growth, net debt-to-EBITDA, interest coverage, weighted average cap rate.

3. **Property** — Individual property records within a REIT's portfolio: location, type, square footage, occupancy, cap rate.

4. **MarketIndicator** — Macro signals that affect all REITs: Fed Funds rate, 10-year Treasury yield, 2s10s spread, CPI, unemployment rate.

5. **SecFiling** — Filing metadata from SEC EDGAR: form type (10-K, 10-Q, 8-K), filing date, extracted key metrics.

6. **ReitSignal** — The output: a composite score with sub-scores, signal rating, AI analysis text, and confidence score.

### The Scoring Engine

The composite Sentinel Score weights five dimensions:

```
Sentinel Score =
    Fundamental Score  × 0.30  (FFO growth, NOI growth, AFFO quality)
  + Valuation Score    × 0.25  (NAV premium/discount, cap rate vs sector)
  + Momentum Score     × 0.20  (price trend, volume, relative strength)
  + Macro Score        × 0.15  (rate environment, spread curve, GDP outlook)
  + Sentiment Score    × 0.10  (insider activity, institutional flows)
```

Each sub-score maps to a 0-100 scale, and the composite maps to a rating: Strong Buy (80+), Buy (65-79), Hold (35-64), Sell (20-34), Strong Sell (0-19).

The dividend safety framework deserves special mention. It classifies REIT dividends into five tiers based on AFFO payout ratio: Very Safe (≤60%), Safe (60-75%), Moderate (75-85%), Risky (85-95%), Unsafe (>95%). This is critical because REIT investors live on distributions — a dividend cut is often a more damaging signal than a temporary earnings miss.

### The AI Layer

Claude Haiku receives a structured prompt containing the REIT's latest metrics, its sector context, and macro conditions. The system prompt instructs it to act as a senior REIT analyst. The output is a 2-3 paragraph investment recommendation covering risks, opportunities, and strategic positioning.

The key design choice: the AI doesn't compute scores. Scores are deterministic, computed from financial data. The AI provides the narrative interpretation — the "why" behind the numbers. This separation ensures reproducibility: if you run the same data through the pipeline tomorrow, you get the same scores. Only the narrative may differ, and that's appropriate for qualitative analysis.

### Infrastructure

- **3 Glue ETL jobs**: financial_etl (peer benchmarks, sector averages), market_data_etl (price history, correlations), sec_filing_etl (10-K/10-Q parsing)
- **3 Athena views**: reit_fundamentals (FFO rankings, dividend safety), property_analysis (geographic distribution, sector mix), market_correlation (rate sensitivity, macro signals)
- **2 Lambda handlers**: sec_ingestion (EventBridge-driven), analysis (Step Functions-driven)
- **2 Iceberg tables**: reits, financial_metrics

---

## Scope.Vantage: Supply Chain Intelligence

### The Problem

Global supply chains are networks of dependencies that most analytical tools treat as spreadsheets. A lithium-ion battery requires lithium from Australia and Chile, cobalt from DR Congo, nickel from Indonesia, graphite from China, and processing in multiple countries before reaching a battery factory in the US. When any single node in this chain faces disruption — a port closure, a tariff hike, a political crisis — the ripple effects propagate unpredictably.

The analytical challenge is threefold: mapping the chain itself, measuring concentration risk (how many nodes can fail before the chain breaks), and monitoring real-time disruption events (weather, geopolitical, logistics).

### The Data Model

Scope.Vantage tracks six entity types:

1. **TradeFlow** — Bilateral trade records from UN Comtrade: reporter, partner, HS6 commodity code, net weight (kg), trade value (USD), direction (Import/Export).

2. **Commodity** — Commodity reference data with HS code linkage. The platform tracks 8 critical minerals: lithium carbonate, cobalt, nickel, copper, rare earth compounds, iron ore, natural graphite, and manganese.

3. **SupplyChainNode** — A node in the supply chain graph: type (Origin Country, Processing Hub, Manufacturer, End Market), country, connections to other nodes, risk score, capacity share.

4. **TariffRegulation** — Trade policy instruments: tariffs, quotas, sanctions, embargoes. Each has an imposing country, target country, commodity scope, rate, effective date, and status.

5. **LogisticsEvent** — Real-time disruption tracking: shipments, delays, disruptions, bottlenecks. Each has severity (Low/Medium/High/Critical), estimated delay days, and a computed cost impact.

6. **IntelligenceBriefing** — The AI-generated output: scope, summary, risk assessment, opportunities, recommendations, confidence score, source references.

### The Comtrade Integration

This is the technical centerpiece of Scope.Vantage. The platform uses the `comtradeapicall` Python library — the official open-source client for the UN Comtrade API — to fetch bilateral trade flows programmatically.

The integration handles several real-world complexities:

- **HS code normalization**: UN Comtrade returns commodity codes in various formats (4-digit, 6-digit, with/without dots). The service normalizes everything to XXXX.XX format for consistent joins.

- **Unit conversion**: Trade volumes arrive in kilograms. The model provides computed properties for net_weight_tonnes, unit_value_usd_per_kg, and unit_value_usd_per_tonne.

- **Rate limiting**: The Comtrade API throttles aggressive callers. The service implements exponential backoff with a configurable minimum interval between calls.

- **Direction handling**: The same bilateral trade flow looks different from each country's perspective. The `fetch_trade_flows` method can pull Import, Export, or both directions for a country pair.

### Concentration Risk: HHI Analysis

The Herfindahl-Hirschman Index is the standard measure of market concentration. Scope.Vantage computes it per commodity, per direction (Import/Export), per year:

- **Below 1,500**: Unconcentrated — healthy supplier diversity
- **1,500 - 2,500**: Moderately Concentrated —值得关注, some dependency
- **Above 2,500**: Highly Concentrated — critical risk, single-point-of-failure exposure

For context: the global cobalt supply chain has an HHI above 3,500 when measuring exports from origin countries. DR Congo alone accounts for ~70% of mined cobalt. That's a supply chain risk that every battery manufacturer and EV company needs to quantify and monitor.

### The Scoring Engine

The composite Vantage Intelligence Score weights four dimensions:

```
Composite Score =
    Supply Risk (HHI-based concentration)    × 0.30
  + Price Volatility (30-day σ/μ)            × 0.25
  + Logistics Risk (active disruptions)      × 0.25
  + Policy Risk (active tariffs, sanctions)   × 0.20
```

### Infrastructure

- **3 Glue ETL jobs**: trade_flow_etl (Comtrade ingestion, Iceberg MERGE), supply_chain_etl (HHI computation, top-5 country breakdown), logistics_event_etl (disruption processing, cost impact, aggregate summary)
- **3 Athena views**: critical_mineral_flows (bilateral analysis with unit values), supply_chain_risk (HHI + disruptions + tariffs combined), tariff_impact (active tariffs joined with trade flow volumes for cost quantification)
- **2 Lambda handlers**: comtrade_ingestion (fetches via comtradeapicall, writes JSONL to S3), intelligence (Step Functions: compute → Bedrock → write)
- **5 Iceberg tables**: trade_flows, logistics_events, tariff_regulations, intelligence_briefings, concentration_metrics

---

## Scope.Glacier: Energy Markets Intelligence

### The Problem

Energy markets are the most geopolitically sensitive commodity markets on Earth. A single OPEC+ meeting can shift crude oil prices by 5% in an hour. A hurricane in the Gulf of Mexico can take 2 million barrels per day of refinery capacity offline. A pipeline sanction can reroute European natural gas flows overnight.

Analyzing these markets requires integrating real-time price data, supply/demand fundamentals, infrastructure status, and geopolitical context — and producing a signal fast enough to be actionable. Most energy analytics platforms are either too slow (monthly reports) or too narrow (price charts without fundamental context).

### The Data Model

Scope.Glacier tracks six entity types:

1. **EnergyCommodity** — Reference data for tracked commodities: WTI, Brent, Henry Hub Natural Gas, RBOB Gasoline, No. 2 Heating Oil. Each has an EIA series ID for automated data pulls.

2. **PriceSeries** — Time series with computed properties: 30-day returns, annualized volatility (rolling 20-day, annualized to 252 trading days), and simple moving averages.

3. **SupplyDemandBalance** — Weekly petroleum data: production (mbd), consumption, imports, exports, inventory (million barrels), inventory change, spare capacity, utilization. Computed properties include implied balance (production minus consumption plus net imports) and inventory coverage days.

4. **Refinery** — Individual refinery tracking: capacity, utilization, throughput, crude type, status (Operating/Maintenance/Shutdown/Idle), offline capacity.

5. **Pipeline** — Pipeline infrastructure: origin, destination, capacity, current flow, utilization, status (Operational/Reduced Flow/Shutdown/Force Majeure), disruption flag.

6. **GlacierSignal** — The composite output: four sub-scores, glacier score, signal rating, AI analysis, confidence, data sources.

### The Supply/Demand Framework

The supply/demand balance model is where Scope.Glacier differentiates itself from simple price-tracking tools. The key computed metrics:

- **Implied Balance** = Production - Consumption + Imports - Exports. A positive value means oversupply (bearish), negative means deficit (bullish).

- **Inventory Coverage Days** = Current Inventory / Daily Consumption. This tells you how many days of buffer exist before a supply disruption becomes a crisis.

- **Drawdown Risk Rating**: Critical (<20 days), Low (20-30), Adequate (30-50), Comfortable (50+). Below 20 days, any supply disruption — a hurricane, a pipeline outage, a sanctions escalation — can cause an immediate price spike.

- **Balance vs 4-Week Average**: Compares current implied balance to the trailing 4-week average. A rapidly tightening balance (negative delta) signals building price pressure even if absolute inventory levels appear adequate.

### Infrastructure Monitoring

The infrastructure disruption view unifies pipelines and refineries into a single monitoring surface. A pipeline with status "Reduced Flow" at 60% utilization and a refinery under "Maintenance" with 500,000 bpd offline both appear in the same query, sorted by estimated impact. This is the kind of cross-asset visibility that institutional energy desks need but rarely have in a single tool.

### The Scoring Engine

The composite Glacier Score weights four dimensions:

```
Glacier Score =
    Supply/Demand Score (utilization, inventory)    × 0.30
  + Price Momentum Score (returns, volatility)       × 0.25
  + Geopolitical Score (risk level, OPEC compliance)  × 0.25
  + Seasonal Score (driving season, heating season)    × 0.20
```

The seasonal component is particularly important for energy. The summer driving season (Memorial Day through Labor Day) creates predictable demand for gasoline and crude. The winter heating season drives natural gas and heating oil demand. A model that doesn't account for these seasonal patterns will systematically misprice energy signals in Q2 and Q4.

### Infrastructure

- **3 Glue ETL jobs**: energy_price_etl (spot price statistics), supply_demand_etl (implied balance, inventory coverage, drawdown risk), infrastructure_etl (pipeline utilization, refinery offline capacity, aggregate disruption summary)
- **3 Athena views**: energy_price_dashboard (multi-commodity comparison, volatility, returns), supply_demand_fundamentals (implied balance, coverage days, 4W average delta), infrastructure_disruption (unified pipeline + refinery disruption monitor)
- **2 Lambda handlers**: eia_ingestion (fetches from EIA API, writes JSONL to S3), glacier_analysis (Step Functions: compute → Bedrock → write)
- **6 Iceberg tables**: price_series, supply_demand_balance, pipelines, refineries, glacier_signals, energy_commodities

---

## Shared Patterns and Lessons Learned

### The Converse API Over InvokeModel

Early prototypes used Bedrock's `InvokeModel` API, which requires constructing raw request payloads in the model-specific format (different JSON schemas for Claude vs Llama vs Titan). Switching to the Converse API provided three immediate benefits:

1. **Model portability** — Changing from Claude 3 Haiku to Claude 3.5 Sonnet requires changing one string (the model ID). The message format stays identical.

2. **System prompt handling** — System prompts are a named parameter, not a message injected into the conversation array. This is semantically correct and avoids edge cases where the model treats the system prompt as a conversational turn.

3. **Structured output** — The `inferenceConfig` parameter provides explicit control over maxTokens and temperature without guessing at model-specific parameter names.

The BedrockClient wrapper across all three platforms is 67 lines of code. It handles retry logic with exponential backoff for throttling, extracts text from the response structure, and provides both `chat()` (single-turn) and `multi_turn()` interfaces.

### Cost Without Surprises

The entire stack for each platform costs approximately $9.50/month at moderate usage:

| Service         | Usage                          | Monthly Cost |
|-----------------|--------------------------------|-------------|
| S3 Storage      | 50 GB (tiered: Standard → IA) | $1.20        |
| Athena          | ~100 queries                   | $5.00        |
| Lambda          | ~10K invocations               | $0.50        |
| Step Functions  | ~500 transitions              | $0.75        |
| Glue ETL        | 3 jobs × ~10 min               | $1.50        |
| Bedrock (Haiku) | ~100K tokens                   | $0.25        |
| EventBridge     | ~30 rules                      | $0.30        |

That's roughly $30/month for all three platforms combined — less than a single Bloomberg Terminal subscription costs per day. The cost scales with query volume and inference calls, but the base architecture is remarkably efficient because every component is serverless. There are no idle EC2 instances, no RDS instances to provision, no EMR clusters to manage.

### Testing Philosophy

All three platforms follow the same testing approach: unit tests at the model and service level, no AWS mocking in tests. Each service is designed to work with local data (lists and dictionaries) when no API keys are present. This means the test suite runs in seconds without requiring AWS credentials, which makes CI/CD trivial and local development fast.

Scope.Sentinel has 147 tests. Scope.Vantage has 59 tests. Scope.Glacier has 41 tests. Total: **247 tests, all passing**.

---

## What's Next

The immediate roadmap includes:

- **Bedrock model upgrade path** — Evaluating Claude 3.5 Sonnet for the intelligence layer. The Converse API makes this a one-line change.
- **Real-time data ingestion** — Moving from scheduled EventBridge triggers to EventBridge Pipes for near-real-time EIA and Comtrade data.
- **Cross-platform correlation** — Joining Vantage's supply chain data with Glacier's energy data. A lithium price spike driven by supply chain disruption (Vantage) should trigger a re-evaluation of energy storage demand (Glacier).
- **Iceberg table optimization** — Implementing partition evolution and snapshot expiration to manage storage costs as historical data grows.
- **Dashboard layer** — QuickSight or a custom Streamlit dashboard for interactive exploration of signals across all three platforms.

---

## The Code

All three platforms are open source under MIT license:

- **Scope.Sentinel** — [Codeberg](https://codeberg.org/cubiczan/scope-sentinel) | [GitHub](https://github.com/Cubiczan/scope-sentinel)
- **Scope.Vantage** — [Codeberg](https://codeberg.org/cubiczan/scope-vantage) | [GitHub](https://github.com/Cubiczan/scope-vantage)
- **Scope.Glacier** — [Codeberg](https://codeberg.org/cubiczan/scope-glacier) | [GitHub](https://github.com/Cubiczan/scope-glacier)

---

*If you're building analytical systems on AWS and want to discuss the architecture, the tradeoffs, or the domain modeling — reach out. The best platforms are built by people who understand both the infrastructure and the markets.*
