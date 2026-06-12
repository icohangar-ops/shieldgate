# 🛡️ ShieldGate — Least-Privilege Agentic SOC

**AuthZed × Splunk: Zero-Trust Authorization for AI-Powered Security Operations**

[![AuthZed](https://img.shields.io/badge/AuthZed-SpiceDB-7B4FD6?logo=authzed)](https://authzed.com)
[![Splunk](https://img.shields.io/badge/Splunk-Agentic%20Ops-65A637?logo=splunk)](https://splunk.devpost.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)

---

## The Problem

AI agents investigating security incidents need access to Splunk — but **unrestricted access is a security nightmare**. An AI agent that can query any index, read any log, or run any SPL query without permission controls is a privilege escalation risk waiting to happen.

Today's SOC teams face an impossible choice:
- **Lock down AI agents** → They can't investigate effectively
- **Give AI agents full access** → Blast radius is unbounded

There is no authorization layer that enforces **least-privilege** for AI agents interacting with Splunk tools.

## The Solution

**ShieldGate** puts [AuthZed](https://authzed.com) (Google Zanzibar-inspired permission system) between every AI agent/human and Splunk. Every tool call — `splunk_run_query`, `splunk_get_indexes`, `splunk_get_alerts` — passes through a SpiceDB permission check before reaching Splunk.

```
┌──────────────┐      ┌─────────────────────┐      ┌──────────────┐
│  AI Agent    │─────▶│  ShieldGate         │─────▶│  Splunk      │
│  (Claude,    │ MCP  │  AuthZed Gateway    │ MCP  │  MCP Server  │
│  Custom)     │◀─────│                     │◀─────│              │
└──────────────┘      └──────────┬──────────┘      └──────────────┘
                                 │
                          ┌──────▼──────┐
                          │  AuthZed    │
                          │  SpiceDB    │
                          │  Cloud      │
                          └─────────────┘
```

### What Makes This Different

| Existing Approach | ShieldGate |
|---|---|
| Role-based access at login | **Per-tool, per-index, per-query** permission checks |
| Static IAM policies | **Zanzibar-style ReBAC** with relationship inheritance |
| Audit logs after the fact | **Pre-execution authorization** — denied queries never hit Splunk |
| All-or-nothing AI access | **Conditional permissions** with constraints (limited SPL, redacted results) |
| Manual compliance reviews | **Real-time auth timeline** — every ALLOW/DENY visible |

## Architecture

### AuthZed SpiceDB Schema

```zed
definition user {}

definition team {
  relation member: user
  permission member_access = member
}

definition splunk_index {
  relation viewer:   team | user
  relation querier:  team | user
  relation admin:    user
  permission read    = viewer + querier + admin
  permission query   = querier + admin
  permission manage  = admin
}

definition splunk_tool {
  relation allowed_role: team
  relation allowed_user: user
  permission execute = allowed_user + allowed_role->member_access - restricted
}

definition incident {
  relation index:          splunk_index
  relation assigned_team:  team
  relation viewer:         user
  permission view = viewer + assigned_team->member_access + index->read
}
```

### Permission Matrix

| Capability | SOC Tier 1 | SOC Tier 2 | SRE | Contractor | AI Agent |
|---|:-:|:-:|:-:|:-:|:-:|
| Read security index | ✅ | ✅ | ❌ | ✅ redacted | ✅ |
| Query security index | ❌ | ✅ | ❌ | ❌ | ✅ w/ approval |
| Read observability | ❌ | ✅ | ✅ | ❌ | ✅ |
| Run SPL queries | ✅ limited | ✅ | ✅ limited | ❌ | ✅ w/ approval |
| Get alerts | ✅ | ✅ | ✅ obs only | ❌ | ✅ |
| AI assistant | ✅ | ✅ | ✅ | ✅ redacted | ✅ |
| Remediation | ❌ | ✅ | ❌ | ❌ | ❌ human gate |

## Features

### 🔐 AuthZed Permission Engine
- **Fine-grained ReBAC** — Permissions derived from relationships, not static roles
- **Per-tool authorization** — 11 Splunk MCP tools each with independent permission rules
- **Per-index isolation** — SRE can't query security, contractors can't query anything
- **Conditional policies** — Limited SPL for Tier 1, redacted results for contractors

### 🤖 AI Incident Investigator
- Chat-based interface that investigates security incidents
- Suggests SPL queries based on incident context
- Correlates events across indexes
- Recommends containment and remediation steps

### 📊 Real-Time Authorization Audit
- Every ALLOW/DENY decision logged with timestamp
- Visual timeline of permission checks
- Drill down into reason for each decision
- Perfect for compliance reporting

### 🔴 Contractor Data Redaction
- Sensitive fields automatically redacted (IPs, usernames, hashes)
- Raw log lines sanitized before display
- No code changes needed — handled at the authorization layer

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| **UI Components** | shadcn/ui (New York style), Lucide Icons |
| **Authorization** | AuthZed SpiceDB (Zanzibar ReBAC) |
| **Data Platform** | Splunk Enterprise / Cloud via MCP Server |
| **AI Chat** | LLM-powered incident investigation |
| **State** | Zustand (client), TanStack Query (server) |
| **Database** | Prisma ORM (SQLite) |
| **Styling** | Tailwind CSS 4, Framer Motion |

## Getting Started

### Prerequisites
- Node.js 18+ / Bun
- AuthZed Cloud account or self-hosted SpiceDB
- Splunk Enterprise/Cloud with MCP Server installed

### Installation

```bash
# Clone the repo
git clone https://github.com/icohangar-ops/shieldgate.git
cd shieldgate

# Install dependencies
bun install

# Set up database
bun run db:push

# Seed demo data (12 incidents + audit logs)
bun run seed.ts

# Start development server
bun run dev
```

### Configuration

Create a `.env.local` file:

```env
# AuthZed
AUTHZED_API_KEY=your_authzed_api_key
AUTHZED_ENDPOINT=grpc.authzed.com:443

# Splunk
SPLUNK_HOST=localhost
SPLUNK_PORT=8089
SPLUNK_TOKEN=your_splunk_token

# Database
DATABASE_URL=file:./db/custom.db
```

## Demo Walkthrough

### Scenario 1: SOC Analyst Investigation
1. Select **SOC Tier 1** role
2. Click the **"Active Data Exfiltration"** critical incident
3. Ask the AI: *"Investigate the data exfiltration — what SPL queries should I run?"*
4. Click a suggested query → AuthZed shows **ALLOW** with constraint (Limited SPL)
5. Switch to **Evidence** tab to review raw events

### Scenario 2: Contractor Isolation
1. Switch to **Contractor** role
2. Notice security incidents show a 🔒 lock icon
3. Try running a SPL query → AuthZed shows **DENY** ("Contractors cannot execute ad-hoc queries")
4. Click an incident → Sensitive fields show **[REDACTED]**

### Scenario 3: SRE Index Isolation
1. Switch to **SRE** role
2. Click a security incident → **ALLOW** (evidence visible)
3. Try `index=security | stats count by action` → **DENY** ("SRE does not have access to index 'security'")
4. Switch to observability query → **ALLOW** with constraint

### Scenario 4: AI Agent Human-in-the-Loop
1. Switch to **AI Agent** role
2. Run a query → **ALLOW** with constraint ("Requires human approval for remediation queries")
3. Check the **AuthZ Log** tab → See all permission decisions in real-time

## Hackathon Alignment

### Splunk Agentic Ops Hackathon Tracks

| Track | How ShieldGate Competes |
|---|---|
| **Best of Security** | Core value proposition — least-privilege for AI agents is THE security story |
| **Best Use of Splunk MCP Server** | Wraps every Splunk MCP tool with AuthZed permission checks |
| **Best AI App** | AI investigator with constrained, auditable, human-in-the-loop actions |
| **Best Developer Experience** | Real-time permission timeline, role switching, instant feedback |

### Why This Wins
- **Real enterprise pain** — Every SOC team struggles with AI agent access control
- **Novel integration** — Nobody has combined Zanzibar ReBAC with Splunk MCP before
- **Production architecture** — Same pattern used by Google (Zanzibar), Airbnb, Carta
- **Complete demo** — 5 role profiles, 12 incidents, working AI chat, real permission engine

## Project Structure

```
src/
├── app/
│   ├── api/
│   │   ├── authz/check/     # AuthZed permission check endpoint
│   │   ├── splunk/query/    # SPL query with AuthZed gate
│   │   ├── splunk/indexes/  # Index listing with permissions
│   │   ├── splunk/alerts/   # Alert filtering by role
│   │   ├── incidents/       # Incident CRUD
│   │   ├── audit/           # Authorization audit trail
│   │   └── chat/            # AI incident investigation
│   ├── layout.tsx
│   ├── page.tsx             # Main dashboard
│   └── globals.css
├── lib/
│   ├── authz.ts             # AuthZed permission engine
│   ├── splunk-sim.ts        # Splunk simulation layer
│   ├── store.ts             # Zustand state management
│   └── db.ts                # Prisma client
└── components/ui/           # shadcn/ui components

prisma/schema.prisma         # Incident, AuditLog, ChatMessage
seed.ts                      # Demo data seeding
```

## Built With

- [AuthZed](https://authzed.com) — Google Zanzibar-inspired authorization
- [Splunk](https://splunk.devpost.com) — AI-powered security operations
- [Next.js](https://nextjs.org) — React framework
- [shadcn/ui](https://ui.shadcn.com) — UI component library
- [Prisma](https://prisma.io) — Database ORM
- [Tailwind CSS](https://tailwindcss.com) — Utility-first CSS

## License

MIT

---

**Built for the [Splunk Agentic Ops Hackathon](https://splunk.devpost.com) by [Cubiczan Technologies](https://www.cubiczan.com)**
