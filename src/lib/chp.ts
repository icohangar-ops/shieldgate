/**
 * CHP-hardened SOC action gate (Consensus Hardening Protocol, Profile A).
 *
 * A TypeScript port of the shape proven in erp-control-plane
 * (api/genbi/chp.py @ 70678cc) — the shape, not the code: shieldgate is a
 * Next.js/TypeScript service, so the gate lives in the request path instead
 * of a Python package. The canonical Python dependency
 * (`consensus-hardening-protocol==0.1.1`, see requirements.txt) documents the
 * protocol version this port follows.
 *
 * Every consequential SOC action — an AI/human SPL query against an index, an
 * incident status transition — is wrapped in four hardening stages:
 *
 * 1. **ReBAC first.** SpiceDB (or its simulation) answers *who may act*;
 *    the caller passes that ALLOW in as `rebacAllowed`. CHP answers *whether
 *    the action should happen* — it never grants authority ReBAC denied.
 * 2. **R0 gate — before the engine.** `evaluateR0Gate` with SOC-shaped
 *    criteria: the action is *solvable* (grounded in alert/incident state),
 *    *scoped* (pinned index + bounded result), *valid* (known execution
 *    surface / authorized actor), and *worth_it* (forward-moving and
 *    investigative). HALT refuses the action with nothing executed.
 * 3. **Foundation pass — after bounded execution.** The deterministic
 *    adversary scores out of 100: 40 guardrails (ReBAC allow + R0 pass +
 *    bounded execution), 30 bounded result (>=1 event within the cap, or for
 *    a transition, the incident resolved to exactly one record), 30 state
 *    corroboration (parity substitute — see below). The CHP general floor of
 *    70 applies: sub-floor outcomes are withheld pending a named human
 *    confirmer. There is NO golden source for SOC work (no pinned truth for
 *    arbitrary live Splunk queries), so per the rollout brief, state
 *    assertions serve in place of golden parity: corroboration means the
 *    request is bound to incident state (context present, or the queried
 *    index matches the incident's sourceIndex) and, for transitions, that
 *    the move is forward in the incident lifecycle. A state *contradiction*
 *    (transition of a nonexistent incident, backward transition) is fatal —
 *    no confirmer can wave it through.
 * 4. **Human lock.** A decision case opens EXPLORING, is explicitly set to
 *    PROVISIONAL_LOCK once the foundation pass completes, and only a named
 *    human confirmer (`confirmed_by`, via third-party validation) sets it
 *    LOCKED. DESTRUCTIVE/containment actions (terminal incident statuses:
 *    resolved/closed) additionally can NEVER self-certify while
 *    `SHIELDGATE_REQUIRE_HUMAN_LOCK` is on (default) — the human lock is the
 *    real control for destructive actions, regardless of foundation score.
 *    AI principals (`ai_agent`) cannot confirm: they must obtain a human's
 *    confirmation through `POST /api/chp/decisions` (the confirmer's own JWT
 *    role is authoritative there).
 * 5. **Decision record.** Each decision is serialised into a payload
 *    envelope and appended to an append-only JSONL ledger. The CHP envelope
 *    validates structure only, so the ledger adds its own SHA-256
 *    `body_sha256` over the sealed body; every read re-validates and exposes
 *    `integrity_valid`.
 */

import { createHash } from "node:crypto";
import { mkdirSync, existsSync, readFileSync, appendFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { UserRole } from "./authz-types";

// ---------------------------------------------------------------------------
// Protocol primitives (ported from chp.gates / chp.models / chp.payloads)
// ---------------------------------------------------------------------------

export type ChpVerdict = "PASS" | "HALT";
export type GateResult = "PASS" | "FATAL";

/** CHP session lifecycle, restricted to the statuses the SOC gate uses. */
export type SessionStatus =
  | "EXPLORING"
  | "PROVISIONAL_LOCK"
  | "LOCKED"
  | "REFRAME_REQUIRED"
  | "HALT";

export interface R0Evaluation {
  /** Capitalized per the CHP contract: Solvable / Scoped / Valid / Worth_it. */
  results: Record<"Solvable" | "Scoped" | "Valid" | "Worth_it", GateResult>;
  verdict: ChpVerdict;
}

/**
 * The pre-execution gate. Any FATAL criterion HALTs the whole evaluation —
 * a partial pass must never let a malformed action reach the engine.
 */
export function evaluateR0Gate(criteria: {
  solvable: boolean;
  scoped: boolean;
  valid: boolean;
  worth_it: boolean;
}): R0Evaluation {
  const results = {
    Solvable: criteria.solvable ? "PASS" : "FATAL",
    Scoped: criteria.scoped ? "PASS" : "FATAL",
    Valid: criteria.valid ? "PASS" : "FATAL",
    Worth_it: criteria.worth_it ? "PASS" : "FATAL",
  } as R0Evaluation["results"];
  const verdict: ChpVerdict = Object.values(results).every((r) => r === "PASS")
    ? "PASS"
    : "HALT";
  return { results, verdict };
}

/** Deterministic body digest. The CHP envelope is structure-only; the ledger
 * owns this digest so tampering is detectable on read. */
export function sha256Hex(body: string): string {
  return createHash("sha256").update(body, "utf8").digest("hex");
}

/** Mirrors chp.payloads.build_payload_envelope: BEGIN/END markers carrying the
 * payload id around the sealed body. */
export function buildPayloadEnvelope(
  body: string,
  route = "RX",
  payloadId?: string
): string {
  const id = payloadId ?? `payload_${sha256Hex(body).slice(0, 16)}`;
  return `BEGIN_PAYLOAD [${id}]\n${body}\nEND_PAYLOAD [${id}]`;
}

/** Structure-only envelope validation (as in CHP 0.1.1): the BEGIN/END markers
 * must wrap the payload and carry a matching id. Content integrity is the
 * ledger's own `body_sha256` concern. */
export function validatePayloadEnvelope(rendered: string): boolean {
  const lines = rendered
    .trim()
    .split("\n")
    .map((l) => l.trimEnd());
  if (lines.length < 3) return false;
  const first = lines[0];
  const last = lines[lines.length - 1];
  if (!first.startsWith("BEGIN_PAYLOAD [") || !last.startsWith("END_PAYLOAD ["))
    return false;
  return (
    first.replace("BEGIN_PAYLOAD", "").trim() ===
    last.replace("END_PAYLOAD", "").trim()
  );
}

// ---------------------------------------------------------------------------
// Deterministic SOC scoring
// ---------------------------------------------------------------------------

// Deterministic adversary scoring (out of 100). The CHP general floor is 70.
const GUARDRAIL_POINTS = 40;
const BOUNDED_RESULT_POINTS = 30;
const STATE_CORROBORATION_POINTS = 30;
export const FULL_SCORE = 100;
export const GENERAL_FLOOR = 70;

/** Upper bound on events a gated query may return. Bounded execution is part
 * of the guardrails score; the real Splunk client caps at 50 and the
 * simulator is dataset-bound. */
export const SOC_SPL_RESULT_CAP = 200;

export type ChpActionKind = "query" | "incident_transition";

export interface FoundationAssessment {
  score: number;
  findings: string[];
  fatal: string | null;
}

const INCIDENT_LIFECYCLE: Record<string, number> = {
  open: 0,
  investigating: 1,
  resolved: 2,
  closed: 3,
};

/** Terminal statuses end an incident's active response — treating them as the
 * destructive/containment class for HITL purposes. */
export function isDestructiveStatus(status: string): boolean {
  return status === "resolved" || status === "closed";
}

/** Strictly forward in the lifecycle; no-ops and reversals are not worth it. */
export function isForwardTransition(from: string, to: string): boolean {
  const a = INCIDENT_LIFECYCLE[from];
  const b = INCIDENT_LIFECYCLE[to];
  return a !== undefined && b !== undefined && b > a;
}

// Deterministic proxy for R0's worth_it on the query path: SPL that speaks the
// vocabulary of investigation (mirrors the erp _ANALYTICAL regex approach).
const INVESTIGATIVE_SPL =
  /\b(severity|action|src_ip|dest_ip|src_port|dest_port|user|sourcetype|eventcode|signature|tag|risk|count|stats|timechart|anomal|brute|exfil|exfiltrat|lateral|c2|auth|failed|sudo|privileg|payload|threat|malware|beacon|dwell)\b/i;

/** Roles that may lend a human confirmation. AI principals never confirm —
 * that is the whole point of the lock. `sre` is observability-only and
 * `contractor` is externally restricted, so neither confirms SOC decisions. */
export function eligibleConfirmerRoles(kind: ChpActionKind): UserRole[] {
  return kind === "incident_transition" ? ["soc_tier2"] : ["soc_tier1", "soc_tier2"];
}

export function isEligibleConfirmer(role: UserRole, kind: ChpActionKind): boolean {
  return eligibleConfirmerRoles(kind).includes(role);
}

/** Actor roles the incident-transition guardrail accepts (mirrors the SpiceDB
 * `resolve` intent encoded in the incidents route). */
const INCIDENT_WRITE_ROLES: UserRole[] = ["soc_tier2", "sre", "ai_agent"];

const INCIDENT_STATUS_ALLOWLIST = new Set([
  "open",
  "investigating",
  "resolved",
  "closed",
]);

// ---------------------------------------------------------------------------
// Decision ledger — append-only JSONL, re-validated on every read
// ---------------------------------------------------------------------------

export interface LedgerEntry {
  decision_id: string;
  created_at: string;
  kind: ChpActionKind;
  actor_role: UserRole;
  actor_id: string;
  session_status: SessionStatus;
  r0_verdict: ChpVerdict;
  r0_results?: R0Evaluation["results"];
  foundation_score: number | null;
  confirmed_by: string | null;
  supersedes: string | null;
  body: string;
  body_sha256: string;
  envelope: string;
}

export interface LedgerRecord extends LedgerEntry {
  envelope_valid: boolean;
  integrity_valid: boolean;
}

/** Append-only JSONL of CHP decision records; envelope structure and body
 * digest are re-checked on read so a tampered record surfaces as
 * `integrity_valid: false` rather than silently reading as truth. */
export class DecisionLedger {
  private readonly path: string;

  constructor(path: string) {
    this.path = resolve(path);
  }

  append(entry: LedgerEntry): LedgerRecord {
    mkdirSync(dirname(this.path), { recursive: true });
    appendFileSync(this.path, JSON.stringify(entry) + "\n", "utf8");
    return { ...entry, ...checkIntegrity(entry) };
  }

  list(limit = 100): LedgerRecord[] {
    const entries = this.readAll();
    return entries
      .slice(-limit)
      .reverse()
      .map((entry) => ({ ...entry, ...checkIntegrity(entry) }));
  }

  /** Latest record for a decision id, or undefined. */
  get(decisionId: string): LedgerRecord | undefined {
    for (let i = this.readAll().length - 1; i >= 0; i--) {
      const entry = this.readAll()[i];
      if (entry && entry.decision_id === decisionId) {
        return { ...entry, ...checkIntegrity(entry) };
      }
    }
    return undefined;
  }

  private readAll(): LedgerEntry[] {
    if (!existsSync(this.path)) return [];
    const lines = readFileSync(this.path, "utf8")
      .split("\n")
      .filter((line) => line.trim().length > 0);
    return lines.map((line, i) => {
      try {
        return JSON.parse(line) as LedgerEntry;
      } catch (error) {
        // A corrupt line is a ledger-integrity event, not noise — rethrow
        // with position context instead of silently skipping it.
        throw new Error(
          `Decision ledger line ${i + 1} is not valid JSON: ${String(error)}`
        );
      }
    });
  }
}

function checkIntegrity(entry: LedgerEntry): {
  envelope_valid: boolean;
  integrity_valid: boolean;
} {
  const digest = sha256Hex(entry.body);
  return {
    envelope_valid: validatePayloadEnvelope(entry.envelope),
    integrity_valid: digest === entry.body_sha256,
  };
}

// ---------------------------------------------------------------------------
// The SOC gate
// ---------------------------------------------------------------------------

export interface IncidentStateSnapshot {
  id: string;
  status: string;
  sourceIndex: string;
}

export interface QueryReviewInput {
  kind: "query";
  spl: string;
  index?: string;
  incidentId?: string;
  incidentContext?: string;
  /** Resolved incident for the investigation, when one is bound. */
  incident: IncidentStateSnapshot | null;
  /** The ReBAC decision that precedes CHP — CHP never re-grants it. */
  rebacAllowed: boolean;
  /** Whether the index resolves against the known index registry (simulation)
   * or was validated by ReBAC against the live Splunk deployment. */
  indexKnown: boolean;
  actorRole: UserRole;
  actorId: string;
  confirmedBy?: string | null;
}

export interface TransitionReviewInput {
  kind: "incident_transition";
  incidentId: string;
  targetStatus: string;
  /** Resolved incident — null when the id does not resolve (fatal). */
  incident: IncidentStateSnapshot | null;
  rebacAllowed: boolean;
  actorRole: UserRole;
  actorId: string;
  confirmedBy?: string | null;
}

export type ChpReviewInput = QueryReviewInput | TransitionReviewInput;

export type ChpReviewOutcome =
  | {
      outcome: "REFUSED";
      reason: string;
      evaluation: R0Evaluation | null;
      decision: null;
    }
  | {
      outcome: "PROCEED";
      decisionId: string;
      evaluation: R0Evaluation;
      /** True when the action may not take effect without a LOCKED record. */
      humanLockRequired: boolean;
      destructive: boolean;
    };

/**
 * Runs a SOC action through CHP: ReBAC precondition -> R0 -> (caller executes
 * within bounds) -> foundation pass -> human lock -> decision record.
 */
export class SocChpGate {
  private readonly ledger: DecisionLedger;
  private readonly requireHumanLock: boolean;

  constructor(
    ledger: DecisionLedger,
    options?: { requireHumanLock?: boolean }
  ) {
    this.ledger = ledger;
    const env = process.env.SHIELDGATE_REQUIRE_HUMAN_LOCK;
    this.requireHumanLock = options?.requireHumanLock ?? env !== "0";
  }

  // ------------------------------------------------------------------ R0 ---
  /** R0Evaluator — must run BEFORE any consequential SOC action. */
  reviewQuery(input: QueryReviewInput): ChpReviewOutcome {
    if (!input.rebacAllowed) {
      // ReBAC denied: CHP has nothing to gate. The decision is ReBAC's.
      return {
        outcome: "REFUSED",
        reason: "ReBAC denied this action; CHP never re-grants authorization",
        evaluation: null,
        decision: null,
      };
    }

    const contextGrounded =
      Boolean(input.incidentContext && input.incidentContext.trim()) ||
      Boolean(input.incidentId);
    const evaluation = evaluateR0Gate({
      solvable: input.spl.trim().length > 0 && contextGrounded,
      scoped:
        Boolean(input.index && input.index.trim()) && SOC_SPL_RESULT_CAP > 0,
      valid: Boolean(input.indexKnown),
      worth_it:
        Boolean(input.incidentContext && input.incidentContext.trim()) ||
        INVESTIGATIVE_SPL.test(input.spl),
    });

    const decisionId = queryDecisionId(input);
    if (evaluation.verdict !== "PASS") {
      const failed = Object.entries(evaluation.results)
        .filter(([, r]) => r !== "PASS")
        .map(([name]) => name)
        .sort();
      return {
        outcome: "REFUSED",
        reason: `CHP R0 gate: the query failed ${failed.join(", ")}`,
        evaluation,
        decision: null,
      };
    }
    return {
      outcome: "PROCEED",
      decisionId,
      evaluation,
      humanLockRequired: false,
      destructive: false,
    };
  }

  reviewIncidentTransition(input: TransitionReviewInput): ChpReviewOutcome {
    if (!input.rebacAllowed) {
      return {
        outcome: "REFUSED",
        reason: "ReBAC denied this action; CHP never re-grants authorization",
        evaluation: null,
        decision: null,
      };
    }

    const from = input.incident?.status ?? null;
    const evaluation = evaluateR0Gate({
      solvable: input.incident !== null,
      scoped: INCIDENT_STATUS_ALLOWLIST.has(input.targetStatus),
      valid: INCIDENT_WRITE_ROLES.includes(input.actorRole),
      worth_it:
        from !== null && isForwardTransition(from, input.targetStatus),
    });

    const decisionId = `soc-incident-${input.incidentId}-${input.targetStatus}`;
    if (evaluation.verdict !== "PASS") {
      const failed = Object.entries(evaluation.results)
        .filter(([, r]) => r !== "PASS")
        .map(([name]) => name)
        .sort();
      return {
        outcome: "REFUSED",
        reason: `CHP R0 gate: the incident transition failed ${failed.join(", ")}`,
        evaluation,
        decision: null,
      };
    }

    const destructive = isDestructiveStatus(input.targetStatus);
    return {
      outcome: "PROCEED",
      decisionId,
      evaluation,
      humanLockRequired: destructive && this.requireHumanLock,
      destructive,
    };
  }

  // ----------------------------------------------------------- foundation ---
  /** Deterministic adversary over the bounded execution result (queries). */
  assessQueryExecution(
    review: Extract<ChpReviewOutcome, { outcome: "PROCEED" }>,
    execution: { eventCount: number },
    input: QueryReviewInput
  ): FoundationAssessment {
    const findings: string[] = [];
    let score = 0;

    score += GUARDRAIL_POINTS;
    findings.push(
      `guardrails passed: ReBAC allow, R0 pass, bounded execution (cap ${SOC_SPL_RESULT_CAP})`
    );

    if (execution.eventCount >= 1 && execution.eventCount <= SOC_SPL_RESULT_CAP) {
      score += BOUNDED_RESULT_POINTS;
      findings.push(`bounded result: ${execution.eventCount} event(s)`);
    } else {
      findings.push("query returned no result evidence within the bound");
    }

    const corroborated = queryStateCorroborated(input);
    if (corroborated) {
      score += STATE_CORROBORATION_POINTS;
      findings.push(
        "state corroboration: request is bound to incident state (parity substitute — no golden SOC source exists)"
      );
    } else {
      findings.push(
        "request is not bound to incident state — corroboration evidence unavailable"
      );
    }

    return { score: Math.min(score, FULL_SCORE), findings, fatal: null };
  }

  /** Deterministic adversary over a pending incident transition. */
  assessIncidentTransition(
    input: TransitionReviewInput
  ): FoundationAssessment {
    const findings: string[] = [];
    let score = 0;
    let fatal: string | null = null;

    score += GUARDRAIL_POINTS;
    findings.push(
      `guardrails passed: role '${input.actorRole}' write-authorized, status '${input.targetStatus}' allowlisted`
    );

    if (input.incident) {
      score += BOUNDED_RESULT_POINTS;
      findings.push(`bounded scope: incident ${input.incident.id} resolves to exactly one record`);
    } else {
      findings.push("incident does not resolve — no bounded target");
    }

    const from = input.incident?.status;
    if (from && isForwardTransition(from, input.targetStatus)) {
      score += STATE_CORROBORATION_POINTS;
      findings.push(
        `state corroboration: '${from}' -> '${input.targetStatus}' is a forward lifecycle transition`
      );
    } else if (from) {
      fatal = `state contradiction: '${from}' -> '${input.targetStatus}' is not a forward transition — no confirmer can validate it`;
      findings.push(fatal);
    }

    return { score: Math.min(score, FULL_SCORE), findings, fatal };
  }

  // ------------------------------------------------------------ human lock ---
  /**
   * Human-lock decision for an assessed action: destructive actions under
   * REQUIRE_HUMAN_LOCK (default on) and sub-floor outcomes may only take
   * effect once a LOCKED record from an eligible human confirmer exists.
   */
  humanLockVerdict(
    kind: ChpActionKind,
    decisionId: string,
    assessment: FoundationAssessment,
    actorRole: UserRole,
    confirmedBy?: string | null
  ): { locked: boolean; via: "confirmed_by" | "ledger" | "self_certified" | "refused"; reason: string } {
    if (assessment.fatal) {
      return { locked: false, via: "refused", reason: assessment.fatal };
    }

    const destructive = kind === "incident_transition";
    const needsLock =
      assessment.score < GENERAL_FLOOR || (destructive && this.requireHumanLock);

    if (!needsLock) {
      return {
        locked: false,
        via: "self_certified",
        reason: "self-certified at or above the general floor",
      };
    }

    if (confirmedBy && confirmedBy.trim()) {
      if (actorRole === "ai_agent") {
        // An AI principal naming a confirmer is self-approval — refused. The
        // human must lend their own confirmation through the decisions API.
        return {
          locked: false,
          via: "refused",
          reason:
            "CHP human lock: an AI principal cannot supply confirmed_by; a human must confirm via POST /api/chp/decisions",
        };
      }
      return {
        locked: true,
        via: "confirmed_by",
        reason: `locked by ${confirmedBy.trim()}`,
      };
    }

    const record = this.ledger.get(decisionId);
    if (
      record &&
      record.integrity_valid &&
      record.envelope_valid &&
      record.session_status === "LOCKED" &&
      record.confirmed_by
    ) {
      return {
        locked: true,
        via: "ledger",
        reason: `locked by ${record.confirmed_by}`,
      };
    }

    return {
      locked: false,
      via: "refused",
      reason: destructive
        ? "CHP human lock: destructive/containment actions never auto-execute — a soc_tier2 confirmation is required"
        : `CHP human lock: foundation score ${assessment.score} below the general floor ${GENERAL_FLOOR} — a named human confirmer is required`,
    };
  }

  // -------------------------------------------------------------- record ---
  /** Seal the decision into a payload envelope and append the ledger. The
   * case is opened EXPLORING and explicitly set PROVISIONAL_LOCK here; only
   * third-party validation (a named confirmer) sets LOCKED. */
  record(input: {
    kind: ChpActionKind;
    decisionId: string;
    actorRole: UserRole;
    actorId: string;
    r0: R0Evaluation;
    assessment: FoundationAssessment;
    sessionStatus: SessionStatus;
    confirmedBy: string | null;
    spl?: string;
    index?: string;
    incidentId?: string | null;
    targetStatus?: string;
    eventCount?: number | null;
  }): LedgerRecord {
    const body = JSON.stringify(
      {
        decision_id: input.decisionId,
        kind: input.kind,
        spl: input.spl ?? null,
        index: input.index ?? null,
        incident_id: input.incidentId ?? null,
        target_status: input.targetStatus ?? null,
        event_count: input.eventCount ?? null,
        r0_verdict: input.r0.verdict,
        r0_results: input.r0.results,
        foundation_score: input.assessment.score,
        adversary_findings: input.assessment.findings,
        core_problem:
          input.kind === "query"
            ? "Decide whether this Splunk query should run against the pinned index for this investigation"
            : "Decide whether this incident status transition should be applied",
        artifacts: { findings: input.assessment.findings },
      },
      null,
      0
    );

    const entry: LedgerEntry = {
      decision_id: input.decisionId,
      created_at: new Date().toISOString(),
      kind: input.kind,
      actor_role: input.actorRole,
      actor_id: input.actorId,
      session_status: input.sessionStatus,
      r0_verdict: input.r0.verdict,
      r0_results: input.r0.results,
      foundation_score: input.assessment.score,
      confirmed_by: input.confirmedBy,
      supersedes: null,
      body,
      body_sha256: sha256Hex(body),
      envelope: buildPayloadEnvelope(body, "SOC_ACTION"),
    };
    return this.ledger.append(entry);
  }

  /** Third-party validation: PROVISIONAL_LOCK -> LOCKED. The confirmer's role
   * comes from their own authenticated session (the decisions API), never
   * from the action request. */
  confirm(decisionId: string, confirmedBy: string, confirmerRole: UserRole): LedgerRecord {
    const prior = this.ledger.get(decisionId);
    if (!prior) {
      throw new ChpRejection(`Unknown decision '${decisionId}'`);
    }
    if (!prior.integrity_valid || !prior.envelope_valid) {
      throw new ChpRejection(
        `Decision '${decisionId}' failed ledger integrity validation`
      );
    }
    if (prior.session_status === "LOCKED") {
      return prior; // idempotent
    }
    if (!isEligibleConfirmer(confirmerRole, prior.kind)) {
      throw new ChpRejection(
        `Role '${confirmerRole}' may not confirm a ${prior.kind} decision`
      );
    }
    if (!confirmedBy.trim()) {
      throw new ChpRejection("confirmed_by is required to lock a decision");
    }

    const entry: LedgerEntry = {
      decision_id: prior.decision_id,
      created_at: new Date().toISOString(),
      kind: prior.kind,
      actor_role: prior.actor_role,
      actor_id: prior.actor_id,
      session_status: "LOCKED",
      r0_verdict: prior.r0_verdict,
      r0_results: prior.r0_results,
      foundation_score: prior.foundation_score,
      confirmed_by: confirmedBy.trim(),
      supersedes: prior.body_sha256,
      body: prior.body,
      body_sha256: prior.body_sha256,
      envelope: buildPayloadEnvelope(prior.body, "SOC_CONFIRM"),
    };
    return this.ledger.append(entry);
  }

  /** Latest record for a decision, integrity-checked. */
  decision(decisionId: string): LedgerRecord | undefined {
    return this.ledger.get(decisionId);
  }

  list(limit?: number): LedgerRecord[] {
    return this.ledger.list(limit);
  }
}

export class ChpRejection extends Error {
  constructor(
    message: string,
    readonly evaluation?: R0Evaluation | null
  ) {
    super(message);
    this.name = "ChpRejection";
  }
}

/** Deterministic decision id for a query review (mirrors the erp
 * question_hash pattern: same request, same decision). */
function queryDecisionId(input: QueryReviewInput): string {
  const identity = JSON.stringify([
    input.spl,
    input.index ?? null,
    input.incidentId ?? null,
    input.incidentContext ?? null,
  ]);
  return `soc-query-${sha256Hex(identity).slice(0, 16)}`;
}

function queryStateCorroborated(input: QueryReviewInput): boolean {
  if (input.incidentContext && input.incidentContext.trim()) return true;
  if (input.incident && input.index && input.incident.sourceIndex === input.index) {
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Process-wide singleton (ledger path via SHIELDGATE_CHP_DECISIONS_PATH)
// ---------------------------------------------------------------------------

let singleton: SocChpGate | null = null;

export function getSocChpGate(): SocChpGate {
  if (!singleton) {
    const ledgerPath =
      process.env.SHIELDGATE_CHP_DECISIONS_PATH ??
      resolve(process.cwd(), ".chp", "decisions.jsonl");
    singleton = new SocChpGate(new DecisionLedger(ledgerPath));
  }
  return singleton;
}
