/// <reference types="bun-types" />
/**
 * CHP-hardened SOC gate suite — mirrors the erp-control-plane CHP test shape
 * (tests/test_genbi_chp.py): R0 refusal, deterministic foundation scoring,
 * human lock flow, and the append-only decision ledger with tamper detection.
 * SOC specifics under test on top: destructive-action HITL enforcement and
 * the ReBAC-before-CHP interplay (ReBAC authorizes who, CHP gates whether).
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  evaluateR0Gate,
  validatePayloadEnvelope,
  buildPayloadEnvelope,
  sha256Hex,
  DecisionLedger,
  SocChpGate,
  ChpRejection,
  GENERAL_FLOOR,
  type IncidentStateSnapshot,
} from '../src/lib/chp';

const INVESTIGATIVE_SPL =
  'index=security sourcetype=auth action=login_failure | stats count by src_ip';
const BARE_SPL = 'hello world';
const CONFIRMER = 'sam@cubiczan.com';

let ledgerDir: string;
let ledgerPath: string;

function makeLedger(): DecisionLedger {
  return new DecisionLedger(ledgerPath);
}

function makeGate(opts?: { requireHumanLock?: boolean }): SocChpGate {
  return new SocChpGate(makeLedger(), opts);
}

const INCIDENT: IncidentStateSnapshot = {
  id: 'inc_1',
  status: 'investigating',
  sourceIndex: 'security',
};

function queryInput(overrides: Record<string, unknown> = {}) {
  return {
    kind: 'query' as const,
    spl: INVESTIGATIVE_SPL,
    index: 'security',
    incidentId: 'inc_1',
    incidentContext: 'Impossible travel alert on exec laptop',
    incident: INCIDENT,
    rebacAllowed: true,
    indexKnown: true,
    actorRole: 'ai_agent' as const,
    actorId: 'agent_1',
    ...overrides,
  };
}

function transitionInput(overrides: Record<string, unknown> = {}) {
  return {
    kind: 'incident_transition' as const,
    incidentId: 'inc_1',
    targetStatus: 'resolved',
    incident: INCIDENT,
    rebacAllowed: true,
    actorRole: 'soc_tier2' as const,
    actorId: 'analyst_1',
    ...overrides,
  };
}

beforeEach(() => {
  ledgerDir = mkdtempSync(join(tmpdir(), 'chp-ledger-'));
  ledgerPath = join(ledgerDir, 'decisions.jsonl');
});

afterEach(() => {
  rmSync(ledgerDir, { recursive: true, force: true });
});

// ------------------------------------------------------------------- R0 ---

describe('CHP R0 gate (queries)', () => {
  test('refuses a non-investigative query before execution (Worth_it FATAL)', () => {
    const review = makeGate().reviewQuery(queryInput({ spl: BARE_SPL, incidentContext: undefined }));
    expect(review.outcome).toBe('REFUSED');
    expect(review.evaluation?.results['Worth_it']).toBe('FATAL');
  });

  test('refuses an empty query (Solvable FATAL)', () => {
    const review = makeGate().reviewQuery(queryInput({ spl: '   ' }));
    expect(review.outcome).toBe('REFUSED');
    expect(review.evaluation?.results['Solvable']).toBe('FATAL');
  });

  test('refuses a query with no grounding in alert/incident state (Solvable FATAL)', () => {
    const review = makeGate().reviewQuery(
      queryInput({ incidentId: undefined, incidentContext: undefined, incident: null })
    );
    expect(review.outcome).toBe('REFUSED');
    expect(review.evaluation?.results['Solvable']).toBe('FATAL');
  });

  test('refuses an unpinned index (Scoped FATAL) and an unknown index (Valid FATAL)', () => {
    const unpinned = makeGate().reviewQuery(queryInput({ index: undefined }));
    expect(unpinned.outcome).toBe('REFUSED');
    expect(unpinned.evaluation?.results['Scoped']).toBe('FATAL');

    const unknown = makeGate().reviewQuery(queryInput({ indexKnown: false }));
    expect(unknown.outcome).toBe('REFUSED');
    expect(unknown.evaluation?.results['Valid']).toBe('FATAL');
  });

  test('accepts an investigative query grounded in incident state', () => {
    const review = makeGate().reviewQuery(queryInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    expect(Object.values(review.evaluation.results).every(r => r === 'PASS')).toBe(true);
    expect(review.evaluation.verdict).toBe('PASS');
  });

  test('accepts a context-bound query even with plain SPL (analyst is investigating)', () => {
    const review = makeGate().reviewQuery(queryInput({ spl: BARE_SPL }));
    expect(review.outcome).toBe('PROCEED');
  });

  test('produces a deterministic decision id for identical requests', () => {
    const a = makeGate().reviewQuery(queryInput());
    const b = makeGate().reviewQuery(queryInput());
    expect(a.outcome).toBe('PROCEED');
    expect(b.outcome).toBe('PROCEED');
    if (a.outcome === 'PROCEED' && b.outcome === 'PROCEED') {
      expect(a.decisionId).toBe(b.decisionId);
    }
  });
});

describe('CHP R0 gate (incident transitions)', () => {
  test('refuses a transition of a nonexistent incident (Solvable FATAL)', () => {
    const review = makeGate().reviewIncidentTransition(
      transitionInput({ incident: null })
    );
    expect(review.outcome).toBe('REFUSED');
    expect(review.evaluation?.results['Solvable']).toBe('FATAL');
  });

  test('refuses backward and no-op transitions (Worth_it FATAL)', () => {
    for (const target of ['open', 'investigating']) {
      const review = makeGate().reviewIncidentTransition(
        transitionInput({ targetStatus: target })
      );
      expect(review.outcome).toBe('REFUSED');
      expect(review.evaluation?.results['Worth_it']).toBe('FATAL');
    }
  });

  test('refuses a non-allowlisted status (Scoped FATAL)', () => {
    const review = makeGate().reviewIncidentTransition(
      transitionInput({ targetStatus: 'deleted' })
    );
    expect(review.outcome).toBe('REFUSED');
    expect(review.evaluation?.results['Scoped']).toBe('FATAL');
  });

  test('refuses a role outside the write set (Valid FATAL)', () => {
    const review = makeGate().reviewIncidentTransition(
      transitionInput({ actorRole: 'soc_tier1' })
    );
    expect(review.outcome).toBe('REFUSED');
    expect(review.evaluation?.results['Valid']).toBe('FATAL');
  });

  test('accepts a forward transition and flags destructive terminal statuses', () => {
    const gate = makeGate();
    const resolve = gate.reviewIncidentTransition(transitionInput({ targetStatus: 'resolved' }));
    if (resolve.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    expect(resolve.humanLockRequired).toBe(true); // REQUIRE_HUMAN_LOCK default ON

    const investigate = gate.reviewIncidentTransition(
      transitionInput({ targetStatus: 'investigating', incident: { ...INCIDENT, status: 'open' } })
    );
    if (investigate.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    expect(investigate.humanLockRequired).toBe(false); // non-destructive
  });
});

// ------------------------------------------------------------ foundation ---

describe('deterministic foundation scoring', () => {
  test('guardrails 40 + bounded result 30 + state corroboration 30 = 100', () => {
    const gate = makeGate();
    const review = gate.reviewQuery(queryInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessQueryExecution(
      review,
      { eventCount: 5 },
      queryInput()
    );
    expect(assessment.score).toBe(100);
    expect(assessment.fatal).toBeNull();
  });

  test('zero-row queries cannot earn the bounded-result component', () => {
    const gate = makeGate();
    const review = gate.reviewQuery(queryInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessQueryExecution(
      review,
      { eventCount: 0 },
      queryInput()
    );
    expect(assessment.score).toBe(70); // exactly at the general floor
  });

  test('a query unbound from incident state loses the corroboration component', () => {
    const gate = makeGate();
    const unbound = queryInput({
      incidentContext: undefined,
      incident: { ...INCIDENT, sourceIndex: 'compliance' }, // queried index differs
    });
    const review = gate.reviewQuery(unbound);
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessQueryExecution(review, { eventCount: 0 }, unbound);
    expect(assessment.score).toBe(40);
    expect(assessment.score).toBeLessThan(GENERAL_FLOOR);
  });

  test('transition scoring: full for a forward transition on an existing incident', () => {
    const gate = makeGate();
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(transitionInput());
    expect(assessment.score).toBe(100);
    expect(assessment.fatal).toBeNull();
  });

  test('transition scoring marks state contradictions fatal', () => {
    const gate = makeGate();
    // Reach the foundation layer with a contradictory state: R0 is checked
    // first in the route, but the adversary independently refuses.
    const assessment = gate.assessIncidentTransition(
      transitionInput({ incident: { ...INCIDENT, status: 'closed' } })
    );
    expect(assessment.fatal).toContain('not a forward transition');
    expect(assessment.score).toBeLessThan(100);
  });
});

// ------------------------------------------------------------ human lock ---

describe('human lock flow', () => {
  test('destructive transitions never auto-execute while REQUIRE_HUMAN_LOCK is on', () => {
    const gate = makeGate({ requireHumanLock: true }); // default
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(transitionInput());
    const verdict = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      'soc_tier2'
    );
    expect(verdict.locked).toBe(false);
    expect(verdict.reason).toContain('never auto-execute');
  });

  test('a named human confirmer locks a destructive transition', () => {
    const gate = makeGate({ requireHumanLock: true });
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(transitionInput());
    const verdict = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      'soc_tier2',
      CONFIRMER
    );
    expect(verdict.locked).toBe(true);
    expect(verdict.via).toBe('confirmed_by');
  });

  test('an AI principal cannot supply confirmed_by (self-approval refused)', () => {
    const gate = makeGate({ requireHumanLock: true });
    const review = gate.reviewIncidentTransition(
      transitionInput({ actorRole: 'ai_agent' })
    );
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(
      transitionInput({ actorRole: 'ai_agent' })
    );
    const verdict = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      'ai_agent',
      'someone'
    );
    expect(verdict.locked).toBe(false);
    expect(verdict.via).toBe('refused');
    expect(verdict.reason).toContain('AI principal');
  });

  test('with REQUIRE_HUMAN_LOCK off, a full-score destructive transition self-certifies', () => {
    const gate = makeGate({ requireHumanLock: false });
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(transitionInput());
    const verdict = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      'soc_tier2'
    );
    expect(verdict.locked).toBe(false);
    expect(verdict.via).toBe('self_certified');
  });

  test('third-party validation locks a provisional decision and is idempotent', () => {
    const gate = makeGate({ requireHumanLock: true });
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(transitionInput());
    gate.record({
      kind: 'incident_transition',
      decisionId: review.decisionId,
      actorRole: 'soc_tier2',
      actorId: 'analyst_1',
      r0: review.evaluation,
      assessment,
      sessionStatus: 'PROVISIONAL_LOCK',
      confirmedBy: null,
      incidentId: 'inc_1',
      targetStatus: 'resolved',
    });

    const locked = gate.confirm(review.decisionId, CONFIRMER, 'soc_tier2');
    expect(locked.session_status).toBe('LOCKED');
    expect(locked.confirmed_by).toBe(CONFIRMER);
    expect(locked.supersedes).not.toBeNull();

    const again = gate.confirm(review.decisionId, CONFIRMER, 'soc_tier2');
    expect(again.created_at).toBe(locked.created_at); // idempotent, no new entry

    // The re-issued action now passes the human lock through the ledger.
    const verdict = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      'soc_tier2'
    );
    expect(verdict.locked).toBe(true);
    expect(verdict.via).toBe('ledger');
  });

  test('confirmation authority: soc_tier2 for destructive, soc_tier1 only for queries', () => {
    const gate = makeGate();
    // Destructive decision recorded...
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    gate.record({
      kind: 'incident_transition',
      decisionId: review.decisionId,
      actorRole: 'soc_tier2',
      actorId: 'analyst_1',
      r0: review.evaluation,
      assessment: gate.assessIncidentTransition(transitionInput()),
      sessionStatus: 'PROVISIONAL_LOCK',
      confirmedBy: null,
      incidentId: 'inc_1',
      targetStatus: 'resolved',
    });
    expect(() => gate.confirm(review.decisionId, CONFIRMER, 'soc_tier1')).toThrow(ChpRejection);
    expect(() => gate.confirm(review.decisionId, CONFIRMER, 'ai_agent')).toThrow(ChpRejection);
    expect(() => gate.confirm(review.decisionId, CONFIRMER, 'sre')).toThrow(ChpRejection);
    expect(() => gate.confirm(review.decisionId, '   ', 'soc_tier2')).toThrow(ChpRejection);
  });

  test('confirming an unknown decision is rejected', () => {
    expect(() => makeGate().confirm('soc-query-does-not-exist', CONFIRMER, 'soc_tier2')).toThrow(
      ChpRejection
    );
  });
});

// ---------------------------------------------------------------- ledger ---

describe('decision ledger', () => {
  function recordQuery(gate: SocChpGate, decisionId: string) {
    const review = gate.reviewQuery(queryInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessQueryExecution(review, { eventCount: 5 }, queryInput());
    return gate.record({
      kind: 'query',
      decisionId,
      actorRole: 'ai_agent',
      actorId: 'agent_1',
      r0: review.evaluation,
      assessment,
      sessionStatus: 'PROVISIONAL_LOCK',
      confirmedBy: null,
      spl: INVESTIGATIVE_SPL,
      index: 'security',
      incidentId: 'inc_1',
      eventCount: 5,
    });
  }

  test('appends records and reads them back newest-first, integrity-checked', () => {
    const gate = makeGate();
    const first = recordQuery(gate, 'soc-query-aaa');
    const second = recordQuery(gate, 'soc-query-bbb');

    const records = gate.list();
    expect(records.length).toBe(2);
    expect(records[0].decision_id).toBe('soc-query-bbb');
    expect(records[1].decision_id).toBe('soc-query-aaa');
    expect(records[0].envelope_valid).toBe(true);
    expect(records[0].integrity_valid).toBe(true);
    expect(first.body_sha256).toBe(sha256Hex(first.body));
    expect(second.session_status).toBe('PROVISIONAL_LOCK');
  });

  test('get returns the latest record for a decision id', () => {
    const gate = makeGate();
    recordQuery(gate, 'soc-query-aaa');
    gate.confirm('soc-query-aaa', CONFIRMER, 'soc_tier1');
    const record = gate.decision('soc-query-aaa');
    expect(record?.session_status).toBe('LOCKED');
    expect(record?.confirmed_by).toBe(CONFIRMER);
  });

  test('a tampered body reads as integrity_invalid', () => {
    const gate = makeGate();
    recordQuery(gate, 'soc-query-tamper');
    gate.confirm('soc-query-tamper', CONFIRMER, 'soc_tier1');

    // Tamper with the second line's body in place.
    const lines = readFileSync(ledgerPath, 'utf8')
      .split('\n')
      .filter(l => l.trim());
    const entry = JSON.parse(lines[1]);
    entry.body = entry.body.replace('"foundation_score":100', '"foundation_score":40');
    lines[1] = JSON.stringify(entry);
    writeFileSync(ledgerPath, lines.join('\n') + '\n');

    const tampered = new DecisionLedger(ledgerPath).get('soc-query-tamper');
    expect(tampered?.integrity_valid).toBe(false);
    expect(tampered?.session_status).toBe('LOCKED'); // visible but unverifiable
  });

  test('a broken envelope reads as envelope_invalid', () => {
    const gate = makeGate();
    recordQuery(gate, 'soc-query-envelope');

    const lines = readFileSync(ledgerPath, 'utf8')
      .split('\n')
      .filter(l => l.trim());
    const entry = JSON.parse(lines[0]);
    entry.envelope = entry.envelope.replace('END_PAYLOAD', 'END_PAYLOAD_X');
    lines[0] = JSON.stringify(entry);
    writeFileSync(ledgerPath, lines.join('\n') + '\n');

    const broken = new DecisionLedger(ledgerPath).get('soc-query-envelope');
    expect(broken?.envelope_valid).toBe(false);
    expect(broken?.integrity_valid).toBe(true); // body digest still intact
  });

  test('a corrupt JSON line is surfaced, never silently skipped', () => {
    const gate = makeGate();
    recordQuery(gate, 'soc-query-ok');
    const original = readFileSync(ledgerPath, 'utf8');
    writeFileSync(ledgerPath, original + 'this is not json\n');

    expect(() => new DecisionLedger(ledgerPath).list()).toThrow(/not valid JSON/);
  });

  test('the human lock refuses a decision whose LOCKED record is tampered', () => {
    const gate = makeGate({ requireHumanLock: true });
    const review = gate.reviewIncidentTransition(transitionInput());
    if (review.outcome !== 'PROCEED') throw new Error('expected PROCEED');
    const assessment = gate.assessIncidentTransition(transitionInput());
    gate.record({
      kind: 'incident_transition',
      decisionId: review.decisionId,
      actorRole: 'soc_tier2',
      actorId: 'analyst_1',
      r0: review.evaluation,
      assessment,
      sessionStatus: 'PROVISIONAL_LOCK',
      confirmedBy: null,
      incidentId: 'inc_1',
      targetStatus: 'resolved',
    });
    gate.confirm(review.decisionId, CONFIRMER, 'soc_tier2');

    // Tamper the LOCKED record's body digest.
    const lines = readFileSync(ledgerPath, 'utf8')
      .split('\n')
      .filter(l => l.trim());
    const entry = JSON.parse(lines[1]);
    entry.body_sha256 = '0'.repeat(64);
    lines[1] = JSON.stringify(entry);
    writeFileSync(ledgerPath, lines.join('\n') + '\n');

    const verdict = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      'soc_tier2'
    );
    expect(verdict.locked).toBe(false); // tamper evidence wins over the lock
  });
});

// ----------------------------------------------------------- primitives ---

describe('CHP protocol primitives', () => {
  test('evaluateR0Gate emits capitalized keys with FATAL values and HALT verdicts', () => {
    const evaluation = evaluateR0Gate({
      solvable: true,
      scoped: false,
      valid: true,
      worth_it: true,
    });
    expect(Object.keys(evaluation.results)).toEqual([
      'Solvable',
      'Scoped',
      'Valid',
      'Worth_it',
    ]);
    expect(evaluation.results['Scoped']).toBe('FATAL');
    expect(evaluation.verdict).toBe('HALT');

    const passing = evaluateR0Gate({
      solvable: true,
      scoped: true,
      valid: true,
      worth_it: true,
    });
    expect(passing.verdict).toBe('PASS');
  });

  test('payload envelope validates structure only (reference-verified tolerance)', () => {
    const body = '{"k":"v"}';
    const envelope = buildPayloadEnvelope(body, 'SOC_ACTION');
    expect(validatePayloadEnvelope(envelope)).toBe(true);
    expect(validatePayloadEnvelope(envelope + '\n')).toBe(true); // trailing newline tolerated
    expect(validatePayloadEnvelope('  ' + envelope + '  ')).toBe(true); // outer whitespace tolerated
    expect(validatePayloadEnvelope(envelope.replace('[', '[').replace('END_PAYLOAD ', 'END_PAYLOAD  '))).toBe(
      false // double space breaks the END_PAYLOAD [ marker
    );
    expect(validatePayloadEnvelope('BEGIN_PAYLOAD [x]\nbody')).toBe(false);
    const mismatched = `BEGIN_PAYLOAD [id1]\n${body}\nEND_PAYLOAD [id2]`;
    expect(validatePayloadEnvelope(mismatched)).toBe(false);
  });
});

// ---------------------------------------------------- ReBAC interplay ---

describe('ReBAC + CHP interplay', () => {
  test('a ReBAC deny short-circuits: CHP is not consulted and cannot re-grant', () => {
    const gate = makeGate();
    const review = gate.reviewQuery(queryInput({ rebacAllowed: false }));
    expect(review.outcome).toBe('REFUSED');
    if (review.outcome !== 'REFUSED') throw new Error('expected REFUSED');
    expect(review.evaluation).toBeNull(); // no R0 evaluation — ReBAC decided
    expect(review.reason).toContain('ReBAC');

    const transition = gate.reviewIncidentTransition(
      transitionInput({ rebacAllowed: false })
    );
    expect(transition.outcome).toBe('REFUSED');
    if (transition.outcome !== 'REFUSED') throw new Error('expected REFUSED');
    expect(transition.evaluation).toBeNull();
  });

  test('ReBAC allow + CHP HALT = refused with a mechanical reason', () => {
    const gate = makeGate();
    const review = gate.reviewQuery(
      queryInput({ spl: BARE_SPL, incidentContext: undefined, incident: null, incidentId: undefined })
    );
    expect(review.outcome).toBe('REFUSED');
    if (review.outcome !== 'REFUSED') throw new Error('expected REFUSED');
    expect(review.reason).toContain('Worth_it');
  });
});
