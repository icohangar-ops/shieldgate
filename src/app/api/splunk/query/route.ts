import { NextResponse } from 'next/server';
import { simulateSplunkQuery, redactEvents, SPLUNK_INDEXES } from '@/lib/splunk-sim';
import { isSplunkConfigured, runSplunkQuery } from '@/lib/splunk-client';
import { checkToolPermission, AuthZUnavailableError } from '@/lib/authz';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';
import { getSocChpGate, type IncidentStateSnapshot } from '@/lib/chp';
import { db } from '@/lib/db';

export const POST = withAuth(async (request: AuthenticatedRequest) => {
  try {
    const body = await request.json();
    const { spl, index, incidentId, incidentContext, confirmed_by } = body as {
      spl: string;
      index?: string;
      incidentId?: string;
      incidentContext?: string;
      confirmed_by?: string;
    };
    const role = request.authRole;

    if (!spl) {
      return NextResponse.json({ error: 'spl is required' }, { status: 400 });
    }

    // Step 1: AuthZed permission check — ReBAC answers WHO may act.
    const permission = await checkToolPermission(role, 'splunk_run_query', index);
    if (!permission.allowed) {
      return NextResponse.json({
        authorized: false,
        error: 'PERMISSION_DENIED',
        reason: permission.reason,
        policy: permission.policy,
      }, { status: 403 });
    }

    // Step 2: CHP R0 gate — the protocol answers WHETHER the action should
    // happen, before the engine sees the query. ReBAC never re-grants here:
    // a ReBAC deny short-circuits above and CHP is not consulted.
    const gate = getSocChpGate();
    const incident: IncidentStateSnapshot | null = incidentId
      ? await db.incident.findUnique({
          where: { id: incidentId },
          select: { id: true, status: true, sourceIndex: true },
        })
      : null;
    const review = gate.reviewQuery({
      kind: 'query',
      spl,
      index,
      incidentId,
      incidentContext,
      incident,
      rebacAllowed: permission.allowed,
      indexKnown: isSplunkConfigured() || SPLUNK_INDEXES.some(i => i.name === index),
      actorRole: role,
      actorId: request.authUserId,
    });

    if (review.outcome === 'REFUSED') {
      return NextResponse.json({
        error: 'CHP_R0_HALT',
        reason: review.reason,
        r0: review.evaluation?.results ?? null,
      }, { status: 422 });
    }

    // Step 3: Execute query within bounds (real Splunk or simulation).
    let result;
    if (isSplunkConfigured()) {
      try {
        result = await runSplunkQuery(spl, index);
      } catch (err) {
        console.error('[Splunk] Real query failed, falling back to simulation:', err);
        result = simulateSplunkQuery(spl, index);
      }
    } else {
      result = simulateSplunkQuery(spl, index);
    }

    // Step 4: CHP foundation pass — the deterministic adversary scores the
    // bounded execution (guardrails 40 + bounded result 30 + state
    // corroboration 30; general floor 70 — no golden SOC source exists, so
    // incident-state binding stands in for golden parity).
    const assessment = gate.assessQueryExecution(
      review,
      { eventCount: result.eventCount },
      {
        kind: 'query',
        spl,
        index,
        incidentId,
        incidentContext,
        incident,
        rebacAllowed: permission.allowed,
        indexKnown: true,
        actorRole: role,
        actorId: request.authUserId,
      }
    );
    const lock = gate.humanLockVerdict(
      'query',
      review.decisionId,
      assessment,
      role,
      confirmed_by ?? null
    );

    if (lock.locked && lock.via === 'confirmed_by') {
      gate.record({
        kind: 'query',
        decisionId: review.decisionId,
        actorRole: role,
        actorId: request.authUserId,
        r0: review.evaluation,
        assessment,
        sessionStatus: 'LOCKED',
        confirmedBy: confirmed_by ?? null,
        spl,
        index,
        incidentId: incidentId ?? null,
        eventCount: result.eventCount,
      });
    } else if (!lock.locked && lock.via !== 'ledger') {
      // Sub-floor and unconfirmed: the decision is opened as PROVISIONAL_LOCK
      // and the result evidence is withheld until a named human confirmer
      // locks it through the decisions API.
      gate.record({
        kind: 'query',
        decisionId: review.decisionId,
        actorRole: role,
        actorId: request.authUserId,
        r0: review.evaluation,
        assessment,
        sessionStatus: 'PROVISIONAL_LOCK',
        confirmedBy: null,
        spl,
        index,
        incidentId: incidentId ?? null,
        eventCount: result.eventCount,
      });

      if (lock.via === 'refused') {
        return NextResponse.json({
          authorized: true,
          error: 'CHP_HUMAN_LOCK_REQUIRED',
          reason: lock.reason,
          decision_id: review.decisionId,
          r0: review.evaluation.results,
          foundation_score: assessment.score,
          adversary_findings: assessment.findings,
          confirm: 'POST /api/chp/decisions { decision_id, confirmed_by } with a soc_tier1/soc_tier2 session',
        }, { status: 403 });
      }
    }
    // lock.via === 'ledger' (or self_certified without a confirmer): the
    // durable record already exists — do not append a stale duplicate.

    // Step 5: Apply role-based data filtering
    if (role === 'contractor') {
      result.results = redactEvents(result.results);
    }

    return NextResponse.json({
      authorized: true,
      permission,
      sid: result.sid,
      results: result.results,
      eventCount: result.eventCount,
      runDurationMs: result.runDurationMs,
      chp: {
        decision_id: review.decisionId,
        session_status: lock.via === 'confirmed_by' ? 'LOCKED' : 'PROVISIONAL_LOCK',
        r0: review.evaluation.results,
        foundation_score: assessment.score,
        adversary_findings: assessment.findings,
        confirmed_by:
          lock.via === 'confirmed_by' ? confirmed_by ?? null : null,
      },
      note: role === 'contractor' ? 'Results have been redacted per contractor access policy' : undefined,
    });
  } catch (error) {
    // SpiceDB unavailable -> surface as 503 (handled by withAuth), never 500.
    if (error instanceof AuthZUnavailableError) throw error;
    return NextResponse.json({ error: 'Query execution failed' }, { status: 500 });
  }
});
