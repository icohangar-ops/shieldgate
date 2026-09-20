import { NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';
import { getSocChpGate, type IncidentStateSnapshot } from '@/lib/chp';
import type { UserRole } from '@/lib/authz-types';

// Allowlist of incident statuses a caller may transition an incident to.
// Anything outside this set is rejected (no free-form status injection).
const ALLOWED_STATUSES = new Set(['open', 'investigating', 'resolved', 'closed']);

// Roles permitted to mutate incident status. Mirrors the SpiceDB `resolve`
// permission intent: triage-only / external roles cannot change incident state.
const STATUS_WRITE_ROLES = new Set<UserRole>(['soc_tier2', 'sre', 'ai_agent']);

export const GET = withAuth(async (request: AuthenticatedRequest) => {
  const incidents = await db.incident.findMany({
    orderBy: { createdAt: 'desc' },
  });
  return NextResponse.json(incidents);
});

export const PATCH = withAuth(async (request: AuthenticatedRequest) => {
  try {
    // Authorization: only senior/automation roles may change incident status.
    if (!STATUS_WRITE_ROLES.has(request.authRole)) {
      return NextResponse.json(
        {
          error: 'PERMISSION_DENIED',
          reason: `Role '${request.authRole}' is not authorized to update incident status`,
          policy: 'least_privilege',
        },
        { status: 403 }
      );
    }

    const body = await request.json();
    const { id, status, confirmed_by } = body as {
      id: string;
      status: string;
      confirmed_by?: string;
    };

    if (!id || !status) {
      return NextResponse.json({ error: 'id and status are required' }, { status: 400 });
    }

    // Status allowlist: reject any value not in the known set.
    if (!ALLOWED_STATUSES.has(status)) {
      return NextResponse.json(
        {
          error: 'INVALID_STATUS',
          reason: `Status '${status}' is not allowed. Allowed: ${[...ALLOWED_STATUSES].join(', ')}`,
        },
        { status: 400 }
      );
    }

    // CHP gate: ReBAC (role check above) answers WHO may transition; CHP
    // answers WHETHER this transition should happen. Terminal statuses
    // (resolved/closed) are the destructive/containment class: while
    // SHIELDGATE_REQUIRE_HUMAN_LOCK is on (default), they never auto-execute.
    const gate = getSocChpGate();
    const existing = await db.incident.findUnique({
      where: { id },
      select: { id: true, status: true, sourceIndex: true },
    });
    const incident: IncidentStateSnapshot | null = existing
      ? {
          id: existing.id,
          status: existing.status,
          sourceIndex: existing.sourceIndex,
        }
      : null;

    const review = gate.reviewIncidentTransition({
      kind: 'incident_transition',
      incidentId: id,
      targetStatus: status,
      incident,
      rebacAllowed: true,
      actorRole: request.authRole,
      actorId: request.authUserId,
    });

    if (review.outcome === 'REFUSED') {
      return NextResponse.json(
        {
          error: 'CHP_R0_HALT',
          reason: review.reason,
          r0: review.evaluation?.results ?? null,
        },
        { status: 422 }
      );
    }

    const assessment = gate.assessIncidentTransition({
      kind: 'incident_transition',
      incidentId: id,
      targetStatus: status,
      incident,
      rebacAllowed: true,
      actorRole: request.authRole,
      actorId: request.authUserId,
    });
    if (assessment.fatal) {
      // A state contradiction must not persist even with a confirmer.
      return NextResponse.json(
        {
          error: 'CHP_STATE_CONTRADICTION',
          reason: assessment.fatal,
          decision_id: review.decisionId,
          adversary_findings: assessment.findings,
        },
        { status: 422 }
      );
    }

    const lock = gate.humanLockVerdict(
      'incident_transition',
      review.decisionId,
      assessment,
      request.authRole,
      confirmed_by ?? null
    );

    if (!lock.locked) {
      // Open the decision as PROVISIONAL_LOCK and refuse to apply the
      // transition until a named human confirmer locks it (fresh
      // confirmed_by from a human session, or a LOCKED record from
      // POST /api/chp/decisions).
      gate.record({
        kind: 'incident_transition',
        decisionId: review.decisionId,
        actorRole: request.authRole,
        actorId: request.authUserId,
        r0: review.evaluation,
        assessment,
        sessionStatus: 'PROVISIONAL_LOCK',
        confirmedBy: null,
        incidentId: id,
        targetStatus: status,
      });
      return NextResponse.json(
        {
          error: 'CHP_HUMAN_LOCK_REQUIRED',
          reason: lock.reason,
          decision_id: review.decisionId,
          r0: review.evaluation.results,
          foundation_score: assessment.score,
          adversary_findings: assessment.findings,
          confirm:
            'POST /api/chp/decisions { decision_id, confirmed_by } with a soc_tier2 session, then re-issue this request',
        },
        { status: 403 }
      );
    }

    const updated = await db.incident.update({
      where: { id },
      data: { status },
    });

    if (lock.via === 'confirmed_by') {
      gate.record({
        kind: 'incident_transition',
        decisionId: review.decisionId,
        actorRole: request.authRole,
        actorId: request.authUserId,
        r0: review.evaluation,
        assessment,
        sessionStatus: 'LOCKED',
        confirmedBy: confirmed_by ?? null,
        incidentId: id,
        targetStatus: status,
      });
    }
    // lock.via === 'ledger': the LOCKED record already exists — no duplicate.

    return NextResponse.json({
      ...updated,
      chp: {
        decision_id: review.decisionId,
        session_status: 'LOCKED',
        r0: review.evaluation.results,
        foundation_score: assessment.score,
        adversary_findings: assessment.findings,
        confirmed_by: lock.reason,
      },
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update incident' }, { status: 500 });
  }
});
