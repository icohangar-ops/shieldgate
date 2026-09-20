import { NextResponse } from 'next/server';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';
import { getSocChpGate, ChpRejection } from '@/lib/chp';

/**
 * CHP decision ledger API — surfaces the SOC gate's decisions through the
 * repo's existing interface (authenticated Next.js routes).
 *
 *   GET  /api/chp/decisions          newest-first records, integrity-checked
 *   GET  /api/chp/decisions?id=...   single decision (latest record)
 *   POST /api/chp/decisions          { decision_id, confirmed_by } — third-
 *                                    party validation: PROVISIONAL_LOCK ->
 *                                    LOCKED. The confirmer's role comes from
 *                                    their own authenticated session, so an
 *                                    ai_agent can never self-confirm.
 */

export const GET = withAuth(async (request: AuthenticatedRequest) => {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const gate = getSocChpGate();

  if (id) {
    const record = gate.decision(id);
    if (!record) {
      return NextResponse.json({ error: 'DECISION_NOT_FOUND' }, { status: 404 });
    }
    return NextResponse.json(record);
  }

  const limitParam = parseInt(searchParams.get('limit') || '50', 10);
  const limit = Number.isFinite(limitParam) && limitParam > 0 ? limitParam : 50;
  return NextResponse.json({
    decisions: gate.list(limit),
    require_human_lock: process.env.SHIELDGATE_REQUIRE_HUMAN_LOCK !== '0',
  });
});

export const POST = withAuth(async (request: AuthenticatedRequest) => {
  try {
    const body = await request.json();
    const { decision_id, confirmed_by } = body as {
      decision_id: string;
      confirmed_by?: string;
    };

    if (!decision_id) {
      return NextResponse.json({ error: 'decision_id is required' }, { status: 400 });
    }

    const confirmer = confirmed_by?.trim() || request.authUserId;
    const record = getSocChpGate().confirm(decision_id, confirmer, request.authRole);

    return NextResponse.json({
      decision_id: record.decision_id,
      session_status: record.session_status,
      confirmed_by: record.confirmed_by,
      supersedes: record.supersedes,
      envelope_valid: record.envelope_valid,
      integrity_valid: record.integrity_valid,
    });
  } catch (error) {
    if (error instanceof ChpRejection) {
      return NextResponse.json(
        { error: 'CHP_CONFIRMATION_REJECTED', reason: error.message },
        { status: 403 }
      );
    }
    return NextResponse.json({ error: 'Confirmation failed' }, { status: 500 });
  }
});
