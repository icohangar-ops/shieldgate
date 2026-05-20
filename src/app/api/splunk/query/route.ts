import { NextResponse } from 'next/server';
import { simulateSplunkQuery, redactEvents } from '@/lib/splunk-sim';
import { isSplunkConfigured, runSplunkQuery } from '@/lib/splunk-client';
import { checkToolPermission, type UserRole } from '@/lib/authz';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { spl, index, role } = body as { spl: string; index?: string; role: UserRole };

    if (!spl || !role) {
      return NextResponse.json({ error: 'spl and role are required' }, { status: 400 });
    }

    // Step 1: AuthZed permission check
    const permission = await checkToolPermission(role, 'splunk_run_query', index);
    if (!permission.allowed) {
      return NextResponse.json({
        authorized: false,
        error: 'PERMISSION_DENIED',
        reason: permission.reason,
        policy: permission.policy,
      }, { status: 403 });
    }

    // Step 2: Execute query (real Splunk or simulation)
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

    // Step 3: Apply role-based data filtering
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
      note: role === 'contractor' ? 'Results have been redacted per contractor access policy' : undefined,
    });
  } catch (error) {
    return NextResponse.json({ error: 'Query execution failed' }, { status: 500 });
  }
}
