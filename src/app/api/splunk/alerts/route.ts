import { NextResponse } from 'next/server';
import { SPLUNK_ALERTS } from '@/lib/splunk-sim';
import { isSplunkConfigured, getSplunkAlerts } from '@/lib/splunk-client';
import { checkToolPermission } from '@/lib/authz';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';

export const GET = withAuth(async (request: AuthenticatedRequest) => {
  const role = request.authRole;

  const perm = await checkToolPermission(role, 'splunk_get_alerts');
  if (!perm.allowed) {
    return NextResponse.json({
      authorized: false,
      error: 'PERMISSION_DENIED',
      reason: perm.reason,
    }, { status: 403 });
  }

  let alerts;
  if (isSplunkConfigured()) {
    try {
      alerts = await getSplunkAlerts();
    } catch {
      alerts = SPLUNK_ALERTS;
    }
  } else {
    alerts = SPLUNK_ALERTS;
  }

  // SRE only sees observability alerts
  if (role === 'sre') {
    alerts = alerts.filter(a =>
      a.name.toLowerCase().includes('k8s') ||
      a.name.toLowerCase().includes('cpu') ||
      a.name.toLowerCase().includes('oom') ||
      a.name.toLowerCase().includes('kafka') ||
      a.name.toLowerCase().includes('redis') ||
      a.name.toLowerCase().includes('latency') ||
      a.name.toLowerCase().includes('payment')
    );
  }

  return NextResponse.json({ authorized: true, alerts });
});
