import { NextResponse } from 'next/server';
import { SPLUNK_ALERTS } from '@/lib/splunk-sim';
import { isSplunkConfigured, getSplunkAlerts } from '@/lib/splunk-client';
import { checkToolPermission, type UserRole } from '@/lib/authz';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const role = searchParams.get('role') as UserRole;

  if (!role) {
    return NextResponse.json({ error: 'Role parameter required' }, { status: 400 });
  }

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
}
