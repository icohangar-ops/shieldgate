import { NextResponse } from 'next/server';
import { SPLUNK_INDEXES, getIndexDetail } from '@/lib/splunk-sim';
import { checkToolPermission, type UserRole } from '@/lib/authz';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const role = searchParams.get('role') as UserRole;
  const name = searchParams.get('name');

  if (!role) {
    return NextResponse.json({ error: 'Role parameter required' }, { status: 400 });
  }

  // Check tool permission
  const toolPerm = checkToolPermission(role, 'splunk_get_indexes');
  if (!toolPerm.allowed) {
    return NextResponse.json({
      authorized: false,
      error: 'PERMISSION_DENIED',
      reason: toolPerm.reason,
    }, { status: 403 });
  }

  // Check index detail permission
  if (name) {
    const detailPerm = checkToolPermission(role, 'splunk_get_index_detail');
    if (!detailPerm.allowed) {
      return NextResponse.json({
        authorized: false,
        error: 'PERMISSION_DENIED',
        reason: detailPerm.reason,
      }, { status: 403 });
    }
    const detail = getIndexDetail(name);
    if (!detail) {
      return NextResponse.json({ error: 'Index not found' }, { status: 404 });
    }
    return NextResponse.json({ authorized: true, index: detail });
  }

  return NextResponse.json({ authorized: true, indexes: SPLUNK_INDEXES });
}
