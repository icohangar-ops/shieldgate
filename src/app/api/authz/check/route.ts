import { NextResponse } from 'next/server';
import { checkToolPermission, checkIndexPermission, getRolePermissions, type UserRole } from '@/lib/authz';
import { db } from '@/lib/db';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { role, action, resource, permissionType } = body as {
      role: UserRole;
      action: string;
      resource?: string;
      permissionType: 'tool' | 'index';
    };

    let decision;

    if (permissionType === 'tool') {
      const index = resource?.startsWith('index:') ? resource.slice(6) : undefined;
      decision = await checkToolPermission(role, action, index);
    } else {
      const index = resource || '';
      decision = await checkIndexPermission(role, index, (action as 'read' | 'query') || 'read');
    }

    // Log the decision to the audit trail
    await db.auditLog.create({
      data: {
        userId: `${role}_user`,
        userRole: role,
        action,
        resource: resource || '',
        decision: decision.allowed ? 'ALLOW' : 'DENY',
        reason: decision.reason,
      },
    });

    return NextResponse.json({
      allowed: decision.allowed,
      reason: decision.reason,
      policy: decision.policy,
      timestamp: decision.timestamp,
    });
  } catch (error) {
    return NextResponse.json({ error: 'Permission check failed' }, { status: 500 });
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const role = searchParams.get('role') as UserRole;

  if (!role) {
    return NextResponse.json({ error: 'Role parameter required' }, { status: 400 });
  }

  const permissions = getRolePermissions(role);
  return NextResponse.json(permissions);
}
