import { NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';

export const GET = withAuth(async (request: AuthenticatedRequest) => {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') || '50');
  const role = request.authRole;

  const where = role ? { userRole: role } : {};

  const logs = await db.auditLog.findMany({
    where,
    orderBy: { timestamp: 'desc' },
    take: limit,
  });

  return NextResponse.json(logs);
});
