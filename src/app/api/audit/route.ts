import { NextResponse } from 'next/server';
import { db } from '@/lib/db';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') || '50');
  const role = searchParams.get('role');

  const where = role ? { userRole: role } : {};

  const logs = await db.auditLog.findMany({
    where,
    orderBy: { timestamp: 'desc' },
    take: limit,
  });

  return NextResponse.json(logs);
}
