import { NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';

export const GET = withAuth(async (request: AuthenticatedRequest) => {
  const incidents = await db.incident.findMany({
    orderBy: { createdAt: 'desc' },
  });
  return NextResponse.json(incidents);
});

export const PATCH = withAuth(async (request: AuthenticatedRequest) => {
  try {
    const body = await request.json();
    const { id, status } = body as { id: string; status: string };

    if (!id || !status) {
      return NextResponse.json({ error: 'id and status are required' }, { status: 400 });
    }

    const updated = await db.incident.update({
      where: { id },
      data: { status },
    });

    return NextResponse.json(updated);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update incident' }, { status: 500 });
  }
});
