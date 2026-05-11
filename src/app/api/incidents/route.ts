import { NextResponse } from 'next/server';
import { db } from '@/lib/db';

export async function GET() {
  const incidents = await db.incident.findMany({
    orderBy: { createdAt: 'desc' },
  });
  return NextResponse.json(incidents);
}

export async function PATCH(request: Request) {
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
}
