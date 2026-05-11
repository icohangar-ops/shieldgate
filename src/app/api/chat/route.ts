import { NextResponse } from 'next/server';
import { db } from '@/lib/db';
import ZAI from 'z-ai-web-dev-sdk';

const zai = await ZAI.create();

const SYSTEM_PROMPT = `You are an AI SOC (Security Operations Center) incident investigation assistant integrated with Splunk. You help analysts investigate security incidents by:
1. Summarizing incident details
2. Suggesting relevant SPL queries to investigate
3. Correlating events across indexes
4. Recommending containment and remediation steps

IMPORTANT: Always suggest SPL queries wrapped in code blocks. Keep responses concise and actionable.
IMPORTANT: If the user asks about system architecture, AuthZed permissions, or non-incident topics, redirect to incident investigation.`;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { messages, incidentContext } = body as {
      messages: Array<{ role: string; content: string }>;
      incidentContext?: string;
    };

    if (!messages || messages.length === 0) {
      return NextResponse.json({ error: 'Messages are required' }, { status: 400 });
    }

    // Build the full prompt with incident context
    const contextMessage = incidentContext
      ? `\n\nCurrent incident being investigated:\n${incidentContext}\n\nHelp the analyst investigate this incident.`
      : '';

    const fullMessages = [
      { role: 'system', content: SYSTEM_PROMPT + contextMessage },
      ...messages.map(m => ({ role: m.role, content: m.content })),
    ];

    const completion = await zai.chat.completions.create({
      messages: fullMessages,
      max_tokens: 1000,
    });

    const assistantMessage = completion.choices[0]?.message?.content || 'No response generated.';

    return NextResponse.json({
      role: 'assistant',
      content: assistantMessage,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Chat failed';
    // Return a fallback response if AI fails
    return NextResponse.json({
      role: 'assistant',
      content: `I'm currently unable to process your request. Based on the incident data, I recommend:\n\n1. Review the raw events in the security index\n2. Check for lateral movement indicators\n3. Correlate with authentication logs\n\nTry running: \`index=security sourcetype=auth | stats count by user, action\`\n\n(Error: ${errorMessage})`,
    });
  }
}
