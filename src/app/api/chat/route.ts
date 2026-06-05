import { NextResponse } from 'next/server';
import { withAuth, type AuthenticatedRequest } from '@/lib/auth-middleware';

const SYSTEM_PROMPT = `You are an AI SOC (Security Operations Center) incident investigation assistant integrated with Splunk. You help analysts investigate security incidents by:
1. Summarizing incident details
2. Suggesting relevant SPL queries to investigate
3. Correlating events across indexes
4. Recommending containment and remediation steps

IMPORTANT: Always suggest SPL queries wrapped in code blocks. Keep responses concise and actionable.
IMPORTANT: If the user asks about system architecture, AuthZed permissions, or non-incident topics, redirect to incident investigation.`;

async function getAiClient() {
  try {
    const ZAI = (await import('z-ai-web-dev-sdk')).default;
    return await ZAI.create();
  } catch {
    return null;
  }
}

export const POST = withAuth(async (request: AuthenticatedRequest) => {
  try {
    const body = await request.json();
    const { messages, incidentContext } = body as {
      messages: Array<{ role: string; content: string }>;
      incidentContext?: string;
    };

    if (!messages || messages.length === 0) {
      return NextResponse.json({ error: 'Messages are required' }, { status: 400 });
    }

    const contextMessage = incidentContext
      ? `\n\nCurrent incident being investigated:\n${incidentContext}\n\nHelp the analyst investigate this incident.`
      : '';

    const fullMessages = [
      { role: 'system', content: SYSTEM_PROMPT + contextMessage },
      ...messages.map(m => ({ role: m.role, content: m.content })),
    ];

    const zai = await getAiClient();
    if (zai) {
      const completion = await zai.chat.completions.create({
        messages: fullMessages,
        max_tokens: 1000,
      });

      const assistantMessage = completion.choices[0]?.message?.content || 'No response generated.';
      return NextResponse.json({ role: 'assistant', content: assistantMessage });
    }

    const lastUserMsg = messages[messages.length - 1]?.content || '';
    return NextResponse.json({
      role: 'assistant',
      content: generateFallbackResponse(lastUserMsg, incidentContext),
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Chat failed';
    return NextResponse.json({
      role: 'assistant',
      content: `I'm currently unable to process your request. Based on the incident data, I recommend:\n\n1. Review the raw events in the security index\n2. Check for lateral movement indicators\n3. Correlate with authentication logs\n\nTry running: \`index=security sourcetype=auth | stats count by user, action\`\n\n(Error: ${errorMessage})`,
    });
  }
});

function generateFallbackResponse(userMsg: string, context?: string): string {
  const lower = userMsg.toLowerCase();

  if (lower.includes('exfiltration') || lower.includes('data loss')) {
    return `Based on the incident context, here are recommended investigation queries:\n\n\`\`\`spl\nindex=security sourcetype=firewall action=allowed bytes_out>1000000 | stats sum(bytes_out) by src_ip, dest_ip\n\`\`\`\n\n\`\`\`spl\nindex=security sourcetype=firewall dest_port=22 OR dest_port=443 | timechart count by dest_ip\n\`\`\`\n\nLook for unusual outbound data volumes and connections to unfamiliar IPs.`;
  }

  if (lower.includes('brute force') || lower.includes('login')) {
    return `To investigate potential brute force activity:\n\n\`\`\`spl\nindex=security sourcetype=auth action=login_failure | stats count by src_ip, user | where count > 10\n\`\`\`\n\n\`\`\`spl\nindex=security sourcetype=auth | transaction src_ip maxpause=5m | where eventcount > 20\n\`\`\`\n\nFocus on high-frequency failed logins from single IPs.`;
  }

  return `I recommend starting with these general investigation queries:\n\n\`\`\`spl\nindex=security severity=critical | stats count by action, src_ip | sort -count\n\`\`\`\n\n\`\`\`spl\nindex=security | timechart span=5m count by sourcetype\n\`\`\`\n\nPlease describe what specific aspect of the incident you'd like to investigate.`;
}
