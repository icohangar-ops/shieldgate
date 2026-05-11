// Splunk Simulation Layer
// In production, this connects to real Splunk via splunk-sdk or MCP Server
// For the hackathon demo, we simulate Splunk responses with synthetic security data

export interface SplunkEvent {
  _time: string;
  _sourcetype: string;
  _index: string;
  _raw: string;
  [key: string]: string;
}

export interface SplunkQueryResult {
  sid: string;
  results: SplunkEvent[];
  eventCount: number;
  runDurationMs: number;
}

export interface SplunkIndex {
  name: string;
  totalEvents: number;
  dataSizeGB: string;
  retentionDays: number;
  status: 'online' | 'offline';
}

export interface SplunkAlert {
  id: string;
  name: string;
  severity: string;
  condition: string;
  triggeredAt: string;
  status: string;
}

// Synthetic security events
const SECURITY_EVENTS: SplunkEvent[] = [
  { _time: '2026-05-12T14:23:11Z', _sourcetype: 'auth', _index: 'security', _raw: 'Failed login attempt detected', src_ip: '203.0.113.42', dest_host: 'bastion-prod-01', user: 'svc_deploy', action: 'login_failure', severity: 'high', geo: 'Beijing, CN' },
  { _time: '2026-05-12T14:22:08Z', _sourcetype: 'auth', _index: 'security', _raw: 'Brute force detection triggered', src_ip: '203.0.113.42', dest_host: 'bastion-prod-01', user: 'svc_deploy', action: 'brute_force', severity: 'critical', geo: 'Beijing, CN', attempt_count: '47' },
  { _time: '2026-05-12T14:20:01Z', _sourcetype: 'firewall', _index: 'security', _raw: 'Unusual outbound connection', src_ip: '10.0.3.55', dest_ip: '91.189.89.42', dest_port: '443', protocol: 'TCP', action: 'allowed', severity: 'medium', bytes_out: '2.3MB' },
  { _time: '2026-05-12T14:15:33Z', _sourcetype: 'ids', _index: 'security', _raw: 'IDS alert: SQL injection attempt blocked', src_ip: '172.16.0.101', dest_ip: '10.0.2.20', dest_port: '8443', signature: 'ET SQL_INJECTION', action: 'blocked', severity: 'high', payload_hash: 'a1b2c3d4' },
  { _time: '2026-05-12T14:10:22Z', _sourcetype: 'auth', _index: 'security', _raw: 'Privilege escalation attempt on DB server', src_ip: '10.0.5.12', dest_host: 'db-prod-primary', user: 'app_readonly', action: 'sudo_attempt', severity: 'critical', sudo_command: '/bin/bash' },
  { _time: '2026-05-12T13:55:11Z', _sourcetype: 'auth', _index: 'security', _raw: 'New user account created outside change window', src_ip: '10.0.1.5', dest_host: 'ad-primary', user: 'svc_deploy', action: 'user_created', severity: 'high', new_user: 'temp_admin' },
  { _time: '2026-05-12T13:45:00Z', _sourcetype: 'auth', _index: 'security', _raw: 'API token used after revocation', src_ip: '198.51.100.23', dest_host: 'api-gateway', user: 'deleted_user', action: 'token_used_post_revoke', severity: 'critical', token_id: 'tk_revoked_8847' },
  { _time: '2026-05-12T13:30:45Z', _sourcetype: 'firewall', _index: 'security', _raw: 'Large data exfiltration detected', src_ip: '10.0.3.55', dest_ip: '45.33.32.156', dest_port: '22', protocol: 'TCP', action: 'allowed', severity: 'critical', bytes_out: '4.7GB', duration_sec: '3600' },
  { _time: '2026-05-12T13:20:00Z', _sourcetype: 'auth', _index: 'security', _raw: 'Kerberos ticket from unauthorized service', src_ip: '10.0.1.200', dest_host: 'dc-primary', user: 'svc_backup', action: 'krb_tamper', severity: 'high', ticket_anomaly: 'encryption_type_mismatch' },
  { _time: '2026-05-12T13:10:15Z', _sourcetype: 'ids', _index: 'security', _raw: ' lateral movement: pass-the-hash attempt', src_ip: '10.0.4.22', dest_ip: '10.0.4.30', protocol: 'SMB', action: 'blocked', severity: 'critical', ntlm_hash: 'detected' },
  { _time: '2026-05-12T12:55:30Z', _sourcetype: 'auth', _index: 'security', _raw: 'Impossible travel: login from 2 countries within 5 min', user: 'admin_jenkins', geo_orig: 'New York, US', geo_dest: 'Lagos, NG', severity: 'critical', action: 'impossible_travel' },
  { _time: '2026-05-12T12:40:00Z', _sourcetype: 'firewall', _index: 'security', _raw: 'Connection to known C2 server detected', src_ip: '10.0.3.55', dest_ip: '91.189.89.42', dest_port: '8080', protocol: 'TCP', action: 'allowed', severity: 'critical', c2_domain: 'update-service[.]ru' },
];

const OBSERVABILITY_EVENTS: SplunkEvent[] = [
  { _time: '2026-05-12T14:22:00Z', _sourcetype: 'metric', _index: 'observability', _raw: 'CPU utilization spike on k8s-worker-07', host: 'k8s-worker-07', metric: 'cpu_percent', value: '94.2', threshold: '80', severity: 'warning' },
  { _time: '2026-05-12T14:18:00Z', _sourcetype: 'log', _index: 'observability', _raw: 'OOM Kill on payment-service pod', host: 'k8s-worker-03', pod: 'payment-service-7d4f9', namespace: 'production', severity: 'critical', memory_limit: '4Gi', memory_used: '6.2Gi' },
  { _time: '2026-05-12T14:15:00Z', _sourcetype: 'metric', _index: 'observability', _raw: 'Disk latency exceeds SLA on db-replica-02', host: 'db-replica-02', metric: 'disk_latency_ms', value: '245', threshold: '100', severity: 'warning' },
  { _time: '2026-05-12T14:10:00Z', _sourcetype: 'apm', _index: 'observability', _raw: 'P99 latency spike on checkout API', service: 'checkout-api', endpoint: '/api/v2/checkout', latency_p99_ms: '3200', latency_p50_ms: '45', error_rate: '2.1%', severity: 'critical' },
  { _time: '2026-05-12T14:05:00Z', _sourcetype: 'log', _index: 'observability', _raw: 'Certificate expiring in 48 hours', host: 'api-gateway', cert_cn: '*.cubiczan.com', expiry: '2026-05-14T00:00:00Z', severity: 'warning' },
  { _time: '2026-05-12T13:55:00Z', _sourcetype: 'metric', _index: 'observability', _raw: 'Kafka consumer lag increasing', topic: 'order-events', consumer_group: 'inventory-service', lag: '152000', severity: 'warning' },
  { _time: '2026-05-12T13:50:00Z', _sourcetype: 'log', _index: 'observability', _raw: 'Redis connection pool exhausted', host: 'cache-primary', pool_size: '200', active_connections: '200', wait_queue: '45', severity: 'critical' },
  { _time: '2026-05-12T13:40:00Z', _sourcetype: 'metric', _index: 'observability', _raw: 'Network throughput anomaly on core switch', host: 'sw-core-01', metric: 'throughput_gbps', value: '85.3', baseline: '42.1', severity: 'warning' },
];

const COMPLIANCE_EVENTS: SplunkEvent[] = [
  { _time: '2026-05-12T14:00:00Z', _sourcetype: 'compliance', _index: 'compliance', _raw: 'GDPR data access without justification logged', user: 'analyst_bob', action: 'pii_access', resource: 'customer_db', regulation: 'GDPR', severity: 'high' },
  { _time: '2026-05-12T13:30:00Z', _sourcetype: 'compliance', _index: 'compliance', _raw: 'SOC2 control failure: unencrypted S3 bucket detected', resource: 's3://legacy-backups', control_id: 'CC6.1', severity: 'critical', encryption_status: 'not_encrypted' },
  { _time: '2026-05-12T13:00:00Z', _sourcetype: 'compliance', _index: 'compliance', _raw: 'HIPAA access log anomaly: bulk patient record export', user: 'dr_temp', action: 'bulk_export', record_count: '15400', regulation: 'HIPAA', severity: 'critical' },
  { _time: '2026-05-12T12:30:00Z', _sourcetype: 'compliance', _index: 'compliance', _raw: 'PCI-DSS: cardholder data in log files detected', host: 'app-server-02', file_path: '/var/log/app/debug.log', pattern: 'card_number', regulation: 'PCI-DSS', severity: 'critical' },
  { _time: '2026-05-12T12:00:00Z', _sourcetype: 'compliance', _index: 'compliance', _raw: 'FISMA audit trail gap: admin action without MFA verified', user: 'admin_root', action: 'config_change', resource: 'firewall-rules', severity: 'high', mfa_verified: 'false' },
];

const ALL_EVENTS: Record<string, SplunkEvent[]> = {
  security: SECURITY_EVENTS,
  observability: OBSERVABILITY_EVENTS,
  compliance: COMPLIANCE_EVENTS,
};

// Simulated indexes
export const SPLUNK_INDEXES: SplunkIndex[] = [
  { name: 'security', totalEvents: 2458934, dataSizeGB: '182.3', retentionDays: 90, status: 'online' },
  { name: 'observability', totalEvents: 8945123, dataSizeGB: '445.7', retentionDays: 30, status: 'online' },
  { name: 'compliance', totalEvents: 523411, dataSizeGB: '23.1', retentionDays: 365, status: 'online' },
  { name: 'hr', totalEvents: 98432, dataSizeGB: '4.2', retentionDays: 365, status: 'online' },
  { name: 'prod', totalEvents: 34782110, dataSizeGB: '1,892.4', retentionDays: 14, status: 'online' },
];

// Simulated alerts
export const SPLUNK_ALERTS: SplunkAlert[] = [
  { id: 'alert-001', name: 'Critical: Data Exfiltration Detected', severity: 'critical', condition: 'bytes_out > 1GB AND duration_sec > 1800', triggeredAt: '2026-05-12T13:30:00Z', status: 'triggered' },
  { id: 'alert-002', name: 'High: Brute Force Attack', severity: 'high', condition: 'action=brute_force', triggeredAt: '2026-05-12T14:22:08Z', status: 'triggered' },
  { id: 'alert-003', name: 'Critical: OOM Kill on Payment Service', severity: 'critical', condition: 'severity=critical AND namespace=production', triggeredAt: '2026-05-12T14:18:00Z', status: 'triggered' },
  { id: 'alert-004', name: 'Warning: CPU Spike on K8s Worker', severity: 'warning', condition: 'cpu_percent > 90', triggeredAt: '2026-05-12T14:22:00Z', status: 'acknowledged' },
  { id: 'alert-005', name: 'Critical: C2 Server Communication', severity: 'critical', condition: 'c2_domain=*', triggeredAt: '2026-05-12T12:40:00Z', status: 'triggered' },
  { id: 'alert-006', name: 'High: SOC2 Control Failure', severity: 'high', condition: 'control_id=CC6.1', triggeredAt: '2026-05-12T13:30:00Z', status: 'investigating' },
  { id: 'alert-007', name: 'Critical: Impossible Travel Detected', severity: 'critical', condition: 'action=impossible_travel', triggeredAt: '2026-05-12T12:55:30Z', status: 'triggered' },
  { id: 'alert-008', name: 'Warning: Kafka Consumer Lag', severity: 'warning', condition: 'lag > 100000', triggeredAt: '2026-05-12T13:55:00Z', status: 'acknowledged' },
];

// Simulate running a Splunk SPL query
export function simulateSplunkQuery(spl: string, index?: string): SplunkQueryResult {
  const sid = `sid_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  let results: SplunkEvent[] = [];

  // Parse simple SPL queries to return relevant results
  const lowerSpl = spl.toLowerCase();

  if (index && ALL_EVENTS[index]) {
    results = [...ALL_EVENTS[index]];
  } else {
    // Return events from all available indexes
    results = Object.values(ALL_EVENTS).flat();
  }

  // Apply filters based on SPL keywords
  if (lowerSpl.includes('severity=critical') || lowerSpl.includes('severity="critical"')) {
    results = results.filter(e => e.severity === 'critical');
  }
  if (lowerSpl.includes('severity=high') || lowerSpl.includes('severity="high"')) {
    results = results.filter(e => e.severity === 'high');
  }
  if (lowerSpl.includes('action=') && !lowerSpl.includes('action=*')) {
    const actionMatch = spl.match(/action[=:]"?(\w+)"?/i);
    if (actionMatch) {
      results = results.filter(e => e.action === actionMatch[1]);
    }
  }
  if (lowerSpl.includes('sourcetype=') && !lowerSpl.includes('sourcetype=*')) {
    const stMatch = spl.match(/sourcetype[=:]"?(\w+)"?/i);
    if (stMatch) {
      results = results.filter(e => e._sourcetype === stMatch[1]);
    }
  }
  if (lowerSpl.includes('stats count')) {
    // For stats, just return fewer representative results
    results = results.slice(0, 3);
  }

  // Apply stats/transforms
  if (lowerSpl.includes('stats count by') || lowerSpl.includes('chart count by')) {
    const fieldMatch = lowerSpl.match(/(?:stats|chart) count by\s+(\w+)/);
    if (fieldMatch) {
      const grouped = new Map<string, number>();
      results.forEach(e => {
        const key = e[fieldMatch[1]] || 'unknown';
        grouped.set(key, (grouped.get(key) || 0) + 1);
      });
      results = Array.from(grouped.entries()).map(([k, v]) => ({
        _time: new Date().toISOString(),
        _sourcetype: 'stats',
        _index: index || 'summary',
        _raw: `${fieldMatch[1]}=${k}, count=${v}`,
        [fieldMatch[1]]: k,
        count: String(v),
      }));
    }
  }

  // Sort by time descending
  results.sort((a, b) => b._time.localeCompare(a._time));

  // Limit results
  results = results.slice(0, 20);

  return {
    sid,
    results,
    eventCount: results.length,
    runDurationMs: Math.floor(Math.random() * 200) + 50,
  };
}

// Redact events for contractor view
export function redactEvents(events: SplunkEvent[]): SplunkEvent[] {
  const sensitiveFields = ['src_ip', 'dest_ip', 'user', 'geo', 'payload_hash', 'ntlm_hash', 'ticket_anomaly', 'password'];
  return events.map(e => {
    const redacted = { ...e };
    sensitiveFields.forEach(f => {
      if (redacted[f]) {
        redacted[f] = '[REDACTED]';
      }
    });
    redacted._raw = redacted._raw.replace(/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/g, '[IP_REDACTED]');
    return redacted;
  });
}

// Get index detail
export function getIndexDetail(name: string): SplunkIndex | undefined {
  return SPLUNK_INDEXES.find(i => i.name === name);
}
