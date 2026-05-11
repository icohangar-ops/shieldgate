import { db } from '@/lib/db';

const incidents = [
  {
    severity: 'critical',
    status: 'open',
    title: 'Active Data Exfiltration to Known C2 Infrastructure',
    description: 'Large-scale data exfiltration (4.7GB) detected from internal host 10.0.3.55 to known C2 server 45.33.32.156 over SSH (port 22). Duration: 1 hour. Host also communicated with update-service[.]ru on port 8080. Requires immediate containment and forensic analysis.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier2',
    rawEvent: JSON.stringify({ _time: '2026-05-12T13:30:45Z', _sourcetype: 'firewall', src_ip: '10.0.3.55', dest_ip: '45.33.32.156', bytes_out: '4.7GB', duration_sec: '3600', action: 'allowed' }),
  },
  {
    severity: 'critical',
    status: 'open',
    title: 'Brute Force Attack Against Production Bastion Host',
    description: '47 failed login attempts targeting svc_deploy account on bastion-prod-01 from source IP 203.0.113.42 (Beijing, CN) within 15 minutes. Account is a service account with deployment privileges. Credentials may be at risk.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier1',
    rawEvent: JSON.stringify({ _time: '2026-05-12T14:22:08Z', _sourcetype: 'auth', src_ip: '203.0.113.42', dest_host: 'bastion-prod-01', user: 'svc_deploy', attempt_count: '47', action: 'brute_force' }),
  },
  {
    severity: 'critical',
    status: 'investigating',
    title: 'Impossible Travel: Simultaneous Logins from US and Nigeria',
    description: 'User admin_jenkins authenticated from New York, US at 12:55:30Z and Lagos, NG at 12:59:15Z. This is physically impossible and strongly suggests credential compromise. Account has access to CI/CD pipelines and production deployments.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier2',
    rawEvent: JSON.stringify({ _time: '2026-05-12T12:55:30Z', _sourcetype: 'auth', user: 'admin_jenkins', geo_orig: 'New York, US', geo_dest: 'Lagos, NG', action: 'impossible_travel' }),
  },
  {
    severity: 'critical',
    status: 'open',
    title: 'Privilege Escalation Attempt on Production Database',
    description: 'User app_readonly attempted sudo /bin/bash on db-prod-primary at 14:10:22Z. This account should only have read-only SQL access. Potential lateral movement or insider threat. Database contains PII and financial records.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier2',
    rawEvent: JSON.stringify({ _time: '2026-05-12T14:10:22Z', _sourcetype: 'auth', src_ip: '10.0.5.12', dest_host: 'db-prod-primary', user: 'app_readonly', action: 'sudo_attempt', sudo_command: '/bin/bash' }),
  },
  {
    severity: 'critical',
    status: 'open',
    title: 'OOM Kill on Payment Service Pod in Production',
    description: 'payment-service-7d4f9 pod on k8s-worker-03 was OOM Killed. Memory usage reached 6.2Gi against a 4Gi limit. Active transactions may have been lost. Checkout flow impacted with 2.1% error rate and P99 latency of 3200ms.',
    sourceIndex: 'observability',
    assignedTeam: 'sre',
    rawEvent: JSON.stringify({ _time: '2026-05-12T14:18:00Z', _sourcetype: 'log', host: 'k8s-worker-03', pod: 'payment-service-7d4f9', namespace: 'production', memory_limit: '4Gi', memory_used: '6.2Gi' }),
  },
  {
    severity: 'high',
    status: 'open',
    title: 'SQL Injection Attack Blocked by WAF',
    description: 'IDS detected and blocked SQL injection attempt from 172.16.0.101 targeting API endpoint 10.0.2.20:8443. Signature: ET SQL_INJECTION. Payload hash: a1b2c3d4. This is the 3rd attempt from this source in 24 hours.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier1',
    rawEvent: JSON.stringify({ _time: '2026-05-12T14:15:33Z', _sourcetype: 'ids', src_ip: '172.16.0.101', dest_ip: '10.0.2.20', signature: 'ET SQL_INJECTION', action: 'blocked' }),
  },
  {
    severity: 'high',
    status: 'open',
    title: 'SOC2 Control Failure: Unencrypted S3 Bucket Detected',
    description: 'Compliance scan detected S3 bucket s3://legacy-backups is not encrypted at rest. SOC2 control CC6.1 (Logical Access Security) is failing. Bucket contains backup data that may include sensitive customer information.',
    sourceIndex: 'compliance',
    assignedTeam: 'soc_tier2',
    rawEvent: JSON.stringify({ _time: '2026-05-12T13:30:00Z', _sourcetype: 'compliance', resource: 's3://legacy-backups', control_id: 'CC6.1', encryption_status: 'not_encrypted' }),
  },
  {
    severity: 'high',
    status: 'open',
    title: 'Pass-the-Hash Lateral Movement Detected',
    description: 'Network IDS blocked SMB pass-the-hash attempt from 10.0.4.22 to 10.0.4.30. NTLM hash replay detected. Source host may be compromised. Both hosts are in the same VLAN segment with unrestricted east-west traffic.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier2',
    rawEvent: JSON.stringify({ _time: '2026-05-12T13:10:15Z', _sourcetype: 'ids', src_ip: '10.0.4.22', dest_ip: '10.0.4.30', protocol: 'SMB', action: 'blocked', ntlm_hash: 'detected' }),
  },
  {
    severity: 'high',
    status: 'open',
    title: 'API Token Used After Revocation',
    description: 'Deleted user account (deleted_user) successfully authenticated using token tk_revoked_8847 against api-gateway from IP 198.51.100.23. Token revocation did not propagate. Potential authentication system vulnerability.',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier1',
    rawEvent: JSON.stringify({ _time: '2026-05-12T13:30:45Z', _sourcetype: 'auth', src_ip: '198.51.100.23', dest_host: 'api-gateway', user: 'deleted_user', token_id: 'tk_revoked_8847' }),
  },
  {
    severity: 'medium',
    status: 'investigating',
    title: 'CPU Utilization Spike on K8s Worker Node',
    description: 'k8s-worker-07 reported 94.2% CPU utilization exceeding the 80% warning threshold. Potential cause: scheduled batch job overlap or resource-intensive pod scheduling. Current pod distribution may need rebalancing.',
    sourceIndex: 'observability',
    assignedTeam: 'sre',
    rawEvent: JSON.stringify({ _time: '2026-05-12T14:22:00Z', _sourcetype: 'metric', host: 'k8s-worker-07', metric: 'cpu_percent', value: '94.2' }),
  },
  {
    severity: 'medium',
    status: 'open',
    title: 'Unusual Outbound Connection to Suspicious Endpoint',
    description: 'Host 10.0.3.55 established connection to 91.189.89.42:443 with 2.3MB outbound transfer. Destination IP has no associated business justification. Same host involved in C2 exfiltration incident (INC-001).',
    sourceIndex: 'security',
    assignedTeam: 'soc_tier1',
    rawEvent: JSON.stringify({ _time: '2026-05-12T14:20:01Z', _sourcetype: 'firewall', src_ip: '10.0.3.55', dest_ip: '91.189.89.42', bytes_out: '2.3MB' }),
  },
  {
    severity: 'high',
    status: 'open',
    title: 'HIPAA Violation: Bulk Patient Record Export',
    description: 'User dr_temp exported 15,400 patient records from the EHR system. This exceeds the normal daily average of 12 records by 1283x. Access was from a temporary account created 2 days ago. Requires immediate investigation.',
    sourceIndex: 'compliance',
    assignedTeam: 'soc_tier2',
    rawEvent: JSON.stringify({ _time: '2026-05-12T13:00:00Z', _sourcetype: 'compliance', user: 'dr_temp', record_count: '15400', regulation: 'HIPAA' }),
  },
];

async function seed() {
  console.log('Seeding database...');

  // Clear existing data
  await db.incident.deleteMany();
  await db.auditLog.deleteMany();

  // Create incidents
  for (const inc of incidents) {
    await db.incident.create({ data: inc });
  }

  console.log(`Created ${incidents.length} incidents`);

  // Create some initial audit log entries to show activity
  const initialAudits = [
    { userId: 'soc_tier1_analyst', userRole: 'soc_tier1', action: 'splunk_get_indexes', resource: 'all', decision: 'ALLOW', reason: 'All roles can list indexes' },
    { userId: 'soc_tier1_analyst', userRole: 'soc_tier1', action: 'splunk_run_query', resource: 'index:security', decision: 'ALLOW', reason: 'Limited SPL only (no subsearches, no eval with eval expressions)' },
    { userId: 'contractor_external', userRole: 'contractor', action: 'splunk_run_query', resource: 'index:security', decision: 'DENY', reason: 'Contractors cannot execute ad-hoc queries' },
    { userId: 'sre_engineer', userRole: 'sre', action: 'splunk_run_query', resource: 'index:security', decision: 'DENY', reason: "Role 'sre' does not have access to index 'security'. Per SpiceDB policy, index membership is required." },
    { userId: 'ai_agent_01', userRole: 'ai_agent', action: 'splunk_get_alerts', resource: 'all', decision: 'ALLOW', reason: 'Permission granted by role policy' },
    { userId: 'soc_tier2_lead', userRole: 'soc_tier2', action: 'splunk_run_query', resource: 'index:compliance', decision: 'ALLOW', reason: 'Permission granted by role policy' },
  ];

  for (const audit of initialAudits) {
    await db.auditLog.create({ data: audit });
  }

  console.log(`Created ${initialAudits.length} audit log entries`);
  console.log('Seed complete!');
}

seed()
  .catch(console.error)
  .finally(() => process.exit(0));
