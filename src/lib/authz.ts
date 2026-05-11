// AuthZed Permission Simulation Layer
// In production, this would use @authzed/authzed-node gRPC client
// For the hackathon demo, we simulate SpiceDB permission checks

export type UserRole = 'soc_tier1' | 'soc_tier2' | 'sre' | 'contractor' | 'ai_agent';

export interface PermissionCheck {
  resource: { objectType: string; objectId: string };
  permission: string;
  subject: { objectType: string; objectId: string };
}

export interface AuthZDecision {
  allowed: boolean;
  reason: string;
  policy: string;
  timestamp: string;
}

// SpiceDB-style schema (displayed in UI, logic implemented below)
export const AUTHZED_SCHEMA = `
definition user {}

definition team {
  relation member: user
  permission member_access = member
}

definition splunk_index {
  relation viewer: team | user
  relation querier: team | user
  relation admin: user

  permission read   = viewer + querier + admin
  permission query  = querier + admin
  permission manage = admin
}

definition splunk_tool {
  relation allowed_role: team
  relation allowed_user: user
  relation restricted: user

  permission execute = allowed_user + allowed_role->member_access - restricted
}

definition incident {
  relation index: splunk_index
  relation assigned_team: team
  relation viewer: user
  relation resolver: user

  permission view     = viewer + assigned_team->member_access + resolver + index->read
  permission resolve  = resolver
}
`;

// Role → Index permissions
const INDEX_PERMISSIONS: Record<UserRole, Record<string, string[]>> = {
  soc_tier1: {
    security: ['read'],
    observability: [],
    compliance: ['read'],
    hr: [],
    prod: [],
  },
  soc_tier2: {
    security: ['read', 'query'],
    observability: ['read'],
    compliance: ['read', 'query'],
    hr: [],
    prod: [],
  },
  sre: {
    security: [],
    observability: ['read', 'query'],
    compliance: [],
    hr: [],
    prod: ['read', 'query'],
  },
  contractor: {
    security: ['read'], // redacted
    observability: [],
    compliance: [],
    hr: [],
    prod: [],
  },
  ai_agent: {
    security: ['read'],
    observability: ['read'],
    compliance: ['read'],
    hr: [],
    prod: ['read'],
  },
};

// Role → Tool permissions
const TOOL_PERMISSIONS: Record<UserRole, Record<string, { allowed: boolean; constraint?: string }>> = {
  soc_tier1: {
    splunk_run_query:       { allowed: true, constraint: 'Limited SPL only (no subsearches, no eval with eval expressions)' },
    splunk_get_indexes:     { allowed: true },
    splunk_get_index_detail:{ allowed: true },
    splunk_search_history:  { allowed: true },
    splunk_get_alerts:      { allowed: true },
    splunk_describe:        { allowed: true },
    splunk_ai_assistant:    { allowed: true },
    splunk_get_kv_store:    { allowed: false },
    splunk_list_inputs:     { allowed: false },
    splunk_get_dashboard:   { allowed: true },
    splunk_get_lookup:      { allowed: false },
  },
  soc_tier2: {
    splunk_run_query:       { allowed: true },
    splunk_get_indexes:     { allowed: true },
    splunk_get_index_detail:{ allowed: true },
    splunk_search_history:  { allowed: true },
    splunk_get_alerts:      { allowed: true },
    splunk_describe:        { allowed: true },
    splunk_ai_assistant:    { allowed: true },
    splunk_get_kv_store:    { allowed: true },
    splunk_list_inputs:     { allowed: true },
    splunk_get_dashboard:   { allowed: true },
    splunk_get_lookup:      { allowed: true },
  },
  sre: {
    splunk_run_query:       { allowed: true, constraint: 'Observability and prod indexes only' },
    splunk_get_indexes:     { allowed: true },
    splunk_get_index_detail:{ allowed: true },
    splunk_search_history:  { allowed: true },
    splunk_get_alerts:      { allowed: true, constraint: 'Observability alerts only' },
    splunk_describe:        { allowed: true },
    splunk_ai_assistant:    { allowed: true },
    splunk_get_kv_store:    { allowed: false },
    splunk_list_inputs:     { allowed: true },
    splunk_get_dashboard:   { allowed: true },
    splunk_get_lookup:      { allowed: false },
  },
  contractor: {
    splunk_run_query:       { allowed: false, constraint: 'Contractors cannot execute ad-hoc queries' },
    splunk_get_indexes:     { allowed: true },
    splunk_get_index_detail:{ allowed: false },
    splunk_search_history:  { allowed: false },
    splunk_get_alerts:      { allowed: false },
    splunk_describe:        { allowed: true },
    splunk_ai_assistant:    { allowed: true, constraint: 'Results are redacted' },
    splunk_get_kv_store:    { allowed: false },
    splunk_list_inputs:     { allowed: false },
    splunk_get_dashboard:   { allowed: false },
    splunk_get_lookup:      { allowed: false },
  },
  ai_agent: {
    splunk_run_query:       { allowed: true, constraint: 'Requires human approval for remediation queries' },
    splunk_get_indexes:     { allowed: true },
    splunk_get_index_detail:{ allowed: true },
    splunk_search_history:  { allowed: true },
    splunk_get_alerts:      { allowed: true },
    splunk_describe:        { allowed: true },
    splunk_ai_assistant:    { allowed: true },
    splunk_get_kv_store:    { allowed: true },
    splunk_list_inputs:     { allowed: true },
    splunk_get_dashboard:   { allowed: true },
    splunk_get_lookup:      { allowed: true },
  },
};

// Check tool permission
export function checkToolPermission(role: UserRole, toolName: string, index?: string): AuthZDecision {
  const timestamp = new Date().toISOString();
  const toolPerms = TOOL_PERMISSIONS[role]?.[toolName];

  if (!toolPerms || !toolPerms.allowed) {
    const reason = toolPerms?.constraint || `Role '${role}' is not authorized to execute '${toolName}'`;
    return {
      allowed: false,
      reason,
      policy: 'least_privilege',
      timestamp,
    };
  }

  // If an index is specified, check index-level access
  if (index) {
    const indexPerms = INDEX_PERMISSIONS[role]?.[index];
    if (!indexPerms || indexPerms.length === 0) {
      return {
        allowed: false,
        reason: `Role '${role}' does not have access to index '${index}'. Per SpiceDB policy, index membership is required.`,
        policy: 'index_isolation',
        timestamp,
      };
    }

    if (toolName === 'splunk_run_query' && !indexPerms.includes('query')) {
      return {
        allowed: false,
        reason: `Role '${role}' has read but not query permission on index '${index}'${toolPerms.constraint ? '. ' + toolPerms.constraint : ''}`,
        policy: 'read_only_access',
        timestamp,
      };
    }
  }

  const extras: string[] = [];
  if (toolPerms.constraint) extras.push(toolPerms.constraint);

  return {
    allowed: true,
    reason: extras.length > 0 ? extras.join('. ') : 'Permission granted by role policy',
    policy: toolPerms.constraint ? 'conditional_allow' : 'full_allow',
    timestamp,
  };
}

// Check index permission
export function checkIndexPermission(role: UserRole, index: string, permission: 'read' | 'query'): AuthZDecision {
  const timestamp = new Date().toISOString();
  const perms = INDEX_PERMISSIONS[role]?.[index] || [];

  if (!perms.includes(permission)) {
    return {
      allowed: false,
      reason: `Role '${role}' does not have '${permission}' permission on index '${index}'`,
      policy: 'index_isolation',
      timestamp,
    };
  }

  return {
    allowed: true,
    reason: `Role '${role}' has '${permission}' access to index '${index}'`,
    policy: 'index_isolation',
    timestamp,
  };
}

// Get all tool permissions for a role
export function getRolePermissions(role: UserRole) {
  return {
    tools: TOOL_PERMISSIONS[role] || {},
    indexes: INDEX_PERMISSIONS[role] || {},
  };
}

// Role display info
export const ROLES: Record<UserRole, { label: string; color: string; icon: string; description: string }> = {
  soc_tier1: {
    label: 'SOC Tier 1 Analyst',
    color: 'bg-emerald-500',
    icon: 'Shield',
    description: 'First responder — triage and escalate',
  },
  soc_tier2: {
    label: 'SOC Tier 2 Analyst',
    color: 'bg-amber-500',
    icon: 'ShieldCheck',
    description: 'Senior investigator — full access',
  },
  sre: {
    label: 'SRE Engineer',
    color: 'bg-sky-500',
    icon: 'Server',
    description: 'Site reliability — observability only',
  },
  contractor: {
    label: 'Contractor',
    color: 'bg-violet-500',
    icon: 'UserX',
    description: 'External — heavily restricted',
  },
  ai_agent: {
    label: 'AI Agent',
    color: 'bg-rose-500',
    icon: 'Bot',
    description: 'Automated — needs human approval for actions',
  },
};
