export type UserRole =
  | "soc_tier1"
  | "soc_tier2"
  | "sre"
  | "contractor"
  | "ai_agent";

export interface AuthZDecision {
  allowed: boolean;
  reason: string;
  policy: string;
  timestamp: string;
  source?: "spicedb" | "simulation";
}

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

export const ROLES: Record<
  UserRole,
  { label: string; color: string; icon: string; description: string }
> = {
  soc_tier1: {
    label: "SOC Tier 1 Analyst",
    color: "bg-emerald-500",
    icon: "Shield",
    description: "First responder — triage and escalate",
  },
  soc_tier2: {
    label: "SOC Tier 2 Analyst",
    color: "bg-amber-500",
    icon: "ShieldCheck",
    description: "Senior investigator — full access",
  },
  sre: {
    label: "SRE Engineer",
    color: "bg-sky-500",
    icon: "Server",
    description: "Site reliability — observability only",
  },
  contractor: {
    label: "Contractor",
    color: "bg-violet-500",
    icon: "UserX",
    description: "External — heavily restricted",
  },
  ai_agent: {
    label: "AI Agent",
    color: "bg-rose-500",
    icon: "Bot",
    description: "Automated — needs human approval for actions",
  },
};
