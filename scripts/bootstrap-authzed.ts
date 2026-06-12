#!/usr/bin/env bun
/**
 * Bootstrap AuthZed Cloud with ShieldGate's SpiceDB schema and relationships.
 *
 * Usage:
 *   bun run scripts/bootstrap-authzed.ts
 *
 * Requires AUTHZED_API_KEY and AUTHZED_ENDPOINT in .env.local
 */

import { v1 } from "@authzed/authzed-node";
import { readFileSync } from "fs";
import { resolve } from "path";

// Load .env.local
// `import.meta.dir` is a Bun-specific extension (this script runs under Bun).
const envPath = resolve((import.meta as ImportMeta & { dir: string }).dir, "../.env.local");
try {
  const envContent = readFileSync(envPath, "utf-8");
  for (const line of envContent.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const val = trimmed.slice(eqIdx + 1).trim();
    if (val && !process.env[key]) {
      process.env[key] = val;
    }
  }
} catch {
  console.log("No .env.local found, using environment variables");
}

const token = process.env.AUTHZED_API_KEY;
const endpoint = process.env.AUTHZED_ENDPOINT || "grpc.authzed.com:443";

if (!token) {
  console.error("ERROR: AUTHZED_API_KEY is not set.");
  console.error("Set it in .env.local or export it before running this script.");
  process.exit(1);
}

console.log(`Connecting to AuthZed at ${endpoint}...`);
const client = v1.NewClient(token, endpoint);
const { promises } = client;

// ──────────────────────────────────────
// Step 1: Write the SpiceDB schema
// ──────────────────────────────────────

const SCHEMA = `
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

console.log("\n[1/3] Writing SpiceDB schema...");
try {
  const schemaReq = v1.WriteSchemaRequest.create({ schema: SCHEMA });
  await promises.writeSchema(schemaReq);
  console.log("  Schema written successfully.");
} catch (err: any) {
  console.error("  Schema write failed:", err.message || err);
  process.exit(1);
}

// ──────────────────────────────────────
// Step 2: Write relationships
// ──────────────────────────────────────

type Rel = {
  resourceType: string;
  resourceId: string;
  relation: string;
  subjectType: string;
  subjectId: string;
};

const relationships: Rel[] = [
  // --- Team memberships ---
  // SOC Tier 1 team
  { resourceType: "team", resourceId: "soc_tier1", relation: "member", subjectType: "user", subjectId: "soc_tier1_user" },
  // SOC Tier 2 team
  { resourceType: "team", resourceId: "soc_tier2", relation: "member", subjectType: "user", subjectId: "soc_tier2_user" },
  // SRE team
  { resourceType: "team", resourceId: "sre", relation: "member", subjectType: "user", subjectId: "sre_user" },
  // Contractor team
  { resourceType: "team", resourceId: "contractors", relation: "member", subjectType: "user", subjectId: "contractor_user" },
  // AI Agent team
  { resourceType: "team", resourceId: "ai_agents", relation: "member", subjectType: "user", subjectId: "ai_agent_user" },

  // --- Index permissions ---
  // Security index
  { resourceType: "splunk_index", resourceId: "security", relation: "viewer", subjectType: "team", subjectId: "soc_tier1" },
  { resourceType: "splunk_index", resourceId: "security", relation: "viewer", subjectType: "team", subjectId: "contractors" },
  { resourceType: "splunk_index", resourceId: "security", relation: "querier", subjectType: "team", subjectId: "soc_tier2" },
  { resourceType: "splunk_index", resourceId: "security", relation: "viewer", subjectType: "user", subjectId: "ai_agent_user" },

  // Observability index
  { resourceType: "splunk_index", resourceId: "observability", relation: "viewer", subjectType: "team", subjectId: "soc_tier2" },
  { resourceType: "splunk_index", resourceId: "observability", relation: "querier", subjectType: "team", subjectId: "sre" },
  { resourceType: "splunk_index", resourceId: "observability", relation: "viewer", subjectType: "user", subjectId: "ai_agent_user" },

  // Compliance index
  { resourceType: "splunk_index", resourceId: "compliance", relation: "viewer", subjectType: "team", subjectId: "soc_tier1" },
  { resourceType: "splunk_index", resourceId: "compliance", relation: "querier", subjectType: "team", subjectId: "soc_tier2" },
  { resourceType: "splunk_index", resourceId: "compliance", relation: "viewer", subjectType: "user", subjectId: "ai_agent_user" },

  // Prod index
  { resourceType: "splunk_index", resourceId: "prod", relation: "querier", subjectType: "team", subjectId: "sre" },
  { resourceType: "splunk_index", resourceId: "prod", relation: "viewer", subjectType: "user", subjectId: "ai_agent_user" },

  // --- Tool permissions ---
  // SOC Tier 1 tools
  ...["splunk_run_query", "splunk_get_indexes", "splunk_get_index_detail", "splunk_search_history", "splunk_get_alerts", "splunk_describe", "splunk_ai_assistant", "splunk_get_dashboard"].map(
    (tool): Rel => ({ resourceType: "splunk_tool", resourceId: tool, relation: "allowed_role", subjectType: "team", subjectId: "soc_tier1" })
  ),

  // SOC Tier 2 tools (all tools)
  ...["splunk_run_query", "splunk_get_indexes", "splunk_get_index_detail", "splunk_search_history", "splunk_get_alerts", "splunk_describe", "splunk_ai_assistant", "splunk_get_kv_store", "splunk_list_inputs", "splunk_get_dashboard", "splunk_get_lookup"].map(
    (tool): Rel => ({ resourceType: "splunk_tool", resourceId: tool, relation: "allowed_role", subjectType: "team", subjectId: "soc_tier2" })
  ),

  // SRE tools
  ...["splunk_run_query", "splunk_get_indexes", "splunk_get_index_detail", "splunk_search_history", "splunk_get_alerts", "splunk_describe", "splunk_ai_assistant", "splunk_list_inputs", "splunk_get_dashboard"].map(
    (tool): Rel => ({ resourceType: "splunk_tool", resourceId: tool, relation: "allowed_role", subjectType: "team", subjectId: "sre" })
  ),

  // Contractor tools (very limited)
  ...["splunk_get_indexes", "splunk_describe", "splunk_ai_assistant"].map(
    (tool): Rel => ({ resourceType: "splunk_tool", resourceId: tool, relation: "allowed_role", subjectType: "team", subjectId: "contractors" })
  ),

  // AI Agent tools (all tools, direct user permissions)
  ...["splunk_run_query", "splunk_get_indexes", "splunk_get_index_detail", "splunk_search_history", "splunk_get_alerts", "splunk_describe", "splunk_ai_assistant", "splunk_get_kv_store", "splunk_list_inputs", "splunk_get_dashboard", "splunk_get_lookup"].map(
    (tool): Rel => ({ resourceType: "splunk_tool", resourceId: tool, relation: "allowed_user", subjectType: "user", subjectId: "ai_agent_user" })
  ),
];

console.log(`\n[2/3] Writing ${relationships.length} relationships...`);

const BATCH_SIZE = 50;
for (let i = 0; i < relationships.length; i += BATCH_SIZE) {
  const batch = relationships.slice(i, i + BATCH_SIZE);

  const updates = batch.map((rel) =>
    v1.RelationshipUpdate.create({
      operation: v1.RelationshipUpdate_Operation.TOUCH,
      relationship: v1.Relationship.create({
        resource: v1.ObjectReference.create({
          objectType: rel.resourceType,
          objectId: rel.resourceId,
        }),
        relation: rel.relation,
        subject: v1.SubjectReference.create({
          object: v1.ObjectReference.create({
            objectType: rel.subjectType,
            objectId: rel.subjectId,
          }),
        }),
      }),
    })
  );

  try {
    const req = v1.WriteRelationshipsRequest.create({ updates });
    await promises.writeRelationships(req);
    console.log(`  Batch ${Math.floor(i / BATCH_SIZE) + 1}: wrote ${batch.length} relationships`);
  } catch (err: any) {
    console.error(`  Batch ${Math.floor(i / BATCH_SIZE) + 1} failed:`, err.message || err);
    process.exit(1);
  }
}

// ──────────────────────────────────────
// Step 3: Verify with test permission checks
// ──────────────────────────────────────

console.log("\n[3/3] Verifying permissions...");

const testCases = [
  { resource: "splunk_tool", id: "splunk_run_query", perm: "execute", user: "soc_tier1_user", expect: true },
  { resource: "splunk_tool", id: "splunk_run_query", perm: "execute", user: "contractor_user", expect: false },
  { resource: "splunk_index", id: "security", perm: "read", user: "soc_tier1_user", expect: true },
  { resource: "splunk_index", id: "security", perm: "query", user: "soc_tier1_user", expect: false },
  { resource: "splunk_index", id: "security", perm: "query", user: "soc_tier2_user", expect: true },
  { resource: "splunk_index", id: "observability", perm: "read", user: "sre_user", expect: true },
  { resource: "splunk_index", id: "security", perm: "read", user: "sre_user", expect: false },
];

let passed = 0;
let failed = 0;

for (const tc of testCases) {
  const req = v1.CheckPermissionRequest.create({
    resource: v1.ObjectReference.create({ objectType: tc.resource, objectId: tc.id }),
    permission: tc.perm,
    subject: v1.SubjectReference.create({
      object: v1.ObjectReference.create({ objectType: "user", objectId: tc.user }),
    }),
    consistency: v1.Consistency.create({
      requirement: { oneofKind: "fullyConsistent", fullyConsistent: true },
    }),
  });

  try {
    const resp = await promises.checkPermission(req);
    const allowed = resp.permissionship === v1.CheckPermissionResponse_Permissionship.HAS_PERMISSION;
    const ok = allowed === tc.expect;

    if (ok) {
      passed++;
      console.log(`  PASS: ${tc.user} ${tc.perm} ${tc.resource}/${tc.id} = ${allowed}`);
    } else {
      failed++;
      console.log(`  FAIL: ${tc.user} ${tc.perm} ${tc.resource}/${tc.id} expected=${tc.expect} got=${allowed}`);
    }
  } catch (err: any) {
    failed++;
    console.log(`  ERROR: ${tc.user} ${tc.perm} ${tc.resource}/${tc.id}: ${err.message}`);
  }
}

console.log(`\nVerification: ${passed} passed, ${failed} failed out of ${testCases.length} tests`);

if (failed > 0) {
  console.error("\nSome permission checks failed. Review your schema and relationships.");
  process.exit(1);
}

console.log("\nAuthZed Cloud bootstrap complete! ShieldGate is ready.");
console.log("Start the dev server with: bun run dev");
