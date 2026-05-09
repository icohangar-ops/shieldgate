'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Activity, AlertTriangle, CheckCircle2, Clock, DollarSign,
  GitBranch, Globe, Layers, Lock, Milestone, Radar, Shield,
  Target, TrendingUp, Users, Workflow, ChevronRight, Zap,
  ArrowRight, BarChart3, FileText, Gavel
} from 'lucide-react'

// ─── Types ───────────────────────────────────────────────────────────────────

interface WorkstreamData {
  id: string
  name: string
  icon: React.ReactNode
  health: 'green' | 'amber' | 'red'
  completionPct: number
  agents: { name: string; role: string; avatar: string }[]
  recentActivity: string[]
  keyMetrics: { label: string; value: string }[]
}

interface RiskItem {
  id: string
  title: string
  category: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'open' | 'mitigating' | 'resolved'
  owner: string
  impact: string
  mitigation: string
}

interface DecisionCase {
  id: string
  title: string
  domain: string
  status: 'EXPLORING' | 'PROVISIONAL_LOCK' | 'LOCKED' | 'HALT' | 'REFRAME_REQUIRED'
  foundationScore: number
  round: number
  lockedCount: number
  owner: string
}

interface SynergyItem {
  id: string
  category: 'cost' | 'revenue' | 'capital'
  description: string
  annualValue: number
  status: 'identified' | 'in_progress' | 'captured' | 'at_risk'
  probability: number
  owner: string
}

// ─── Mock Data ───────────────────────────────────────────────────────────────

const WORKSTREAMS: WorkstreamData[] = [
  {
    id: 'coa_mapping',
    name: 'Chart of Accounts Mapping',
    icon: <Layers className="h-5 w-5" />,
    health: 'green',
    completionPct: 82,
    agents: [
      { name: 'Finance Agent', role: 'Account mapping, KPI alignment', avatar: 'F' },
      { name: 'Strategy Agent', role: 'Management view alignment', avatar: 'S' },
      { name: 'Compliance Agent', role: 'Restatement estimates', avatar: 'C' },
    ],
    recentActivity: [
      'Revenue accounts mapped (412 -> 847 lines)',
      'Intercompany elimination rules defined',
      'KPI alignment score: 91%',
    ],
    keyMetrics: [
      { label: 'Accounts Mapped', value: '2,847 / 3,412' },
      { label: 'Mapping Accuracy', value: '97.3%' },
      { label: 'KPI Alignment', value: '91%' },
    ],
  },
  {
    id: 'close_harmonization',
    name: 'Close Harmonization',
    icon: <Clock className="h-5 w-5" />,
    health: 'amber',
    completionPct: 58,
    agents: [
      { name: 'Finance Agent', role: 'Close calendar, reconciliation', avatar: 'F' },
      { name: 'Strategy Agent', role: 'Policy alignment', avatar: 'S' },
      { name: 'Compliance Agent', role: 'Audit readiness', avatar: 'C' },
    ],
    recentActivity: [
      'Close calendar draft circulated',
      '3 reconciliation gaps identified in AP',
      'Audit trail requirements documented',
    ],
    keyMetrics: [
      { label: 'Close Items', value: '34 / 59' },
      { label: 'Recon Gaps', value: '3 open' },
      { label: 'Days to Close', value: 'T+12 target' },
    ],
  },
  {
    id: 'systems_integration',
    name: 'Systems Integration',
    icon: <Globe className="h-5 w-5" />,
    health: 'amber',
    completionPct: 41,
    agents: [
      { name: 'Finance Agent', role: 'System dependency map', avatar: 'F' },
      { name: 'Strategy Agent', role: 'Migration plan', avatar: 'S' },
      { name: 'Compliance Agent', role: 'Security review', avatar: 'C' },
    ],
    recentActivity: [
      'System dependency map v2 published',
      'ERP cutover risk assessment complete',
      'Sandbox environment provisioned',
    ],
    keyMetrics: [
      { label: 'Systems Mapped', value: '18 / 31' },
      { label: 'Cutover Risks', value: '5 identified' },
      { label: 'Migration Plan', value: 'v2 draft' },
    ],
  },
  {
    id: 'synergy_tracking',
    name: 'Synergy Tracking',
    icon: <DollarSign className="h-5 w-5" />,
    health: 'green',
    completionPct: 67,
    agents: [
      { name: 'Finance Agent', role: 'Synergy pipeline, value capture', avatar: 'F' },
      { name: 'Strategy Agent', role: 'Risk register', avatar: 'S' },
      { name: 'Compliance Agent', role: 'Reporting controls', avatar: 'C' },
    ],
    recentActivity: [
      'Q2 synergy tracker updated: $14.2M YTD',
      '2 cost synergies at risk flagged',
      'Board deck v3 circulated',
    ],
    keyMetrics: [
      { label: 'Synergies YTD', value: '$14.2M' },
      { label: 'Annual Target', value: '$28M' },
      { label: 'At-Risk Items', value: '2' },
    ],
  },
]

const RISKS: RiskItem[] = [
  { id: 'R-001', title: 'ERP cutover data loss risk', category: 'Systems', severity: 'critical', status: 'mitigating', owner: 'CTO Office', impact: 'Potential 48-hour close delay', mitigation: 'Parallel run with reconciliation checkpoints' },
  { id: 'R-002', title: 'Intercompany elimination complexity', category: 'Accounting', severity: 'high', status: 'open', owner: 'Controller', impact: 'Material misstatement in consolidated filings', mitigation: 'CHP-governed mapping validation via 3-agent consensus' },
  { id: 'R-003', title: 'Tax jurisdiction overlap', category: 'Tax', severity: 'high', status: 'mitigating', owner: 'Tax Director', impact: 'Double taxation on intercompany transactions', mitigation: 'Transfer pricing documentation underway' },
  { id: 'R-004', title: 'Key personnel attrition', category: 'People', severity: 'medium', status: 'open', owner: 'HR', impact: 'Knowledge loss in target close processes', mitigation: 'Retention packages + documentation sprint' },
  { id: 'R-005', title: 'Vendor contract renegotiation', category: 'Operations', severity: 'low', status: 'resolved', owner: 'Procurement', impact: 'Minimal — resolved via early renewal', mitigation: 'Completed Q1' },
]

const DECISIONS: DecisionCase[] = [
  { id: 'DC-001', title: 'Unified revenue recognition policy', domain: 'coa_mapping', status: 'LOCKED', foundationScore: 94, round: 4, lockedCount: 3, owner: 'CFO' },
  { id: 'DC-002', title: 'Combined close calendar (T+12)', domain: 'close_harmonization', status: 'PROVISIONAL_LOCK', foundationScore: 82, round: 3, lockedCount: 2, owner: 'Controller' },
  { id: 'DC-003', title: 'ERP migration: big-bang vs phased', domain: 'systems_integration', status: 'EXPLORING', foundationScore: 71, round: 1, lockedCount: 0, owner: 'CTO' },
  { id: 'DC-004', title: 'Headcount rationalization plan', domain: 'synergy_tracking', status: 'LOCKED', foundationScore: 88, round: 3, lockedCount: 3, owner: 'CHRO' },
  { id: 'DC-005', title: 'Transfer pricing methodology', domain: 'coa_mapping', status: 'EXPLORING', foundationScore: 65, round: 1, lockedCount: 0, owner: 'Tax Director' },
  { id: 'DC-006', title: 'Cloud infrastructure consolidation', domain: 'systems_integration', status: 'HALT', foundationScore: 42, round: 0, lockedCount: 0, owner: 'CTO' },
  { id: 'DC-007', title: 'Customer contract migration', domain: 'close_harmonization', status: 'REFRAME_REQUIRED', foundationScore: 55, round: 2, lockedCount: 0, owner: 'CRO' },
]

const SYNERGIES: SynergyItem[] = [
  { id: 'SYN-001', category: 'cost', description: 'Shared services consolidation', annualValue: 4200000, status: 'captured', probability: 0.95, owner: 'COO' },
  { id: 'SYN-002', category: 'cost', description: 'Facility rationalization', annualValue: 3100000, status: 'in_progress', probability: 0.80, owner: 'COO' },
  { id: 'SYN-003', category: 'cost', description: 'IT infrastructure redundancy removal', annualValue: 2800000, status: 'in_progress', probability: 0.85, owner: 'CTO' },
  { id: 'SYN-004', category: 'revenue', description: 'Cross-sell to combined customer base', annualValue: 5600000, status: 'in_progress', probability: 0.60, owner: 'CRO' },
  { id: 'SYN-005', category: 'revenue', description: 'Pricing optimization (combined volumes)', annualValue: 3800000, status: 'identified', probability: 0.50, owner: 'CRO' },
  { id: 'SYN-006', category: 'cost', description: 'Procurement leverage (combined vendor base)', annualValue: 2400000, status: 'captured', probability: 0.90, owner: 'CPO' },
  { id: 'SYN-007', category: 'revenue', description: 'Geographic expansion via target channels', annualValue: 2100000, status: 'at_risk', probability: 0.35, owner: 'CRO' },
  { id: 'SYN-008', category: 'capital', description: 'Working capital optimization', annualValue: 1500000, status: 'in_progress', probability: 0.70, owner: 'CFO' },
]

const MILESTONES = [
  { id: 'M-001', title: 'First combined monthly close', date: '2026-10-31', status: 'pending' as const, workstream: 'close_harmonization' },
  { id: 'M-002', title: 'CoA mapping 100% complete', date: '2026-09-15', status: 'in_progress' as const, workstream: 'coa_mapping' },
  { id: 'M-003', title: 'ERP sandbox validated', date: '2026-08-01', status: 'complete' as const, workstream: 'systems_integration' },
  { id: 'M-004', title: 'Synergy tracker operational', date: '2026-07-15', status: 'complete' as const, workstream: 'synergy_tracking' },
  { id: 'M-005', title: 'Board integration report #1', date: '2026-08-30', status: 'complete' as const, workstream: 'synergy_tracking' },
  { id: 'M-006', title: 'Tax filing under combined entity', date: '2027-04-15', status: 'pending' as const, workstream: 'coa_mapping' },
  { id: 'M-007', title: 'Full ERP cutover', date: '2027-01-01', status: 'pending' as const, workstream: 'systems_integration' },
]

// ─── Helper Components ───────────────────────────────────────────────────────

function HealthBadge({ health }: { health: 'green' | 'amber' | 'red' }) {
  const config = {
    green: { bg: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300', dot: 'bg-emerald-500', label: 'GREEN' },
    amber: { bg: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300', dot: 'bg-amber-500', label: 'AMBER' },
    red: { bg: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300', dot: 'bg-red-500', label: 'RED' },
  }
  const c = config[health]
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${c.bg}`}>
      <span className={`h-2 w-2 rounded-full ${c.dot} animate-pulse`} />
      {c.label}
    </span>
  )
}

function SeverityBadge({ severity }: { severity: string }) {
  const config: Record<string, string> = {
    critical: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
    high: 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300',
    low: 'bg-slate-100 text-slate-600 dark:bg-slate-800/40 dark:text-slate-400',
  }
  return <Badge variant="outline" className={`${config[severity] || ''} border-0 font-semibold text-xs`}>{severity.toUpperCase()}</Badge>
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, string> = {
    open: 'bg-slate-100 text-slate-700 dark:bg-slate-800/40 dark:text-slate-300',
    mitigating: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
    resolved: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  }
  return <Badge variant="outline" className={`${config[status] || ''} border-0 font-semibold text-xs`}>{status.replace('_', ' ').toUpperCase()}</Badge>
}

function CHPStatusBadge({ status }: { status: string }) {
  const config: Record<string, string> = {
    EXPLORING: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
    PROVISIONAL_LOCK: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
    LOCKED: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    HALT: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
    REFRAME_REQUIRED: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300',
  }
  const icons: Record<string, React.ReactNode> = {
    EXPLORING: <Radar className="h-3 w-3" />,
    PROVISIONAL_LOCK: <Lock className="h-3 w-3" />,
    LOCKED: <Lock className="h-3 w-3" />,
    HALT: <AlertTriangle className="h-3 w-3" />,
    REFRAME_REQUIRED: <GitBranch className="h-3 w-3" />,
  }
  return (
    <Badge variant="outline" className={`${config[status] || ''} border-0 font-semibold text-xs gap-1`}>
      {icons[status]}
      {status.replace('_', ' ')}
    </Badge>
  )
}

function MilestoneStatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'complete': return <CheckCircle2 className="h-4 w-4 text-emerald-500" />
    case 'in_progress': return <Clock className="h-4 w-4 text-amber-500 animate-pulse" />
    default: return <Milestone className="h-4 w-4 text-slate-400" />
  }
}

// ─── Main Dashboard ──────────────────────────────────────────────────────────

export default function Home() {
  const [activeTab, setActiveTab] = useState('overview')
  const [currentTime, setCurrentTime] = useState(new Date())
  const overallHealth: 'green' | 'amber' | 'red' =
    WORKSTREAMS.some(w => w.health === 'red') ? 'red' :
    WORKSTREAMS.some(w => w.health === 'amber') ? 'amber' : 'green'
  const avgCompletion = Math.round(WORKSTREAMS.reduce((s, w) => s + w.completionPct, 0) / WORKSTREAMS.length)
  const totalSynergies = SYNERGIES.reduce((s, sy) => s + sy.annualValue * sy.probability, 0)
  const lockedDecisions = DECISIONS.filter(d => d.status === 'LOCKED').length
  const openCriticalRisks = RISKS.filter(r => r.severity === 'critical' && r.status !== 'resolved').length

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-200/60 dark:border-slate-800/60 bg-white/80 dark:bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-slate-800 to-slate-600 dark:from-slate-200 dark:to-slate-400 flex items-center justify-center shadow-lg">
              <Zap className="h-5 w-5 text-white dark:text-slate-900" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">Convergence</h1>
              <p className="text-[10px] font-medium text-slate-500 dark:text-slate-400 tracking-wider uppercase">Post-Merger Integration</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
              <Clock className="h-4 w-4" />
              <span>{currentTime.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}</span>
            </div>
            <HealthBadge health={overallHealth} />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full max-w-3xl mx-auto grid-cols-5 bg-slate-100 dark:bg-slate-800/50 p-1 rounded-xl h-auto">
            <TabsTrigger value="overview" className="text-xs sm:text-sm data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700 rounded-lg shadow-sm gap-1.5 py-2">
              <BarChart3 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Overview</span>
            </TabsTrigger>
            <TabsTrigger value="workstreams" className="text-xs sm:text-sm data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700 rounded-lg shadow-sm gap-1.5 py-2">
              <Workflow className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Workstreams</span>
            </TabsTrigger>
            <TabsTrigger value="decisions" className="text-xs sm:text-sm data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700 rounded-lg shadow-sm gap-1.5 py-2">
              <Gavel className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">CHP</span>
            </TabsTrigger>
            <TabsTrigger value="risks" className="text-xs sm:text-sm data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700 rounded-lg shadow-sm gap-1.5 py-2">
              <Shield className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Risks</span>
            </TabsTrigger>
            <TabsTrigger value="synergies" className="text-xs sm:text-sm data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700 rounded-lg shadow-sm gap-1.5 py-2">
              <TrendingUp className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Synergies</span>
            </TabsTrigger>
          </TabsList>

          {/* ─── OVERVIEW TAB ─── */}
          <TabsContent value="overview" className="space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Overall Health</span>
                    <Activity className="h-4 w-4 text-slate-400" />
                  </div>
                  <div className="flex items-center gap-2">
                    <HealthBadge health={overallHealth} />
                  </div>
                  <p className="text-xs text-slate-400 mt-2">Based on 4 workstreams</p>
                </CardContent>
              </Card>

              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Completion</span>
                    <Target className="h-4 w-4 text-slate-400" />
                  </div>
                  <div className="text-3xl font-bold text-slate-900 dark:text-white">{avgCompletion}%</div>
                  <Progress value={avgCompletion} className="mt-2 h-1.5" />
                  <p className="text-xs text-slate-400 mt-2">Avg across all workstreams</p>
                </CardContent>
              </Card>

              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Decisions Locked</span>
                    <Lock className="h-4 w-4 text-slate-400" />
                  </div>
                  <div className="text-3xl font-bold text-slate-900 dark:text-white">{lockedDecisions}<span className="text-lg text-slate-400 font-normal">/{DECISIONS.length}</span></div>
                  <Progress value={(lockedDecisions / DECISIONS.length) * 100} className="mt-2 h-1.5" />
                  <p className="text-xs text-slate-400 mt-2">CHP-validated and locked</p>
                </CardContent>
              </Card>

              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Synergies (EWE)</span>
                    <DollarSign className="h-4 w-4 text-slate-400" />
                  </div>
                  <div className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">${(totalSynergies / 1000000).toFixed(1)}M</div>
                  <Progress value={(totalSynergies / 28000000) * 100} className="mt-2 h-1.5" />
                  <p className="text-xs text-slate-400 mt-2">Expected weighted value vs $28M target</p>
                </CardContent>
              </Card>
            </div>

            {/* Workstream Summary + Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Workstream Health */}
              <Card className="lg:col-span-2 border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base font-semibold">Workstream Health</CardTitle>
                    <Badge variant="outline" className="text-xs">{openCriticalRisks} critical risks</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {WORKSTREAMS.map(ws => (
                    <div key={ws.id} className="flex items-center gap-4 p-3 rounded-xl bg-slate-50/80 dark:bg-slate-800/30 hover:bg-slate-100 dark:hover:bg-slate-800/50 transition-colors cursor-pointer" onClick={() => setActiveTab('workstreams')}>
                      <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-white dark:bg-slate-700 flex items-center justify-center text-slate-600 dark:text-slate-300 shadow-sm">
                        {ws.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium truncate">{ws.name}</span>
                          <HealthBadge health={ws.health} />
                        </div>
                        <Progress value={ws.completionPct} className="h-1.5" />
                      </div>
                      <span className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex-shrink-0">{ws.completionPct}%</span>
                      <ChevronRight className="h-4 w-4 text-slate-400 flex-shrink-0" />
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Milestones */}
              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base font-semibold">Milestones</CardTitle>
                  <CardDescription className="text-xs">Integration timeline</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[320px] pr-3">
                    <div className="space-y-4">
                      {MILESTONES.map((ms, i) => (
                        <div key={ms.id} className="flex gap-3">
                          <div className="flex flex-col items-center">
                            <MilestoneStatusIcon status={ms.status} />
                            {i < MILESTONES.length - 1 && <div className="w-px h-full min-h-[20px] bg-slate-200 dark:bg-slate-700 mt-1" />}
                          </div>
                          <div className="pb-3">
                            <p className={`text-sm font-medium ${ms.status === 'complete' ? 'line-through text-slate-400 dark:text-slate-500' : 'text-slate-800 dark:text-slate-200'}`}>{ms.title}</p>
                            <div className="flex items-center gap-2 mt-0.5">
                              <span className="text-xs text-slate-400">{ms.date}</span>
                              <Badge variant="outline" className="text-[10px] h-4 border-0 bg-slate-100 dark:bg-slate-800 text-slate-500">{ms.workstream.replace('_', ' ')}</Badge>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            {/* CHP Pipeline Quick View */}
            <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base font-semibold">CHP Decision Pipeline</CardTitle>
                    <CardDescription className="text-xs mt-0.5">Consensus Hardening Protocol — adversarial multi-agent validation</CardDescription>
                  </div>
                  <button onClick={() => setActiveTab('decisions')} className="text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 flex items-center gap-1 transition-colors">
                    View all <ArrowRight className="h-3 w-3" />
                  </button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {(['EXPLORING', 'PROVISIONAL_LOCK', 'LOCKED', 'HALT'] as const).map(status => {
                    const count = DECISIONS.filter(d => d.status === status).length
                    const color = { EXPLORING: 'blue', PROVISIONAL_LOCK: 'amber', LOCKED: 'emerald', HALT: 'red' }[status]
                    return (
                      <div key={status} className="rounded-xl bg-slate-50/80 dark:bg-slate-800/30 p-4 text-center">
                        <div className={`text-2xl font-bold text-${color}-600 dark:text-${color}-400`}>{count}</div>
                        <div className="text-xs text-slate-500 mt-1 font-medium">{status.replace('_', ' ')}</div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── WORKSTREAMS TAB ─── */}
          <TabsContent value="workstreams" className="space-y-6">
            {WORKSTREAMS.map(ws => (
              <Card key={ws.id} className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="h-10 w-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-600 dark:text-slate-300">
                        {ws.icon}
                      </div>
                      <div>
                        <CardTitle className="text-base font-semibold">{ws.name}</CardTitle>
                        <CardDescription className="text-xs">
                          <span className="mr-2">Progress: {ws.completionPct}%</span>
                          <HealthBadge health={ws.health} />
                        </CardDescription>
                      </div>
                    </div>
                  </div>
                  <Progress value={ws.completionPct} className="mt-2 h-2" />
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Agents */}
                    <div>
                      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Multi-Agent Team</h4>
                      <div className="space-y-2">
                        {ws.agents.map(agent => (
                          <div key={agent.name} className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-800/30">
                            <div className="h-7 w-7 rounded-full bg-gradient-to-br from-slate-600 to-slate-800 dark:from-slate-300 dark:to-slate-500 flex items-center justify-center text-white dark:text-slate-900 text-xs font-bold">
                              {agent.avatar}
                            </div>
                            <div>
                              <p className="text-xs font-medium">{agent.name}</p>
                              <p className="text-[10px] text-slate-400">{agent.role}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Key Metrics */}
                    <div>
                      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Key Metrics</h4>
                      <div className="space-y-3">
                        {ws.keyMetrics.map(m => (
                          <div key={m.label} className="flex justify-between items-center p-2 rounded-lg bg-slate-50 dark:bg-slate-800/30">
                            <span className="text-xs text-slate-500">{m.label}</span>
                            <span className="text-xs font-semibold">{m.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Recent Activity */}
                    <div>
                      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Recent Activity</h4>
                      <div className="space-y-2">
                        {ws.recentActivity.map((act, i) => (
                          <div key={i} className="flex items-start gap-2">
                            <ChevronRight className="h-3 w-3 text-slate-400 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-slate-600 dark:text-slate-400">{act}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          {/* ─── CHP DECISIONS TAB ─── */}
          <TabsContent value="decisions" className="space-y-6">
            <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
              <CardHeader className="pb-3">
                <div>
                  <CardTitle className="text-base font-semibold">Consensus Hardening Protocol</CardTitle>
                  <CardDescription className="text-xs mt-1">
                    Every integration decision passes through adversarial multi-agent validation. R0 Gate → Foundation → Attack → Lock.
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                {/* CHP Flow Diagram */}
                <div className="flex flex-wrap items-center gap-2 mb-6 p-4 rounded-xl bg-slate-50 dark:bg-slate-800/30">
                  <Badge variant="outline" className="gap-1 bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800">
                    <FileText className="h-3 w-3" /> DecisionCase
                  </Badge>
                  <ArrowRight className="h-4 w-4 text-slate-400" />
                  <Badge variant="outline" className="gap-1 bg-slate-50 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700">
                    R0 Gate
                  </Badge>
                  <ArrowRight className="h-4 w-4 text-slate-400" />
                  <Badge variant="outline" className="gap-1 bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/30 dark:text-purple-300 dark:border-purple-800">
                    Foundation Disclosure
                  </Badge>
                  <ArrowRight className="h-4 w-4 text-slate-400" />
                  <Badge variant="outline" className="gap-1 bg-red-50 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800">
                    <AlertTriangle className="h-3 w-3" /> Foundation Attack
                  </Badge>
                  <ArrowRight className="h-4 w-4 text-slate-400" />
                  <Badge variant="outline" className="gap-1 bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-800">
                    {'Score >= 70'}
                  </Badge>
                  <ArrowRight className="h-4 w-4 text-slate-400" />
                  <Badge variant="outline" className="gap-1 bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-800">
                    <Lock className="h-3 w-3" /> LOCKED
                  </Badge>
                </div>

                {/* Decision Table */}
                <ScrollArea className="h-[500px]">
                  <div className="space-y-3">
                    {DECISIONS.map(decision => (
                      <div key={decision.id} className="p-4 rounded-xl border border-slate-100 dark:border-slate-800 hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition-colors">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono text-slate-400">{decision.id}</span>
                            <span className="text-sm font-semibold">{decision.title}</span>
                          </div>
                          <CHPStatusBadge status={decision.status} />
                        </div>
                        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                          <span className="flex items-center gap-1"><Users className="h-3 w-3" /> {decision.owner}</span>
                          <span className="flex items-center gap-1"><Layers className="h-3 w-3" /> {decision.domain.replace('_', ' ')}</span>
                          <span className="flex items-center gap-1"><Target className="h-3 w-3" /> Foundation: {decision.foundationScore}</span>
                          <span className="flex items-center gap-1"><GitBranch className="h-3 w-3" /> Round {decision.round}</span>
                          {decision.lockedCount > 0 && (
                            <span className="flex items-center gap-1"><Lock className="h-3 w-3" /> {decision.lockedCount} locked</span>
                          )}
                        </div>
                        {decision.status !== 'LOCKED' && decision.status !== 'HALT' && (
                          <div className="mt-2">
                            <Progress
                              value={decision.status === 'PROVISIONAL_LOCK' ? 85 : decision.foundationScore}
                              className={`h-1 ${decision.foundationScore >= 70 ? '[&>div]:bg-emerald-500' : '[&>div]:bg-amber-500'}`}
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── RISKS TAB ─── */}
          <TabsContent value="risks" className="space-y-6">
            {/* Risk Summary */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {(['critical', 'high', 'medium', 'low'] as const).map(sev => {
                const count = RISKS.filter(r => r.severity === sev && r.status !== 'resolved').length
                const colors = { critical: 'red', high: 'orange', medium: 'yellow', low: 'slate' }
                return (
                  <Card key={sev} className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                    <CardContent className="p-5 text-center">
                      <div className={`text-3xl font-bold text-${colors[sev]}-600 dark:text-${colors[sev]}-400`}>{count}</div>
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mt-1">{sev}</div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Risk Registry */}
            <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-base font-semibold">Risk Registry</CardTitle>
                <CardDescription className="text-xs">Integration risk items with severity and mitigation tracking</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[450px]">
                  <div className="space-y-3">
                    {RISKS.map(risk => (
                      <div key={risk.id} className={`p-4 rounded-xl border transition-colors ${risk.status === 'resolved' ? 'border-emerald-200 bg-emerald-50/50 dark:border-emerald-800/50 dark:bg-emerald-900/10' : risk.severity === 'critical' ? 'border-red-200 bg-red-50/50 dark:border-red-800/50 dark:bg-red-900/10' : 'border-slate-100 dark:border-slate-800 hover:bg-slate-50/50 dark:hover:bg-slate-800/30'}`}>
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono text-slate-400">{risk.id}</span>
                            <span className="text-sm font-semibold">{risk.title}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <SeverityBadge severity={risk.severity} />
                            <StatusBadge status={risk.status} />
                          </div>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs text-slate-500 mb-2">
                          <div><span className="font-medium text-slate-600 dark:text-slate-400">Owner:</span> {risk.owner}</div>
                          <div><span className="font-medium text-slate-600 dark:text-slate-400">Category:</span> {risk.category}</div>
                          <div><span className="font-medium text-slate-600 dark:text-slate-400">Impact:</span> {risk.impact}</div>
                        </div>
                        <div className="text-xs text-slate-500 bg-white dark:bg-slate-900/50 p-2 rounded-lg">
                          <span className="font-medium text-slate-600 dark:text-slate-400">Mitigation:</span> {risk.mitigation}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ─── SYNERGIES TAB ─── */}
          <TabsContent value="synergies" className="space-y-6">
            {/* Synergy Summary */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Total Synergy Pipeline</div>
                  <div className="text-3xl font-bold text-slate-900 dark:text-white">
                    ${(SYNERGIES.reduce((s, sy) => s + sy.annualValue, 0) / 1000000).toFixed(1)}M
                  </div>
                  <p className="text-xs text-slate-400 mt-1">Annual run-rate target</p>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Expected Value (Weighted)</div>
                  <div className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">${(totalSynergies / 1000000).toFixed(1)}M</div>
                  <p className="text-xs text-slate-400 mt-1">Probability-adjusted</p>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
                <CardContent className="p-5">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Captured to Date</div>
                  <div className="text-3xl font-bold text-slate-900 dark:text-white">
                    ${(SYNERGIES.filter(s => s.status === 'captured').reduce((sum, s) => sum + s.annualValue, 0) / 1000000).toFixed(1)}M
                  </div>
                  <p className="text-xs text-slate-400 mt-1">Realized synergies</p>
                </CardContent>
              </Card>
            </div>

            {/* Synergy Pipeline */}
            <Card className="border-0 shadow-lg shadow-slate-200/30 dark:shadow-slate-900/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-base font-semibold">Synergy Pipeline</CardTitle>
                <CardDescription className="text-xs">Cost, revenue, and capital synergy tracking across the integration</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[450px]">
                  <div className="space-y-3">
                    {SYNERGIES.map(synergy => (
                      <div key={synergy.id} className="p-4 rounded-xl border border-slate-100 dark:border-slate-800 hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition-colors">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className={`border-0 text-[10px] font-semibold ${synergy.category === 'cost' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300' : synergy.category === 'revenue' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300' : 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300'}`}>
                              {synergy.category.toUpperCase()}
                            </Badge>
                            <span className="text-sm font-semibold">{synergy.description}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-slate-900 dark:text-white">${(synergy.annualValue / 1000000).toFixed(1)}M</span>
                            <Badge variant="outline" className={`border-0 text-xs font-semibold ${synergy.status === 'captured' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300' : synergy.status === 'in_progress' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300' : synergy.status === 'at_risk' ? 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-800/40 dark:text-slate-400'}`}>
                              {synergy.status.replace('_', ' ').toUpperCase()}
                            </Badge>
                          </div>
                        </div>
                        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                          <span><span className="font-medium">Owner:</span> {synergy.owner}</span>
                          <span><span className="font-medium">Probability:</span> {(synergy.probability * 100).toFixed(0)}%</span>
                          <span><span className="font-medium">Weighted Value:</span> ${((synergy.annualValue * synergy.probability) / 1000000).toFixed(1)}M</span>
                        </div>
                        <div className="mt-2">
                          <Progress value={synergy.probability * 100} className="h-1.5" />
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200/60 dark:border-slate-800/60 mt-8">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between text-xs text-slate-400">
          <span>Convergence v1.0 — Post-Merger Integration Intelligence Platform</span>
          <span>CHP-governed multi-agent system on DigitalOcean</span>
        </div>
      </footer>
    </div>
  )
}
