'use client';

import { useEffect, useCallback, useRef, useState } from 'react';
import { useStore } from '@/lib/store';
import type { UserRole } from '@/lib/authz-types';
import { ROLES } from '@/lib/authz-types';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Shield,
  ShieldCheck,
  Server,
  UserX,
  Bot,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Search,
  Send,
  Terminal,
  Lock,
  Unlock,
  Eye,
  Activity,
  FileWarning,
  Zap,
  Loader2,
  AlertCircle,
  Database,
} from 'lucide-react';

// ─── Severity Color Map ───
const severityConfig: Record<string, { bg: string; text: string; border: string }> = {
  critical: { bg: 'bg-red-500/10', text: 'text-red-600', border: 'border-red-500/30' },
  high: { bg: 'bg-amber-500/10', text: 'text-amber-600', border: 'border-amber-500/30' },
  medium: { bg: 'bg-yellow-500/10', text: 'text-yellow-600', border: 'border-yellow-500/30' },
  low: { bg: 'bg-emerald-500/10', text: 'text-emerald-600', border: 'border-emerald-500/30' },
};

const roleIcons: Record<UserRole, React.ReactNode> = {
  soc_tier1: <Shield className="h-4 w-4" />,
  soc_tier2: <ShieldCheck className="h-4 w-4" />,
  sre: <Server className="h-4 w-4" />,
  contractor: <UserX className="h-4 w-4" />,
  ai_agent: <Bot className="h-4 w-4" />,
};

// ─── Main Page Component ───
export default function ShieldGatePage() {
  const store = useStore();
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [splQuery, setSplQuery] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [stats, setStats] = useState({ totalIncidents: 0, openIncidents: 0, critical: 0, auditLogs: 0 });
  const [permStats, setPermStats] = useState({ allowed: 0, denied: 0 });

  // Fetch initial data
  useEffect(() => {
    fetchIncidents();
    fetchAuditLogs();
  }, []);

  // Update stats when data changes
  useEffect(() => {
    setStats({
      totalIncidents: store.incidents.length,
      openIncidents: store.incidents.filter(i => i.status === 'open').length,
      critical: store.incidents.filter(i => i.severity === 'critical').length,
      auditLogs: store.auditLogs.length,
    });
    const allowed = store.auditLogs.filter(l => l.decision === 'ALLOW').length;
    const denied = store.auditLogs.filter(l => l.decision === 'DENY').length;
    setPermStats({ allowed, denied });
  }, [store.incidents, store.auditLogs]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [store.chatMessages]);

  const fetchIncidents = async () => {
    try {
      const res = await fetch('/api/incidents');
      const data = await res.json();
      store.setIncidents(data);
    } catch (e) {
      console.error('Failed to fetch incidents', e);
    }
  };

  const fetchAuditLogs = async () => {
    try {
      const res = await fetch('/api/audit?limit=100');
      const data = await res.json();
      store.setAuditLogs(data);
    } catch (e) {
      console.error('Failed to fetch audit logs', e);
    }
  };

  const runSplunkQuery = async () => {
    if (!splQuery.trim()) return;
    store.setIsLoading(true);
    store.setActivePanel('query');

    try {
      const incidentIndex = store.selectedIncident?.sourceIndex || undefined;
      const res = await fetch('/api/splunk/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spl: splQuery,
          index: incidentIndex,
          role: store.currentRole,
        }),
      });
      const data = await res.json();
      store.setQueryResults(data);

      // Refresh audit logs to show the new decision
      fetchAuditLogs();
    } catch (e) {
      store.setQueryResults({ authorized: false, error: 'Query execution failed' });
    } finally {
      store.setIsLoading(false);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || store.isChatLoading) return;

    const userMsg = { role: 'user' as const, content: chatInput, timestamp: new Date().toISOString() };
    store.addChatMessage(userMsg);
    setChatInput('');
    store.setIsChatLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...store.chatMessages, userMsg],
          incidentContext: store.selectedIncident
            ? `Title: ${store.selectedIncident.title}\nSeverity: ${store.selectedIncident.severity}\nIndex: ${store.selectedIncident.sourceIndex}\nDescription: ${store.selectedIncident.description}`
            : undefined,
        }),
      });
      const data = await res.json();
      store.addChatMessage({ role: 'assistant', content: data.content, timestamp: new Date().toISOString() });
    } catch (e) {
      store.addChatMessage({
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      });
    } finally {
      store.setIsChatLoading(false);
    }
  };

  const handleRoleChange = useCallback((role: UserRole) => {
    store.setCurrentRole(role);
    // Clear chat and query results when role changes
    store.setChatMessages([]);
    store.setQueryResults(null);
  }, [store]);

  const currentRoleConfig = ROLES[store.currentRole];

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950 text-zinc-100">
      {/* ─── Header ─── */}
      <header className="border-b border-zinc-800 px-4 py-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Shield className="h-6 w-6 text-emerald-400" />
              <h1 className="text-lg font-bold tracking-tight">ShieldGate</h1>
              <Badge variant="outline" className="text-emerald-400 border-emerald-400/30 text-xs">
                AuthZed x Splunk
              </Badge>
            </div>
            <Separator orientation="vertical" className="h-6 bg-zinc-700" />
            <span className="text-xs text-zinc-500">Least-Privilege Agentic SOC</span>
          </div>

          {/* Role Switcher */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Active Role:</span>
            <Select value={store.currentRole} onValueChange={(v) => handleRoleChange(v as UserRole)}>
              <SelectTrigger className="w-52 bg-zinc-900 border-zinc-700 text-xs h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-zinc-900 border-zinc-700">
                {(Object.entries(ROLES) as [UserRole, typeof ROLES[UserRole]][]).map(([key, val]) => (
                  <SelectItem key={key} value={key} className="text-xs">
                    <div className="flex items-center gap-2">
                      <div className={`${val.color} p-0.5 rounded`}>
                        {roleIcons[key]}
                      </div>
                      <span>{val.label}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className={`${currentRoleConfig.color} px-2 py-0.5 rounded text-xs text-white font-medium`}>
              {currentRoleConfig.label}
            </div>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="flex items-center gap-4 mt-2 text-xs text-zinc-400">
          <div className="flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
            <span>{stats.critical} Critical</span>
          </div>
          <div className="flex items-center gap-1.5">
            <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
            <span>{stats.openIncidents} Open</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            <span>{permStats.allowed} Allowed</span>
          </div>
          <div className="flex items-center gap-1.5">
            <XCircle className="h-3.5 w-3.5 text-red-400" />
            <span>{permStats.denied} Denied</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Activity className="h-3.5 w-3.5 text-blue-400" />
            <span>{stats.auditLogs} Audit Events</span>
          </div>
        </div>
      </header>

      {/* ─── Main Content ─── */}
      <main className="flex-1 flex overflow-hidden">
        {/* ─── Left Panel: Incidents ─── */}
        <aside className="w-80 border-r border-zinc-800 flex flex-col shrink-0">
          <div className="p-3 border-b border-zinc-800">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <FileWarning className="h-4 w-4 text-amber-400" />
                Incidents
              </h2>
              <Badge variant="secondary" className="text-xs">{store.incidents.length}</Badge>
            </div>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-2 space-y-1">
              {store.incidents.map((inc) => {
                const sev = severityConfig[inc.severity] || severityConfig.low;
                const isSelected = store.selectedIncident?.id === inc.id;
                const canView = store.currentRole === 'soc_tier1' || store.currentRole === 'soc_tier2' ||
                  inc.sourceIndex !== 'security' ||
                  (store.currentRole === 'ai_agent') ||
                  (store.currentRole === 'contractor');

                return (
                  <button
                    key={inc.id}
                    onClick={() => canView && store.setSelectedIncident(inc)}
                    className={`w-full text-left p-2.5 rounded-lg transition-all text-xs ${
                      isSelected
                        ? 'bg-zinc-800 border border-zinc-600'
                        : canView
                        ? 'hover:bg-zinc-800/50 border border-transparent'
                        : 'opacity-40 cursor-not-allowed border border-transparent'
                    }`}
                    disabled={!canView}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate text-zinc-200">{inc.title}</p>
                        <div className="flex items-center gap-1.5 mt-1">
                          <Badge className={`${sev.bg} ${sev.text} ${sev.border} text-[10px] px-1.5 py-0`}>
                            {inc.severity.toUpperCase()}
                          </Badge>
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-zinc-500">
                            {inc.sourceIndex}
                          </Badge>
                        </div>
                      </div>
                      {!canView && (
                        <Lock className="h-3.5 w-3.5 text-red-400 shrink-0 mt-0.5" />
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-1.5 text-zinc-500">
                      <Clock className="h-3 w-3" />
                      <span>{new Date(inc.createdAt).toLocaleTimeString()}</span>
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        {inc.status}
                      </Badge>
                    </div>
                  </button>
                );
              })}
            </div>
          </ScrollArea>
        </aside>

        {/* ─── Center + Right Panels ─── */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* ─── Top Section: Tabs ─── */}
          <div className="flex-1 flex overflow-hidden">
            {/* ─── Chat Panel ─── */}
            <div className="flex-1 flex flex-col border-r border-zinc-800">
              <div className="p-3 border-b border-zinc-800">
                <h2 className="text-sm font-semibold flex items-center gap-2">
                  <Bot className="h-4 w-4 text-violet-400" />
                  AI Incident Investigator
                  <Badge variant="outline" className="text-[10px] text-violet-400 border-violet-400/30">
                    {currentRoleConfig.label}
                  </Badge>
                </h2>
              </div>

              {/* Chat Messages */}
              <ScrollArea className="flex-1 p-4">
                {store.chatMessages.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-full text-zinc-500 text-xs gap-2">
                    <Bot className="h-8 w-8 text-zinc-700" />
                    <p>Select an incident and ask me to investigate</p>
                    <div className="text-zinc-600 text-[10px] space-y-1">
                      <p>&quot;Investigate the data exfiltration incident&quot;</p>
                      <p>&quot;Suggest SPL queries for this brute force&quot;</p>
                      <p>&quot;What containment steps should I take?&quot;</p>
                    </div>
                  </div>
                )}
                {store.chatMessages.map((msg, i) => (
                  <div key={i} className={`mb-3 ${msg.role === 'user' ? 'flex justify-end' : ''}`}>
                    {msg.role === 'assistant' && (
                      <div className="flex items-start gap-2">
                        <Avatar className="h-6 w-6 mt-0.5">
                          <AvatarFallback className="bg-violet-500/20 text-violet-400 text-[10px]">
                            <Bot className="h-3.5 w-3.5" />
                          </AvatarFallback>
                        </Avatar>
                        <div className="bg-zinc-800/50 rounded-lg p-2.5 max-w-[85%] text-xs text-zinc-300 leading-relaxed">
                          <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>
                        </div>
                      </div>
                    )}
                    {msg.role === 'user' && (
                      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-2.5 max-w-[70%] ml-auto text-xs text-emerald-200">
                        {msg.content}
                      </div>
                    )}
                  </div>
                ))}
                {store.isChatLoading && (
                  <div className="flex items-center gap-2 text-zinc-500 text-xs">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Analyzing incident...</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </ScrollArea>

              {/* Chat Input */}
              <div className="p-3 border-t border-zinc-800">
                <div className="flex gap-2">
                  <Textarea
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendChatMessage();
                      }
                    }}
                    placeholder="Ask about the incident..."
                    className="min-h-[36px] max-h-[80px] resize-none bg-zinc-900 border-zinc-700 text-xs"
                    rows={1}
                  />
                  <Button
                    onClick={sendChatMessage}
                    disabled={!chatInput.trim() || store.isChatLoading}
                    size="sm"
                    className="bg-emerald-600 hover:bg-emerald-700 text-white h-9 w-9 p-0 shrink-0"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* ─── Right Panel: Evidence + AuthZ Timeline ─── */}
            <div className="w-[420px] flex flex-col shrink-0">
              <Tabs value={store.activePanel} onValueChange={(v) => store.setActivePanel(v as typeof store.activePanel)} className="flex-1 flex flex-col">
                <TabsList className="bg-zinc-900 rounded-none border-b border-zinc-800 w-full justify-start px-2 h-9">
                  <TabsTrigger value="incidents" className="text-xs data-[state=active]:bg-zinc-800 h-7 px-3">
                    <Eye className="h-3.5 w-3.5 mr-1" />
                    Evidence
                  </TabsTrigger>
                  <TabsTrigger value="query" className="text-xs data-[state=active]:bg-zinc-800 h-7 px-3">
                    <Terminal className="h-3.5 w-3.5 mr-1" />
                    SPL Query
                  </TabsTrigger>
                  <TabsTrigger value="alerts" className="text-xs data-[state=active]:bg-zinc-800 h-7 px-3">
                    <Zap className="h-3.5 w-3.5 mr-1" />
                    AuthZ Log
                  </TabsTrigger>
                </TabsList>

                {/* Evidence Panel */}
                <TabsContent value="incidents" className="flex-1 overflow-hidden m-0">
                  <ScrollArea className="h-full">
                    <div className="p-3">
                      {store.selectedIncident ? (
                        <div className="space-y-4">
                          <div>
                            <div className="flex items-center gap-2 mb-1.5">
                              {(() => {
                                const sev = severityConfig[store.selectedIncident.severity] || severityConfig.low;
                                return <Badge className={`${sev.bg} ${sev.text} ${sev.border} text-[10px]`}>{store.selectedIncident.severity.toUpperCase()}</Badge>;
                              })()}
                              <Badge variant="outline" className="text-[10px]">{store.selectedIncident.status}</Badge>
                              <Badge variant="outline" className="text-[10px]">{store.selectedIncident.sourceIndex}</Badge>
                            </div>
                            <h3 className="text-sm font-semibold text-zinc-200">{store.selectedIncident.title}</h3>
                          </div>

                          <div className="bg-zinc-900 rounded-lg p-3 text-xs space-y-2">
                            <h4 className="font-medium text-zinc-400 text-[10px] uppercase tracking-wider">Description</h4>
                            <p className="text-zinc-300 leading-relaxed">{store.selectedIncident.description}</p>
                          </div>

                          <div className="bg-zinc-900 rounded-lg p-3 text-xs space-y-2">
                            <h4 className="font-medium text-zinc-400 text-[10px] uppercase tracking-wider">Raw Event</h4>
                            <pre className="text-[11px] text-emerald-400/80 font-mono whitespace-pre-wrap bg-black/30 rounded p-2 overflow-x-auto">
                              {JSON.stringify(JSON.parse(store.selectedIncident.rawEvent), null, 2)}
                            </pre>
                          </div>

                          <div className="bg-zinc-900 rounded-lg p-3 text-xs space-y-2">
                            <h4 className="font-medium text-zinc-400 text-[10px] uppercase tracking-wider">AuthZed Access Check</h4>
                            <div className="flex items-center gap-2">
                              {store.currentRole === 'contractor' && store.selectedIncident.sourceIndex === 'security' ? (
                                <>
                                  <XCircle className="h-4 w-4 text-red-400" />
                                  <span className="text-red-400">Data will be redacted for contractor role</span>
                                </>
                              ) : (
                                <>
                                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                                  <span className="text-emerald-400">Accessible — {currentRoleConfig.label}</span>
                                </>
                              )}
                            </div>
                          </div>

                          {/* Quick Actions */}
                          <div className="space-y-1.5">
                            <h4 className="font-medium text-zinc-400 text-[10px] uppercase tracking-wider">Quick SPL Queries</h4>
                            {[
                              `index=${store.selectedIncident.sourceIndex} severity=${store.selectedIncident.severity} | stats count`,
                              `index=${store.selectedIncident.sourceIndex} | stats count by src_ip, action`,
                              `index=${store.selectedIncident.sourceIndex} sourcetype=auth | table _time, user, action, src_ip`,
                            ].map((q, i) => (
                              <button
                                key={i}
                                onClick={() => {
                                  setSplQuery(q);
                                  store.setActivePanel('query');
                                }}
                                className="w-full text-left px-3 py-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-[11px] text-emerald-400 font-mono flex items-center gap-2 transition-colors"
                              >
                                <Terminal className="h-3 w-3" />
                                <span className="truncate">{q}</span>
                                <ChevronRight className="h-3 w-3 ml-auto shrink-0 text-zinc-600" />
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center h-40 text-zinc-600 text-xs gap-2">
                          <Eye className="h-6 w-6" />
                          <p>Select an incident to view evidence</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>

                {/* SPL Query Panel */}
                <TabsContent value="query" className="flex-1 flex flex-col overflow-hidden m-0">
                  <div className="p-3 border-b border-zinc-800">
                    <div className="flex gap-2">
                      <Input
                        value={splQuery}
                        onChange={(e) => setSplQuery(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') runSplunkQuery();
                        }}
                        placeholder='index=security severity=critical | stats count by action'
                        className="bg-zinc-900 border-zinc-700 text-xs font-mono h-9"
                      />
                      <Button
                        onClick={runSplunkQuery}
                        disabled={!splQuery.trim() || store.isLoading}
                        size="sm"
                        className="bg-emerald-600 hover:bg-emerald-700 text-white h-9 shrink-0"
                      >
                        {store.isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                      </Button>
                    </div>
                    <div className="flex items-center gap-1.5 mt-1.5 text-[10px] text-zinc-500">
                      <Lock className="h-3 w-3" />
                      <span>Every query passes through AuthZed before reaching Splunk</span>
                    </div>
                  </div>

                  <ScrollArea className="flex-1">
                    <div className="p-3">
                      {store.queryResults ? (
                        <div className="space-y-3">
                          {/* Auth Decision */}
                          <div className={`rounded-lg p-2.5 text-xs flex items-start gap-2 ${
                            store.queryResults.authorized
                              ? 'bg-emerald-500/10 border border-emerald-500/20'
                              : 'bg-red-500/10 border border-red-500/20'
                          }`}>
                            {store.queryResults.authorized ? (
                              <Unlock className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                            ) : (
                              <Lock className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
                            )}
                            <div>
                              <p className={`font-medium ${store.queryResults.authorized ? 'text-emerald-400' : 'text-red-400'}`}>
                                {store.queryResults.authorized ? 'AuthZed: ALLOW' : 'AuthZed: DENY'}
                              </p>
                              <p className="text-zinc-400 mt-0.5">{store.queryResults.reason}</p>
                            </div>
                          </div>

                          {/* Results */}
                          {store.queryResults.results && store.queryResults.results.length > 0 ? (
                            <div>
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">
                                  {store.queryResults.eventCount} events ({store.queryResults.runDurationMs}ms)
                                </span>
                                {store.queryResults.note && (
                                  <Badge variant="outline" className="text-[10px] text-amber-400 border-amber-400/30">
                                    {store.queryResults.note}
                                  </Badge>
                                )}
                              </div>
                              <div className="space-y-1.5">
                                {store.queryResults.results.map((row, i) => (
                                  <div key={i} className="bg-zinc-900 rounded-lg p-2 text-[11px] font-mono">
                                    {Object.entries(row).map(([k, v]) => (
                                      <div key={k} className="flex gap-2">
                                        <span className="text-zinc-500 min-w-[80px]">{k}:</span>
                                        <span className={
                                          k === 'severity'
                                            ? v === 'critical' ? 'text-red-400' : v === 'high' ? 'text-amber-400' : 'text-zinc-300'
                                            : 'text-zinc-300'
                                        }>{v}</span>
                                      </div>
                                    ))}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : store.queryResults.authorized ? (
                            <div className="text-center text-zinc-500 text-xs py-6">
                              <Database className="h-6 w-6 mx-auto mb-2 opacity-50" />
                              No events found for this query
                            </div>
                          ) : null}
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center h-40 text-zinc-600 text-xs gap-2">
                          <Terminal className="h-6 w-6" />
                          <p>Enter a SPL query to investigate</p>
                          <p className="text-zinc-700">AuthZed will check permissions first</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>

                {/* AuthZ Log Panel */}
                <TabsContent value="alerts" className="flex-1 overflow-hidden m-0">
                  <ScrollArea className="h-full">
                    <div className="p-2 space-y-1">
                      {store.auditLogs.map((log) => (
                        <div
                          key={log.id}
                          className={`rounded-lg p-2.5 text-xs border ${
                            log.decision === 'ALLOW'
                              ? 'bg-emerald-500/5 border-emerald-500/10'
                              : 'bg-red-500/5 border-red-500/10'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-1.5">
                              {log.decision === 'ALLOW' ? (
                                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                              ) : (
                                <XCircle className="h-3.5 w-3.5 text-red-400" />
                              )}
                              <span className={`font-medium ${log.decision === 'ALLOW' ? 'text-emerald-400' : 'text-red-400'}`}>
                                {log.decision}
                              </span>
                            </div>
                            <span className="text-[10px] text-zinc-600">{new Date(log.timestamp).toLocaleTimeString()}</span>
                          </div>
                          <div className="flex items-center gap-2 text-zinc-400">
                            <Badge variant="outline" className="text-[10px] px-1.5 py-0">{log.userRole}</Badge>
                            <code className="text-[11px] font-mono text-zinc-500">{log.action}</code>
                            {log.resource && (
                              <span className="text-[10px] text-zinc-600">on {log.resource}</span>
                            )}
                          </div>
                          <p className="text-[11px] text-zinc-500 mt-1">{log.reason}</p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            </div>
          </div>

          {/* ─── Bottom Panel: AuthZed Schema + Permission Matrix ─── */}
          <div className="h-48 border-t border-zinc-800 shrink-0 overflow-hidden">
            <div className="p-3 border-b border-zinc-800">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <Database className="h-4 w-4 text-sky-400" />
                AuthZed SpiceDB Schema
                <Badge variant="outline" className="text-[10px] text-sky-400 border-sky-400/30">
                  Live Permission Engine
                </Badge>
              </h2>
            </div>
            <ScrollArea className="h-[calc(100%-44px)]">
              <div className="p-3">
                <pre className="text-[11px] font-mono text-zinc-400 leading-relaxed">
{`definition user {}

definition team {
  relation member: user
  permission member_access = member
}

definition splunk_index {
  relation viewer:   team | user
  relation querier:  team | user
  relation admin:    user
  permission read    = viewer + querier + admin
  permission query   = querier + admin
  permission manage  = admin
}

definition splunk_tool {
  relation allowed_role: team
  relation allowed_user: user
  permission execute = allowed_user + allowed_role->member_access - restricted
}

definition incident {
  relation index:          splunk_index
  relation assigned_team:  team
  relation viewer:         user
  permission view = viewer + assigned_team->member_access + index->read
}`}
                </pre>
              </div>
            </ScrollArea>
          </div>
        </div>
      </main>
    </div>
  );
}
