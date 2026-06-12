import { create } from 'zustand';
import type { UserRole } from '@/lib/authz-types';

interface Incident {
  id: string;
  severity: string;
  status: string;
  title: string;
  description: string;
  sourceIndex: string;
  assignedTeam: string;
  rawEvent: string;
  createdAt: string;
  updatedAt: string;
}

interface AuditLogEntry {
  id: string;
  userId: string;
  userRole: string;
  action: string;
  resource: string;
  decision: string;
  reason: string;
  timestamp: string;
}

interface QueryResult {
  authorized: boolean;
  error?: string;
  reason?: string;
  sid?: string;
  results?: Array<Record<string, string>>;
  eventCount?: number;
  runDurationMs?: number;
  note?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
}

interface ShieldGateState {
  // Current role
  currentRole: UserRole;
  setCurrentRole: (role: UserRole) => void;

  // Incidents
  incidents: Incident[];
  setIncidents: (incidents: Incident[]) => void;
  selectedIncident: Incident | null;
  setSelectedIncident: (incident: Incident | null) => void;
  updateIncidentStatus: (id: string, status: string) => void;

  // Audit logs
  auditLogs: AuditLogEntry[];
  setAuditLogs: (logs: AuditLogEntry[]) => void;
  addAuditLog: (log: AuditLogEntry) => void;

  // Query results
  queryResults: QueryResult | null;
  setQueryResults: (results: QueryResult | null) => void;

  // Chat
  chatMessages: ChatMessage[];
  setChatMessages: (messages: ChatMessage[]) => void;
  addChatMessage: (message: ChatMessage) => void;

  // UI state
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  isChatLoading: boolean;
  setIsChatLoading: (loading: boolean) => void;

  // Active panel
  activePanel: 'incidents' | 'query' | 'alerts';
  setActivePanel: (panel: 'incidents' | 'query' | 'alerts') => void;
}

export const useStore = create<ShieldGateState>((set) => ({
  currentRole: 'soc_tier1',
  setCurrentRole: (role) => set({ currentRole: role }),

  incidents: [],
  setIncidents: (incidents) => set({ incidents }),
  selectedIncident: null,
  setSelectedIncident: (incident) => set({ selectedIncident: incident }),
  updateIncidentStatus: (id, status) =>
    set((state) => ({
      incidents: state.incidents.map((inc) =>
        inc.id === id ? { ...inc, status } : inc
      ),
      selectedIncident:
        state.selectedIncident?.id === id
          ? { ...state.selectedIncident, status }
          : state.selectedIncident,
    })),

  auditLogs: [],
  setAuditLogs: (logs) => set({ auditLogs: logs }),
  addAuditLog: (log) =>
    set((state) => ({ auditLogs: [log, ...state.auditLogs] })),

  queryResults: null,
  setQueryResults: (results) => set({ queryResults: results }),

  chatMessages: [],
  setChatMessages: (messages) => set({ chatMessages: messages }),
  addChatMessage: (message) =>
    set((state) => ({ chatMessages: [...state.chatMessages, message] })),

  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  isChatLoading: false,
  setIsChatLoading: (loading) => set({ isChatLoading: loading }),

  activePanel: 'incidents',
  setActivePanel: (panel) => set({ activePanel: panel }),
}));
