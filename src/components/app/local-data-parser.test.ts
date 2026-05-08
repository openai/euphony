import { describe, expect, it } from 'vitest';
import { parseLocalData } from './local-data-parser';

function buildSimpleAtif() {
  return {
    schema_version: 'ATIF-v1.6',
    session_id: 'parser-test-session',
    agent: { name: 'parser-agent', version: '0.1.0' },
    steps: [
      {
        step_id: 1,
        timestamp: '2026-05-06T12:30:00Z',
        source: 'system',
        message: 'You are concise.'
      },
      {
        step_id: 2,
        timestamp: '2026-05-06T12:30:01Z',
        source: 'user',
        message: 'Hello'
      }
    ]
  };
}

function buildCodexJSONL(): string {
  // Minimal Codex session JSONL: a session_meta event followed by a
  // response_item event, matching what `isCodexSessionJSONL` accepts.
  const sessionMeta = {
    timestamp: '2026-05-06T12:30:00Z',
    type: 'session_meta',
    payload: { id: 'codex-session-1', cwd: '/tmp', originator: 'unit-test' }
  };
  const responseItem = {
    timestamp: '2026-05-06T12:30:01Z',
    type: 'response_item',
    payload: { type: 'message', role: 'user', content: [] }
  };
  return `${JSON.stringify(sessionMeta)}\n${JSON.stringify(responseItem)}`;
}

describe('parseLocalData', () => {
  it('routes a single ATIF JSON document to the atif branch', () => {
    const result = parseLocalData(JSON.stringify(buildSimpleAtif()));
    expect(result.dataType).toBe('atif');
    if (result.dataType === 'atif') {
      expect(result.atifTrajectoryData.length).toBe(1);
    }
  });

  it('detects ATIF before Codex even if the doc contains event-shaped fields', () => {
    // ATIF docs are allowed to carry extra fields; this fixture mimics the
    // shape that previously could be misclassified as Codex JSONL.
    const atif = {
      ...buildSimpleAtif(),
      extra: { type: 'response_item', payload: { type: 'message' } }
    };
    const result = parseLocalData(JSON.stringify(atif));
    expect(result.dataType).toBe('atif');
  });

  it('routes Codex session JSONL to the codex branch', () => {
    const result = parseLocalData(buildCodexJSONL());
    expect(result.dataType).toBe('codex');
    if (result.dataType === 'codex') {
      expect(result.codexSessionData.length).toBe(2);
    }
  });

  it('routes harmony conversation JSONL to the conversation branch', () => {
    const conversation = {
      id: 'conv-1',
      messages: [{ role: 'user', content: [{ text: 'hi' }] }]
    };
    const result = parseLocalData(JSON.stringify(conversation));
    expect(result.dataType).toBe('conversation');
    if (result.dataType === 'conversation') {
      expect(result.conversationData[0].id).toBe('conv-1');
    }
  });

  it('falls back to the json branch when no harmony shape is found', () => {
    const result = parseLocalData(JSON.stringify({ unknown: 'shape' }));
    expect(result.dataType).toBe('json');
    if (result.dataType === 'json') {
      expect(result.jsonData[0]).toEqual({ unknown: 'shape' });
    }
  });

  it('throws when no JSON or JSONL data could be parsed', () => {
    expect(() => parseLocalData('')).toThrow(
      /Failed to read any JSON or JSONL data/
    );
  });
});
