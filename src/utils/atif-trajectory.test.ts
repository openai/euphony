import { describe, expect, it } from 'vitest';
import { Role } from '../types/harmony-types';
import {
  isAtifTrajectory,
  parseAtifTrajectory,
  serializeAtifTrajectory,
  toAtifJSONString,
  validateAtifTrajectory
} from './atif-trajectory';

function firstContentText(content: unknown): string {
  if (!Array.isArray(content) || content.length === 0) return '';
  const first = content[0] as unknown;
  if (typeof first === 'string') return first;
  if (
    typeof first === 'object' &&
    first !== null &&
    'text' in first &&
    typeof (first as { text?: unknown }).text === 'string'
  ) {
    return (first as { text: string }).text;
  }
  return '';
}

function buildSimpleAtif() {
  return {
    schema_version: 'ATIF-v1.6',
    session_id: 'atif-test-session-001',
    agent: {
      name: 'test-agent',
      version: '0.1.0',
      model_name: 'gpt-4o-mini'
    },
    notes: 'ATIF parser test fixture',
    steps: [
      {
        step_id: 1,
        timestamp: '2026-05-06T12:30:00Z',
        source: 'system',
        message: 'You are concise.'
      },
      {
        step_id: 2,
        timestamp: '2026-05-06T12:30:02Z',
        source: 'user',
        message: 'What is 2 + 2?'
      },
      {
        step_id: 3,
        timestamp: '2026-05-06T12:30:03Z',
        source: 'agent',
        message: "I'll calculate it.",
        reasoning_content: 'Simple arithmetic reasoning.',
        tool_calls: [
          {
            tool_call_id: 'call-1',
            function_name: 'calculator',
            arguments: { expression: '2+2' }
          }
        ],
        observation: {
          results: [{ source_call_id: 'call-1', content: '4' }]
        }
      },
      {
        step_id: 4,
        timestamp: '2026-05-06T12:30:04Z',
        source: 'agent',
        message: '2 + 2 = 4.'
      }
    ]
  };
}

describe('ATIF detector', () => {
  it('accepts both bare and one-element wrapped trajectories', () => {
    const atif = buildSimpleAtif();
    expect(isAtifTrajectory(atif)).toBe(true);
    expect(isAtifTrajectory([atif])).toBe(true);
  });

  it('rejects non-ATIF shaped payloads', () => {
    const atif = buildSimpleAtif();
    expect(
      isAtifTrajectory({
        ...atif,
        schema_version: 'NOT-ATIF'
      })
    ).toBe(false);
    expect(isAtifTrajectory([{ foo: 'bar' }])).toBe(false);
  });
});

describe('ATIF parser', () => {
  it('parses messages, reasoning, tool calls and observations', () => {
    const parsed = parseAtifTrajectory(buildSimpleAtif());
    expect(parsed).not.toBeNull();

    const conversation = parsed!.conversation;
    expect(conversation.id).toBe('atif-test-session-001');
    expect(conversation.messages.length).toBeGreaterThanOrEqual(7);

    const summary = conversation.messages[0];
    expect(summary.role).toBe(Role.System);
    expect(firstContentText(summary.content)).toContain('Harbor ATIF trajectory');

    const userMessage = conversation.messages.find(
      m =>
        m.role === Role.User &&
        firstContentText(m.content).includes('What is 2 + 2?')
    );
    expect(userMessage).toBeDefined();

    const reasoningMessage = conversation.messages.find(
      m =>
        m.channel === 'analysis' &&
        firstContentText(m.content).includes('arithmetic')
    );
    expect(reasoningMessage).toBeDefined();

    const toolCallMessage = conversation.messages.find(
      m => m.channel === 'call' && m.name === 'calculator'
    );
    expect(toolCallMessage).toBeDefined();

    const toolOutputMessage = conversation.messages.find(
      m => m.channel === 'output' && m.name === 'calculator'
    );
    expect(toolOutputMessage).toBeDefined();
    expect(firstContentText(toolOutputMessage!.content)).toContain('4');
  });

  it('returns null for non-ATIF payloads', () => {
    expect(parseAtifTrajectory([{ foo: 'bar' }])).toBeNull();
  });
});

describe('ATIF serializer and validator', () => {
  it('round-trips through conversation metadata', () => {
    const atif = buildSimpleAtif();
    const parsed = parseAtifTrajectory(atif);
    expect(parsed).not.toBeNull();

    const serialized = serializeAtifTrajectory(parsed!.conversation);
    expect(serialized).not.toBeNull();
    expect(serialized).toEqual(atif);

    const jsonString = toAtifJSONString(parsed!.conversation);
    expect(jsonString).not.toBeNull();
    expect(JSON.parse(jsonString!)).toEqual(atif);
  });

  it('reports schema and step-reference validation errors', () => {
    const baseAtif = buildSimpleAtif();
    const atifWithErrors = {
      ...baseAtif,
      schema_version: 'ATIF-v9.9',
      steps: [
        {
          ...baseAtif.steps[0],
          step_id: 2
        },
        {
          ...baseAtif.steps[1],
          step_id: 2
        },
        {
          ...baseAtif.steps[2],
          step_id: 3,
          observation: {
            results: [{ source_call_id: 'missing-call-id', content: '4' }]
          }
        }
      ]
    };

    const errors = validateAtifTrajectory(atifWithErrors);
    expect(
      errors.some(error => error.includes('not in the list of supported versions'))
    ).toBe(true);
    expect(errors.some(error => error.includes('steps[0].step_id'))).toBe(true);
    expect(
      errors.some(error => error.includes('does not reference any tool_call_id'))
    ).toBe(true);
  });

  it('rejects agent-only fields on non-agent steps', () => {
    const atif = buildSimpleAtif();
    const invalid = {
      ...atif,
      steps: [
        {
          ...atif.steps[0],
          source: 'user',
          reasoning_content: 'should not be here',
          tool_calls: [
            {
              tool_call_id: 'x',
              function_name: 'tool',
              arguments: {}
            }
          ]
        },
        ...atif.steps.slice(1)
      ]
    };

    const errors = validateAtifTrajectory(invalid);
    expect(
      errors.some(error =>
        error.includes('agent-only field') && error.includes('reasoning_content')
      )
    ).toBe(true);
    expect(
      errors.some(error =>
        error.includes('agent-only field') && error.includes('tool_calls')
      )
    ).toBe(true);
  });

  it('validates content parts and tool_definitions shape', () => {
    const atif = buildSimpleAtif();
    const invalid = {
      ...atif,
      agent: {
        ...atif.agent,
        tool_definitions: [{ type: 'function' }]
      },
      steps: [
        {
          ...atif.steps[0],
          message: [{ type: 'text' }]
        },
        ...atif.steps.slice(1)
      ]
    };

    const errors = validateAtifTrajectory(invalid);
    expect(
      errors.some(error => error.includes('agent.tool_definitions[0].function'))
    ).toBe(true);
    expect(
      errors.some(error =>
        error.includes("steps[0].message[0] text part requires string 'text'")
      )
    ).toBe(true);
  });

  // Harbor's `ImageSource.media_type` is a Pydantic Literal restricted to four
  // MIME types — see harbor/src/harbor/models/trajectories/content.py.
  // The validator should reject any other string.
  it('rejects image parts with disallowed media_type values', () => {
    const atif = buildSimpleAtif();
    const invalid = {
      ...atif,
      steps: [
        {
          ...atif.steps[0],
          message: [
            {
              type: 'image',
              source: { media_type: 'image/bmp', path: 'foo.bmp' }
            }
          ]
        },
        ...atif.steps.slice(1)
      ]
    };

    const errors = validateAtifTrajectory(invalid);
    expect(
      errors.some(
        error =>
          error.includes('steps[0].message[0]') && error.includes('media_type')
      )
    ).toBe(true);
  });

  // Harbor's `ContentPart.validate_content_type` rejects extra fields:
  //   - text parts may not carry `source`
  //   - image parts may not carry `text`
  // See harbor/src/harbor/models/trajectories/content.py.
  it('rejects content parts that mix text and image fields', () => {
    const atif = buildSimpleAtif();
    const invalid = {
      ...atif,
      steps: [
        {
          ...atif.steps[0],
          message: [
            {
              type: 'text',
              text: 'hi',
              source: { media_type: 'image/png', path: 'x.png' }
            }
          ]
        },
        {
          ...atif.steps[1],
          message: [
            {
              type: 'image',
              text: 'caption',
              source: { media_type: 'image/png', path: 'x.png' }
            }
          ]
        },
        ...atif.steps.slice(2)
      ]
    };

    const errors = validateAtifTrajectory(invalid);
    expect(
      errors.some(
        error =>
          error.includes('steps[0].message[0]') &&
          error.includes("'source' is not allowed")
      )
    ).toBe(true);
    expect(
      errors.some(
        error =>
          error.includes('steps[1].message[0]') &&
          error.includes("'text' is not allowed")
      )
    ).toBe(true);
  });

  it('rejects timestamps that are not strict ISO-8601', () => {
    const atif = buildSimpleAtif();
    const invalid = {
      ...atif,
      steps: [
        { ...atif.steps[0], timestamp: '2026-Jan-01' },
        { ...atif.steps[1], timestamp: '2026/05/06 12:30:02' },
        { ...atif.steps[2] },
        { ...atif.steps[3] }
      ]
    };

    const errors = validateAtifTrajectory(invalid);
    expect(
      errors.some(
        error =>
          error.includes('steps[0].timestamp') && error.includes('ISO-8601')
      )
    ).toBe(true);
    expect(
      errors.some(
        error =>
          error.includes('steps[1].timestamp') && error.includes('ISO-8601')
      )
    ).toBe(true);
  });

  it('accepts canonical ISO-8601 timestamps', () => {
    const atif = buildSimpleAtif();
    const errors = validateAtifTrajectory(atif);
    expect(errors.some(error => error.includes('timestamp'))).toBe(false);
  });
});
