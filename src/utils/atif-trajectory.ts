/**
 * Reader, writer, and detector for Harbor's Agent Trajectory Interchange
 * Format (ATIF) — a single-document JSON format (not JSONL) capturing a
 * complete agent run. Mirrors the public surface of `codex-session.ts`. The
 * original document is preserved verbatim under
 * `conversation.metadata.atif_trajectory` for lossless round-trips.
 *
 * Schema reference: harbor/rfcs/0001-trajectory-format.md (ATIF v1.0 — v1.6).
 */

import { config } from '../config/config';
import type { Conversation, Message } from '../types/harmony-types';
import { Role } from '../types/harmony-types';

// ATIF schema types — mirror of harbor/src/harbor/models/trajectories/*.py.
// Extra fields are tolerated where the spec does, so provider-specific
// metadata round-trips without loss.

/** Versions this module knows how to read and write. */
export const SUPPORTED_ATIF_SCHEMA_VERSIONS = [
  'ATIF-v1.0',
  'ATIF-v1.1',
  'ATIF-v1.2',
  'ATIF-v1.3',
  'ATIF-v1.4',
  'ATIF-v1.5',
  'ATIF-v1.6'
] as const;

export type AtifSchemaVersion = (typeof SUPPORTED_ATIF_SCHEMA_VERSIONS)[number];

export type AtifStepSource = 'system' | 'user' | 'agent';

export type AtifImageMediaType =
  | 'image/jpeg'
  | 'image/png'
  | 'image/gif'
  | 'image/webp';

export interface AtifImageSource {
  media_type: AtifImageMediaType;
  path: string;
}

export interface AtifContentPart {
  type: 'text' | 'image';
  text?: string;
  source?: AtifImageSource;
}

export interface AtifToolCall {
  tool_call_id: string;
  function_name: string;
  arguments: Record<string, unknown>;
}

export interface AtifSubagentTrajectoryRef {
  session_id: string;
  trajectory_path?: string | null;
  extra?: Record<string, unknown> | null;
}

export interface AtifObservationResult {
  source_call_id?: string | null;
  content?: string | AtifContentPart[] | null;
  subagent_trajectory_ref?: AtifSubagentTrajectoryRef[] | null;
}

export interface AtifObservation {
  results: AtifObservationResult[];
}

export interface AtifMetrics {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  cached_tokens?: number | null;
  cost_usd?: number | null;
  prompt_token_ids?: number[] | null;
  completion_token_ids?: number[] | null;
  logprobs?: number[] | null;
  extra?: Record<string, unknown> | null;
}

export interface AtifFinalMetrics {
  total_prompt_tokens?: number | null;
  total_completion_tokens?: number | null;
  total_cached_tokens?: number | null;
  total_cost_usd?: number | null;
  total_steps?: number | null;
  extra?: Record<string, unknown> | null;
}

export interface AtifAgent {
  name: string;
  version: string;
  model_name?: string | null;
  tool_definitions?: Record<string, unknown>[] | null;
  extra?: Record<string, unknown> | null;
}

export interface AtifStep {
  step_id: number;
  timestamp?: string | null;
  source: AtifStepSource;
  model_name?: string | null;
  reasoning_effort?: string | number | null;
  message: string | AtifContentPart[];
  reasoning_content?: string | null;
  tool_calls?: AtifToolCall[] | null;
  observation?: AtifObservation | null;
  metrics?: AtifMetrics | null;
  is_copied_context?: boolean | null;
  extra?: Record<string, unknown> | null;
}

export interface AtifTrajectory {
  // Plain string so unknown future ATIF versions still parse;
  // `SUPPORTED_ATIF_SCHEMA_VERSIONS` is enforced in `validateAtifTrajectory`.
  schema_version: string;
  session_id: string;
  agent: AtifAgent;
  steps: AtifStep[];
  notes?: string | null;
  final_metrics?: AtifFinalMetrics | null;
  continued_trajectory_ref?: string | null;
  extra?: Record<string, unknown> | null;
}

export interface AtifTrajectoryParseResult {
  conversation: Conversation;
  customLabels: string[][];
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isAtifVersion = (value: unknown): value is string =>
  typeof value === 'string' && value.startsWith('ATIF-');

// Strict ISO-8601 with date + time + timezone (Z or +/-HH:MM). Rejects the
// engine-permissive shapes Date.parse accepts (e.g. "2025-Jan-01"). Mirrors
// the constraint harbor enforces server-side via
// `datetime.fromisoformat(v.replace("Z", "+00:00"))` in
// harbor/src/harbor/models/trajectories/step.py (`Step.validate_timestamp`).
const STRICT_ISO_8601_RE =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;

function isStrictIso8601(value: unknown): value is string {
  if (typeof value !== 'string' || !STRICT_ISO_8601_RE.test(value)) {
    return false;
  }
  return !Number.isNaN(Date.parse(value));
}

const isPlausibleAtifStep = (value: unknown): boolean =>
  isRecord(value) &&
  typeof value.step_id === 'number' &&
  (value.source === 'system' ||
    value.source === 'user' ||
    value.source === 'agent') &&
  (typeof value.message === 'string' || Array.isArray(value.message));

/**
 * Returns true if `raw` looks like an ATIF trajectory document. Accepts the
 * trajectory object itself or a one-element array wrapping it (what
 * `parseSourceText` produces when JSON.parse succeeds on a single document).
 */
export const isAtifTrajectory = (raw: unknown): boolean => {
  let candidate: unknown = raw;
  if (Array.isArray(raw)) {
    if (raw.length !== 1) return false;
    candidate = raw[0];
  }
  if (!isRecord(candidate)) return false;
  if (!isAtifVersion(candidate.schema_version)) return false;
  if (typeof candidate.session_id !== 'string') return false;

  const agent = candidate.agent;
  if (
    !isRecord(agent) ||
    typeof agent.name !== 'string' ||
    typeof agent.version !== 'string'
  ) {
    return false;
  }

  if (!Array.isArray(candidate.steps) || candidate.steps.length === 0) {
    return false;
  }
  // Sample-check up to 5 steps to keep detection cheap on huge trajectories.
  const sampleSize = Math.min(candidate.steps.length, 5);
  for (let i = 0; i < sampleSize; i += 1) {
    if (!isPlausibleAtifStep(candidate.steps[i])) return false;
  }
  return true;
};

// ---------------------------------------------------------------------------
// Reader: ATIF -> Conversation
// ---------------------------------------------------------------------------

/** Coerce a possibly-array `raw` into the bare trajectory object. */
const unwrapTrajectory = (raw: unknown): AtifTrajectory | null => {
  const candidate: unknown =
    Array.isArray(raw) && raw.length === 1 ? raw[0] : raw;
  return isRecord(candidate) ? (candidate as unknown as AtifTrajectory) : null;
};

const formatJSON = (value: unknown): string => JSON.stringify(value, null, 2);
function cloneJSON<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/**
 * Flatten ATIF content (string or ContentPart[]) to plain text. Image parts
 * become `[image: <path>]` markers. Anything non-conforming is JSON-stringified
 * rather than throwing — detection only sample-checks the first 5 steps, so
 * later steps may have shapes we never validated.
 */
const flattenContent = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return formatJSON(content);

  const parts: string[] = [];
  for (const part of content) {
    if (!isRecord(part)) {
      parts.push(formatJSON(part));
    } else if (part.type === 'text' && typeof part.text === 'string') {
      parts.push(part.text);
    } else if (
      part.type === 'image' &&
      isRecord(part.source) &&
      typeof part.source.path === 'string'
    ) {
      parts.push(`[image: ${part.source.path}]`);
    }
  }
  return parts.join('\n');
};

/**
 * Parse a strict ISO-8601 timestamp into seconds-since-epoch, or null if the
 * value is missing or non-conforming.
 *
 * Shares its format check with `isStrictIso8601` so the parser cannot accept a
 * timestamp shape that the validator would reject — i.e. a doc whose validator
 * errors mention `timestamp must be a strict ISO-8601 string` will also have
 * `create_time === undefined` here, instead of getting a Date.parse-coerced
 * value that disagrees with the validator's verdict.
 */
const parseTimestampSeconds = (timestamp?: string | null): number | null => {
  if (!isStrictIso8601(timestamp)) return null;
  const ms = Date.parse(timestamp);
  return Number.isNaN(ms) ? null : ms / 1000;
};

const sourceToRole = (source: AtifStepSource): Role => {
  switch (source) {
    case 'system':
      return Role.System;
    case 'user':
      return Role.User;
    case 'agent':
      return Role.Assistant;
  }
};

/**
 * Parse an ATIF trajectory into a Euphony Conversation. Returns null if the
 * input doesn't look like ATIF — callers should fall through to other format
 * handlers.
 */
export const parseAtifTrajectory = (
  raw: unknown
): AtifTrajectoryParseResult | null => {
  if (!isAtifTrajectory(raw)) return null;
  const trajectory = unwrapTrajectory(raw);
  if (!trajectory) return null;

  const messages: Message[] = [];
  const sessionId = trajectory.session_id;
  const agentName = trajectory.agent.name;
  const agentVersion = trajectory.agent.version;
  const defaultModel = trajectory.agent.model_name ?? null;

  // First system message — a one-line session summary, mirroring how the
  // Codex parser surfaces session_meta at the top of the conversation.
  const summaryLines = [
    'Harbor ATIF trajectory',
    `session_id: ${sessionId}`,
    `schema: ${trajectory.schema_version}`,
    `agent: ${agentName} (v${agentVersion})`
  ];
  if (defaultModel) summaryLines.push(`model: ${defaultModel}`);
  if (trajectory.notes) summaryLines.push(`notes: ${trajectory.notes}`);
  if (trajectory.continued_trajectory_ref) {
    summaryLines.push(`continued from: ${trajectory.continued_trajectory_ref}`);
  }

  messages.push({
    id: `${sessionId}-summary`,
    role: Role.System,
    name: 'atif',
    content: [{ text: summaryLines.join('\n') }],
    metadata: {
      atif_section: 'summary',
      atif_agent: trajectory.agent,
      atif_schema_version: trajectory.schema_version,
      atif_notes: trajectory.notes ?? null,
      atif_extra: trajectory.extra ?? null,
      atif_continued_trajectory_ref:
        trajectory.continued_trajectory_ref ?? null
    }
  });

  // Walk steps in order. Each yields one primary "dialogue turn" message,
  // plus optional follow-ups for reasoning, tool calls, and observations.
  for (const step of trajectory.steps) {
    const ts = parseTimestampSeconds(step.timestamp);
    const role = sourceToRole(step.source);
    // Clone before stashing in metadata: the same `step` (and its `metrics`)
    // are also referenced by `metadata.atif_trajectory` on the conversation
    // for round-tripping. If a downstream consumer mutates a message's
    // metadata (e.g. an inspector adding annotations) the shared reference
    // would corrupt the source-of-truth document used by
    // `serializeAtifTrajectory`. Cloning is cheap relative to render cost
    // and keeps the round-trip invariant even under unfriendly callers.
    const safeStepForMetadata = cloneJSON(step);

    const baseMetadata: Record<string, unknown> = {
      atif_step_id: step.step_id,
      atif_source: step.source,
      atif_step: safeStepForMetadata
    };
    if (step.is_copied_context) baseMetadata.atif_is_copied_context = true;
    if (step.reasoning_effort != null) {
      baseMetadata.atif_reasoning_effort = step.reasoning_effort;
    }
    if (step.metrics) baseMetadata.atif_metrics = cloneJSON(step.metrics);
    if (step.model_name) baseMetadata.atif_model_name = step.model_name;
    if (step.extra) baseMetadata.atif_step_extra = step.extra;

    // 1. Dialogue turn. Empty messages render as `[empty message]` so they
    //    stay visible in the viewer.
    const messageText = flattenContent(step.message);
    messages.push({
      id: `${sessionId}-step-${step.step_id}`,
      role,
      name: step.source === 'agent' ? agentName : undefined,
      content: [{ text: messageText === '' ? '[empty message]' : messageText }],
      create_time: ts ?? undefined,
      metadata: { ...baseMetadata, atif_section: 'step.message' }
    });

    // 2. Reasoning surfaces on the analysis channel (parity with Codex).
    if (step.reasoning_content && step.source === 'agent') {
      messages.push({
        id: `${sessionId}-step-${step.step_id}-reasoning`,
        role: Role.Assistant,
        name: agentName,
        content: [{ text: step.reasoning_content }],
        create_time: ts ?? undefined,
        channel: 'analysis',
        metadata: { ...baseMetadata, atif_section: 'step.reasoning_content' }
      });
    }

    // 3. Tool calls. Track callId → function_name so the matching
    //    observation can render under the same label.
    //
    // Tool-call `arguments` are stringified eagerly here, matching codex-
    //   session.ts (`code: formatJSON(payload)` ~line 554). The conversation
    //   viewer renders every message we emit, so there is no off-screen work
    //   to skip — eager is simpler and equivalent in cost.
    const callIdToFunctionName = new Map<string, string>();
    for (const toolCall of step.tool_calls ?? []) {
      callIdToFunctionName.set(toolCall.tool_call_id, toolCall.function_name);
      messages.push({
        id: `${sessionId}-step-${step.step_id}-call-${toolCall.tool_call_id}`,
        role: Role.Tool,
        name: toolCall.function_name,
        recipient: toolCall.function_name,
        channel: 'call',
        content: [
          {
            content_type: 'code',
            text: formatJSON(toolCall.arguments),
            language: 'json'
          }
        ],
        create_time: ts ?? undefined,
        metadata: {
          ...baseMetadata,
          atif_section: 'step.tool_calls',
          atif_tool_call_id: toolCall.tool_call_id,
          atif_function_name: toolCall.function_name
        }
      });
    }

    // 4. Observation results. Label preference: matching tool function_name
    //    → source_call_id → 'environment' (non-tool / system observations).
    for (const [idx, result] of step.observation?.results.entries() ?? []) {
      const resultId = `${sessionId}-step-${step.step_id}-obs-${idx}`;
      const callId = result.source_call_id ?? null;
      const label =
        (callId !== null ? callIdToFunctionName.get(callId) : undefined) ??
        callId ??
        'environment';

      const subagentRefs = result.subagent_trajectory_ref;
      if (subagentRefs && subagentRefs.length > 0) {
        messages.push({
          id: resultId,
          role: Role.Tool,
          name: label,
          recipient: label,
          channel: 'output',
          content: [
            {
              content_type: 'code',
              text: formatJSON(subagentRefs),
              language: 'json'
            }
          ],
          create_time: ts ?? undefined,
          metadata: {
            ...baseMetadata,
            atif_section: 'step.observation.subagent_trajectory_ref',
            atif_source_call_id: callId
          }
        });
        continue;
      }

      messages.push({
        id: resultId,
        role: Role.Tool,
        name: label,
        recipient: label,
        channel: 'output',
        content: [
          {
            content_type: 'code',
            text:
              result.content == null
                ? '[empty output]'
                : flattenContent(result.content),
            language: 'text'
          }
        ],
        create_time: ts ?? undefined,
        metadata: {
          ...baseMetadata,
          atif_section: 'step.observation.results',
          atif_source_call_id: callId
        }
      });
    }
  }

  // First-step timestamp anchors conversation create_time; fall back to "now".
  // Codex parsers use `events[0].timestamp` directly (codex-session.ts line
  // ~741) because every Codex event has a timestamp; ATIF's `step.timestamp`
  // is OPTIONAL per the RFC, so we walk steps until we find a parseable one.
  let firstTs: number | null = null;
  for (const step of trajectory.steps) {
    firstTs = parseTimestampSeconds(step.timestamp);
    if (firstTs !== null) break;
  }

  // Side-channel labels shown in the conversation header.
  const customLabels: string[][] = [
    ['Session', sessionId.slice(0, 8), sessionId, config.colors['blue-700']],
    [
      'Agent',
      `${agentName}@${agentVersion}`,
      'ATIF agent name and version',
      config.colors['purple-700']
    ]
  ];
  if (defaultModel) {
    customLabels.push([
      'Model',
      defaultModel,
      'Default model from agent.model_name',
      config.colors['indigo-700']
    ]);
  }
  customLabels.push(
    [
      'Schema',
      trajectory.schema_version,
      'ATIF schema version',
      config.colors['gray-700']
    ],
    [
      'Steps',
      String(trajectory.steps.length),
      'Number of steps in trajectory',
      config.colors['green-700']
    ]
  );

  return {
    conversation: {
      id: sessionId,
      create_time: firstTs ?? Date.now() / 1000,
      messages,
      metadata: {
        // Original document preserved verbatim for lossless round-trips.
        atif_trajectory: trajectory,
        atif_schema_version: trajectory.schema_version,
        atif_agent: trajectory.agent,
        atif_final_metrics: trajectory.final_metrics ?? null,
        atif_step_count: trajectory.steps.length,
        'euphony-custom-labels': customLabels
      }
    },
    customLabels
  };
};

// ---------------------------------------------------------------------------
// Writer: Conversation -> ATIF
// ---------------------------------------------------------------------------

/**
 * Recover the original ATIF trajectory from a Conversation produced by
 * `parseAtifTrajectory`. Returns null if the conversation was not loaded from
 * ATIF.
 *
 * UX tradeoff (read-only by design):
 *   This function returns the ORIGINAL parsed document, not a re-serialization
 *   of the in-memory Conversation. That means user edits made through the
 *   conversation viewer are not reflected here. The tradeoff is intentional
 *   and works because `<euphony-atif>` is read-only end-to-end:
 *
 *     - The component does not declare an `is-editable` property and the app
 *       does not pass `?is-editable` to it (see `app.ts`, ATIF render
 *       branch). The inner `<euphony-conversation>` therefore defaults to
 *       `isEditable=false`, so the editing UI never appears.
 *     - `?disable-editing-mode-save-button=${true}` is set as a
 *       belt-and-suspenders fallback in case someone wires editing in later
 *       without realizing the writer can't reflect those edits.
 *
 *   Synthesizing ATIF from arbitrary Harmony conversations is *not* a
 *   reasonable alternative: the schema mandates sequential step IDs, an
 *   `agent.name`/`agent.version` pair, and tool-call → observation
 *   `source_call_id` references, none of which a generic Harmony
 *   conversation reliably preserves. If we ever need lossy export of an
 *   edited trajectory we should build a separate writer that does that
 *   reconstruction explicitly, rather than silently dropping edits here.
 */
export const serializeAtifTrajectory = (
  conversation: Conversation
): AtifTrajectory | null => {
  const stored = isRecord(conversation.metadata)
    ? conversation.metadata.atif_trajectory
    : null;
  if (!isRecord(stored)) return null;
  // Defensive deep clone so callers can't mutate our internal copy.
  return JSON.parse(JSON.stringify(stored)) as AtifTrajectory;
};

/** Serialize a Conversation back to a pretty-printed ATIF JSON string. */
export const toAtifJSONString = (conversation: Conversation): string | null => {
  const trajectory = serializeAtifTrajectory(conversation);
  return trajectory ? JSON.stringify(trajectory, null, 2) : null;
};

/**
 * Lightweight runtime validator that mirrors the Pydantic models. Returns an
 * array of human-readable error strings; an empty array means the document
 * is valid. Used by the parser-level smoke tests and by callers that want to
 * surface format errors to the user.
 */
export const validateAtifTrajectory = (raw: unknown): string[] => {
  const errors: string[] = [];
  const trajectory = unwrapTrajectory(raw);
  if (!trajectory) {
    errors.push('Top-level value is not a JSON object');
    return errors;
  }

  if (!isAtifVersion(trajectory.schema_version)) {
    errors.push('schema_version is missing or not an ATIF-* string');
  } else if (
    !(SUPPORTED_ATIF_SCHEMA_VERSIONS as readonly string[]).includes(
      trajectory.schema_version
    )
  ) {
    errors.push(
      `schema_version "${trajectory.schema_version}" is not in the list of ` +
        'supported versions ' +
        `(${SUPPORTED_ATIF_SCHEMA_VERSIONS.join(', ')})`
    );
  }

  if (typeof trajectory.session_id !== 'string') {
    errors.push('session_id is missing or not a string');
  }

  const agent = trajectory.agent as unknown;
  if (!isRecord(agent)) {
    errors.push('agent is missing or not an object');
  } else {
    if (typeof agent.name !== 'string') errors.push('agent.name is required');
    if (typeof agent.version !== 'string')
      errors.push('agent.version is required');
    // tool_definitions: harbor's Pydantic model only types this as
    // `list[dict[str, Any]]` (agent.py), but the RFC § AgentSchema spells
    // out the per-element shape: "Each element follows OpenAI's function
    // calling schema with `type` and `function` fields containing the
    // tool's signature and docs." We enforce that here so a mis-shaped
    // tool list is caught at validation time rather than at render time.
    if (agent.tool_definitions != null) {
      if (!Array.isArray(agent.tool_definitions)) {
        errors.push('agent.tool_definitions must be an array when provided');
      } else {
        for (const [i, toolDef] of agent.tool_definitions.entries()) {
          if (!isRecord(toolDef)) {
            errors.push(`agent.tool_definitions[${i}] must be an object`);
            continue;
          }
          if (toolDef.type !== 'function') {
            errors.push(`agent.tool_definitions[${i}].type must be "function"`);
          }
          if (!isRecord(toolDef.function)) {
            errors.push(`agent.tool_definitions[${i}].function is required`);
          }
        }
      }
    }
  }

  if (!Array.isArray(trajectory.steps) || trajectory.steps.length === 0) {
    errors.push('steps must be a non-empty array');
    return errors;
  }

  // ContentPart constraints mirror harbor's Pydantic model:
  //   harbor/src/harbor/models/trajectories/content.py
  //   - `type='text'` REQUIRES `text` and forbids `source`
  //   - `type='image'` REQUIRES `source` and forbids `text`
  //   - `source.media_type` is a Literal restricted to four MIME types
  // We enforce all three so a document that fails harbor's server-side
  // validation also fails ours, instead of looking valid here and then
  // breaking downstream.
  const ALLOWED_IMAGE_MEDIA_TYPES = new Set([
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp'
  ]);
  const validateContentParts = (path: string, value: unknown) => {
    if (!Array.isArray(value)) return;
    for (const [i, part] of value.entries()) {
      if (!isRecord(part)) {
        errors.push(`${path}[${i}] must be an object`);
        continue;
      }
      if (part.type === 'text') {
        if (typeof part.text !== 'string') {
          errors.push(`${path}[${i}] text part requires string 'text'`);
        }
        if (part.source !== undefined) {
          errors.push(
            `${path}[${i}] 'source' is not allowed when type is "text"`
          );
        }
        continue;
      }
      if (part.type === 'image') {
        if (part.text !== undefined) {
          errors.push(
            `${path}[${i}] 'text' is not allowed when type is "image"`
          );
        }
        if (
          !isRecord(part.source) ||
          typeof part.source.path !== 'string' ||
          typeof part.source.media_type !== 'string'
        ) {
          errors.push(
            `${path}[${i}] image part requires source.path and source.media_type`
          );
          continue;
        }
        if (!ALLOWED_IMAGE_MEDIA_TYPES.has(part.source.media_type)) {
          errors.push(
            `${path}[${i}].source.media_type "${part.source.media_type}" ` +
              'is not one of image/jpeg, image/png, image/gif, image/webp'
          );
        }
        continue;
      }
      errors.push(`${path}[${i}].type must be "text" or "image"`);
    }
  };

  const agentOnlyFields = [
    'model_name',
    'reasoning_effort',
    'reasoning_content',
    'tool_calls',
    'metrics'
  ] as const;

  // step_ids must be sequential starting at 1 (per Trajectory.validate_step_ids).
  for (let i = 0; i < trajectory.steps.length; i += 1) {
    const step = trajectory.steps[i] as unknown;
    if (!isRecord(step)) {
      errors.push(`steps[${i}] is not an object`);
      continue;
    }
    if (step.step_id !== i + 1) {
      errors.push(
        `steps[${i}].step_id: expected ${i + 1}, got ${String(step.step_id)}`
      );
    }
    if (
      step.source !== 'system' &&
      step.source !== 'user' &&
      step.source !== 'agent'
    ) {
      errors.push(`steps[${i}].source must be system|user|agent`);
    }
    if (typeof step.message !== 'string' && !Array.isArray(step.message)) {
      errors.push(`steps[${i}].message is required`);
    }
    if (step.timestamp != null && !isStrictIso8601(step.timestamp)) {
      errors.push(
        `steps[${i}].timestamp must be a strict ISO-8601 string when provided`
      );
    }
    validateContentParts(`steps[${i}].message`, step.message);

    if (step.source !== 'agent') {
      for (const field of agentOnlyFields) {
        if (step[field] != null) {
          errors.push(
            `steps[${i}].${field} is an agent-only field (source is ${String(step.source)})`
          );
        }
      }
    }

    // observation source_call_id must reference a tool_call_id on the same
    // step (per validate_tool_call_references).
    const obs = step.observation;
    if (!isRecord(obs) || !Array.isArray(obs.results)) continue;
    const toolCalls = Array.isArray(step.tool_calls) ? step.tool_calls : [];
    const validIds = new Set(
      (toolCalls as { tool_call_id?: unknown }[])
        .map(tc => tc.tool_call_id)
        .filter((id): id is string => typeof id === 'string')
    );
    for (const [j, result] of (
      obs.results as { source_call_id?: unknown }[]
    ).entries()) {
      const resultRecord = result as unknown;
      if (isRecord(resultRecord) && resultRecord.content != null) {
        validateContentParts(
          `steps[${i}].observation.results[${j}].content`,
          resultRecord.content
        );
      }
      const sid = result.source_call_id;
      if (typeof sid === 'string' && !validIds.has(sid)) {
        errors.push(
          `steps[${i}].observation.results[${j}].source_call_id ` +
            `'${sid}' does not reference any tool_call_id in the same step`
        );
      }
    }
  }

  return errors;
};
