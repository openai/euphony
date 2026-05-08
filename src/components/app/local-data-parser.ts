// Pure parsing helpers for the local-data worker. Extracted from
// `local-data-worker.ts` so the dispatch logic (ATIF / Codex / conversation /
// JSON) is testable without spinning up a Web Worker (`self.onmessage`).

import type { Conversation } from '../../types/harmony-types';
import { isAtifTrajectory } from '../../utils/atif-trajectory';
import { isCodexSessionJSONL } from '../../utils/codex-session';

export type ParsedItem = Record<string, unknown> | string | Conversation;

export type ParseLocalDataResult =
  | { dataType: 'codex'; codexSessionData: unknown[] }
  | { dataType: 'atif'; atifTrajectoryData: unknown[] }
  | { dataType: 'conversation'; conversationData: Conversation[] }
  | { dataType: 'json'; jsonData: Record<string, unknown>[] };

function isConversation(data: unknown): boolean {
  if (typeof data !== 'object' || data === null) {
    return false;
  }
  return 'messages' in data && Array.isArray(data.messages);
}

function extractConversationFromJSONL(
  data: unknown[]
): Conversation[] | null {
  let curData: Record<string, Conversation | string>[] | null = null;

  if (
    data.length > 0 &&
    typeof data[0] === 'object' &&
    !isConversation(data[0])
  ) {
    curData = data as Record<string, Conversation | string>[];
  }

  if (data.length > 0 && typeof data[0] === 'string') {
    let shouldSkipTransformation = false;
    try {
      const conversation = JSON.parse(data[0]) as Conversation;
      if (isConversation(conversation)) {
        shouldSkipTransformation = true;
      }
    } catch (_error) {
      shouldSkipTransformation = true;
    }

    if (!shouldSkipTransformation) {
      curData = [];
      for (const d of data) {
        const record = JSON.parse(d as string) as Record<
          string,
          Conversation | string
        >;
        curData.push(record);
      }
    }
  }

  if (curData !== null) {
    let conversationKey: string | null = null;
    let conversationFieldIsString = false;

    for (const key in curData[0]) {
      if (typeof curData[0][key] === 'string') {
        try {
          const conversation = JSON.parse(curData[0][key]) as Conversation;
          if (isConversation(conversation)) {
            conversationKey = key;
            conversationFieldIsString = true;
            break;
          }
        } catch (_error) {
          continue;
        }
      } else if (isConversation(curData[0][key])) {
        conversationKey = key;
        break;
      }
    }

    if (conversationKey !== null) {
      const conversationData: Conversation[] = [];

      for (const d of curData) {
        const conversation = conversationFieldIsString
          ? (JSON.parse(d[conversationKey] as string) as Conversation)
          : (d[conversationKey] as Conversation);
        conversation.metadata ??= {};

        for (const k in d) {
          if (k !== conversationKey) {
            conversation.metadata[`euphonyTransformed-${k}`] = d[k];
          }
        }

        conversationData.push(conversation);
      }
      return conversationData;
    }
  }

  return null;
}

/**
 * Validates that the parsed payload is an array of `Conversation`s and
 * back-fills `id` from `conversation_id` where missing.
 *
 * Note: this function mutates `conversations` in place (rewriting each entry
 * with a normalized form) — preserved verbatim from the original
 * `local-data-worker` implementation it was extracted from. Future improvement:
 * return a new array so the input can be treated as read-only and the type
 * predicate becomes a true narrowing of unmodified data.
 */
function validateAndTransformConversations(
  conversations: ParsedItem[]
): conversations is Conversation[] {
  const allValid: boolean[] = [];

  for (const [i, conversation] of conversations.entries()) {
    if (typeof conversation === 'string') {
      const conversationData = JSON.parse(conversation) as Record<
        string,
        unknown
      >;
      let newItem = conversation;

      if (
        conversationData.conversation_id !== undefined &&
        conversationData.id === undefined
      ) {
        conversationData.id = conversationData.conversation_id;
        newItem = JSON.stringify(conversationData);
      }

      conversations[i] = newItem;
      allValid.push(Array.isArray(conversationData.messages));
    } else {
      const conversationData = conversation as Record<string, unknown>;

      if (
        conversationData.conversation_id !== undefined &&
        conversationData.id === undefined
      ) {
        conversationData.id = conversationData.conversation_id;
      }

      conversations[i] = conversationData;
      allValid.push(Array.isArray(conversationData.messages));
    }
  }

  return allValid.every(Boolean);
}

export function parseSourceText(sourceText: string): ParsedItem[] {
  const allData: ParsedItem[] = [];

  try {
    const jsonData = JSON.parse(sourceText) as Record<string, unknown>;
    allData.push(jsonData);
    return allData;
  } catch (_error) {
    for (const line of sourceText.split('\n')) {
      try {
        allData.push(JSON.parse(line) as Record<string, unknown> | string);
      } catch (_innerError) {
        // Skip invalid JSONL lines.
      }
    }
  }

  return allData;
}

// Dispatch order matters: ATIF must be detected before Codex, since an ATIF
// document with an event-shaped extra field could otherwise be misclassified
// as Codex JSONL.
export function parseLocalData(sourceText: string): ParseLocalDataResult {
  let allData = parseSourceText(sourceText);

  if (allData.length === 0) {
    throw new Error('Failed to read any JSON or JSONL data.');
  }

  if (isAtifTrajectory(allData as unknown[])) {
    return { dataType: 'atif', atifTrajectoryData: allData as unknown[] };
  }

  if (isCodexSessionJSONL(allData as unknown[])) {
    return {
      dataType: 'codex',
      codexSessionData: allData as unknown[]
    };
  }

  const transformedConversationData = extractConversationFromJSONL(
    allData as unknown[]
  );
  if (transformedConversationData) {
    allData = transformedConversationData;
  }

  if (!validateAndTransformConversations(allData)) {
    return {
      dataType: 'json',
      jsonData: allData as Record<string, unknown>[]
    };
  }

  const conversationData: Conversation[] = [];
  for (const item of allData) {
    if (typeof item === 'string') {
      conversationData.push(JSON.parse(item) as Conversation);
    } else {
      conversationData.push(item);
    }
  }

  return {
    dataType: 'conversation',
    conversationData
  };
}
