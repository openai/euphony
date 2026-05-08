import type { Conversation } from '../../types/harmony-types';
import { parseLocalData } from './local-data-parser';

export type LocalDataWorkerMessage =
  | {
      command: 'startParseData';
      payload: {
        requestID: number;
        sourceName: 'clipboard' | 'file';
        sourceText?: string;
        sourceFile?: File;
      };
    }
  | {
      command: 'finishParseData';
      payload:
        | {
            requestID: number;
            sourceName: 'clipboard' | 'file';
            dataType: 'codex';
            codexSessionData: unknown[];
          }
        | {
            requestID: number;
            sourceName: 'clipboard' | 'file';
            dataType: 'atif';
            atifTrajectoryData: unknown[];
          }
        | {
            requestID: number;
            sourceName: 'clipboard' | 'file';
            dataType: 'conversation';
            conversationData: Conversation[];
          }
        | {
            requestID: number;
            sourceName: 'clipboard' | 'file';
            dataType: 'json';
            jsonData: Record<string, unknown>[];
          };
    }
  | {
      command: 'error';
      payload: {
        requestID: number;
        sourceName: 'clipboard' | 'file';
        message: string;
      };
    };

self.onmessage = async (e: MessageEvent<LocalDataWorkerMessage>) => {
  if (e.data.command !== 'startParseData') {
    console.error('Worker: unknown message', e.data.command);
    return;
  }

  const { requestID, sourceName, sourceText, sourceFile } = e.data.payload;

  try {
    const text = sourceText ?? (await sourceFile?.text());
    if (text === undefined) {
      throw new Error('No source text or file was provided.');
    }
    const result = parseLocalData(text);
    const message: LocalDataWorkerMessage = {
      command: 'finishParseData',
      payload: {
        requestID,
        sourceName,
        ...result
      }
    };
    postMessage(message);
  } catch (error) {
    const message: LocalDataWorkerMessage = {
      command: 'error',
      payload: {
        requestID,
        sourceName,
        message: error instanceof Error ? error.message : String(error)
      }
    };
    postMessage(message);
  }
};
