import { Semaphore } from 'async-mutex';
import type {
  HarmonyRenderRequest,
  MessageSharingRequest,
  RefreshRendererListRequest,
  TranslateResponse,
  TranslationRequest
} from '../../types/common-types';
import { APIManager, BrowserAPIManager } from '../../utils/api-manager';
import { URLManager } from './url-manager';

const MAX_CONCURRENT_TRANSLATIONS = 128;

const stringifyRequestError = (error: unknown): string => {
  if (error instanceof Error) {
    return JSON.stringify({
      name: error.name,
      message: error.message,
      stack: error.stack
    });
  }
  if (typeof error === 'string') {
    return error;
  }
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
};

/**
 * A class that handles requests from the euphony-conversation component.
 */
export class RequestWorker {
  apiBaseURL: string;
  apiManager: APIManager;
  browserAPIManager: BrowserAPIManager;

  constructor(apiBaseURL: string) {
    this.apiBaseURL = apiBaseURL;
    this.apiManager = new APIManager(apiBaseURL);
    this.browserAPIManager = new BrowserAPIManager();
  }

  // Handles a translation request with max concurrency control.
  static translationSemaphore = new Semaphore(MAX_CONCURRENT_TRANSLATIONS);

  /**
   * Handles a translation request with max concurrency control using translationSemaphore.
   * @param e The custom event containing the translation request details
   */
  async translationRequestHandler(e: CustomEvent<TranslationRequest>) {
    const { text, resolve, reject } = e.detail;
    const requestOptions: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        source: text
      }),
      credentials: 'include'
    };

    // Helper function to delay execution
    const delay = (ms: number) => new Promise(res => setTimeout(res, ms));

    // Use the translationSemaphore to control concurrency
    const [value, release] = await RequestWorker.translationSemaphore.acquire();
    try {
      // Maximum number of retries
      const maxRetries = 3;
      // Add jitter: random delay between 300ms and 700ms
      const retryDelay = 300 + Math.floor(Math.random() * (700 - 300 + 1));
      let attempt = 0;
      let lastError = null;

      while (attempt < maxRetries) {
        try {
          const response = await fetch(
            `${this.apiBaseURL}translate/`,
            requestOptions
          );
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const result = (await response.json()) as TranslateResponse;
          resolve(result);
          lastError = null;
          break;
        } catch (err) {
          lastError = err;
          attempt++;
          if (attempt < maxRetries) {
            await delay(retryDelay);
          }
        }
      }
      if (lastError) {
        // If all retries have failed, return the original text
        const dummyTranslateResponse: TranslateResponse = {
          translation: text,
          is_translated: false,
          language: 'Failed',
          has_command: false
        };
        console.error('Translation failed, returning original text', lastError);
        resolve(dummyTranslateResponse);
      }
    } catch (error) {
      // If all retries have failed, return the original text
      const dummyTranslateResponse: TranslateResponse = {
        translation: text,
        is_translated: false,
        language: 'Failed',
        has_command: false
      };
      console.error('Translation failed, returning original text', error);
      resolve(dummyTranslateResponse);
    } finally {
      // Always release the semaphore after the request is done
      release();
    }
  }

  /**
   * Handles a translation request using the OpenAI API directly from browser.
   * @param e The custom event containing the translation request details
   */
  async frontendOnlyTranslationRequestHandler(
    e: CustomEvent<TranslationRequest>,
    apiKey: string
  ) {
    const { text, resolve, reject } = e.detail;
    try {
      const result = await this.browserAPIManager.translateTextWithOpenAI(
        text,
        apiKey
      );
      resolve(result);
    } catch (error) {
      reject(stringifyRequestError(error));
    }
  }

  /**
   * Handles a fetch message sharing URL request.
   * @param e The custom event containing the fetch message sharing URL request details
   */
  fetchMessageSharingURLRequestHandler(
    e: CustomEvent<MessageSharingRequest>,
    conversationIndex: number,
    urlManager: URLManager,
    blobPath: string | null
  ) {
    const { messageIndex, resolve, reject } = e.detail;
    const url = urlManager.getMessageShareURL(
      conversationIndex,
      messageIndex,
      blobPath
    );
    resolve(url);
  }

  /**
   * Handles a renderer refresh request by calling API
   */
  async refreshRendererListRequestHandler(
    e: CustomEvent<RefreshRendererListRequest>
  ) {
    const { resolve, reject } = e.detail;

    try {
      const result = await this.apiManager.refreshRendererList();
      resolve(result);
    } catch (error) {
      reject(stringifyRequestError(error));
    }
  }

  /**
   * Handles a renderer refresh request in frontend-only mode.
   */
  async frontendOnlyRefreshRendererListRequestHandler(
    e: CustomEvent<RefreshRendererListRequest>
  ) {
    const { resolve, reject } = e.detail;

    try {
      const result = await this.browserAPIManager.refreshRendererList();
      resolve(result);
    } catch (error) {
      reject(stringifyRequestError(error));
    }
  }

  /**
   * Handles a harmony render request by calling API
   */
  async harmonyRenderRequestHandler(e: CustomEvent<HarmonyRenderRequest>) {
    const { conversation, renderer, resolve, reject } = e.detail;

    try {
      const result = await this.apiManager.harmonyRender(
        conversation,
        renderer
      );
      resolve(result);
    } catch (error) {
      reject(stringifyRequestError(error));
    }
  }

  /**
   * Handles a harmony render request in frontend-only mode.
   */
  async frontendOnlyHarmonyRenderRequestHandler(
    e: CustomEvent<HarmonyRenderRequest>
  ) {
    const { conversation, renderer, resolve, reject } = e.detail;

    try {
      const result = await this.browserAPIManager.harmonyRender(
        conversation,
        renderer
      );
      resolve(result);
    } catch (error) {
      reject(stringifyRequestError(error));
    }
  }
}
