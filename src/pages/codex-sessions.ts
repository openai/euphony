import { css, html, LitElement, unsafeCSS } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { EUPHONY_API_URL } from '../utils/api-manager';

import '../components/pagination/pagination';
import componentCSS from './codex-sessions.css?inline';

interface CodexSessionSummary {
  relative_path: string;
  file_name: string;
  session_id: string | null;
  cwd: string | null;
  started_at: string | null;
  first_prompt: string | null;
  first_prompt_time: string | null;
  last_prompt: string | null;
  last_prompt_time: string | null;
  last_response: string | null;
  last_response_time: string | null;
}

interface CodexSessionListResponse {
  data: CodexSessionSummary[];
  offset: number;
  limit: number;
  total: number;
  matchedCount: number;
}

const DEFAULT_ITEMS_PER_PAGE = 25;
const ITEMS_PER_PAGE_OPTIONS = [10, 25, 50, 100];

@customElement('euphony-codex-sessions-page')
export class EuphonyCodexSessionsPage extends LitElement {
  @state()
  sessions: CodexSessionSummary[] = [];

  @state()
  totalSessions = 0;

  @state()
  matchedSessions = 0;

  @state()
  curPage = 1;

  @state()
  itemsPerPage = DEFAULT_ITEMS_PER_PAGE;

  @state()
  appliedFirstPromptFilter = '';

  @state()
  appliedLastPromptFilter = '';

  @state()
  firstPromptFilterInput = '';

  @state()
  lastPromptFilterInput = '';

  @state()
  isLoading = false;

  @state()
  errorMessage: string | null = null;

  constructor() {
    super();
    this.readStateFromURL();
  }

  connectedCallback(): void {
    super.connectedCallback();
    void this.loadSessions();
  }

  private get totalPageNum() {
    return Math.max(1, Math.ceil(this.matchedSessions / this.itemsPerPage));
  }

  private get offset() {
    return (this.curPage - 1) * this.itemsPerPage;
  }

  private readStateFromURL() {
    const params = new URLSearchParams(window.location.search);

    const pageParam = params.get('page');
    if (pageParam) {
      const parsedPage = parseInt(pageParam);
      if (!Number.isNaN(parsedPage)) {
        this.curPage = Math.max(1, parsedPage);
      }
    }

    const limitParam = params.get('limit');
    if (limitParam) {
      const parsedLimit = parseInt(limitParam);
      if (!Number.isNaN(parsedLimit)) {
        this.itemsPerPage = Math.max(1, parsedLimit);
      }
    }

    const firstPromptParam = params.get('firstPrompt') ?? '';
    const lastPromptParam = params.get('lastPrompt') ?? '';
    this.appliedFirstPromptFilter = firstPromptParam;
    this.appliedLastPromptFilter = lastPromptParam;
    this.firstPromptFilterInput = firstPromptParam;
    this.lastPromptFilterInput = lastPromptParam;
  }

  private updateURL() {
    const params = new URLSearchParams();
    params.set('page', this.curPage.toString());
    params.set('limit', this.itemsPerPage.toString());

    if (this.appliedFirstPromptFilter) {
      params.set('firstPrompt', this.appliedFirstPromptFilter);
    }

    if (this.appliedLastPromptFilter) {
      params.set('lastPrompt', this.appliedLastPromptFilter);
    }

    const query = params.toString();
    history.replaceState({}, '', query ? `?${query}` : window.location.pathname);
  }

  private async getErrorMessage(response: Response) {
    try {
      const payload = (await response.json()) as { detail?: string };
      if (typeof payload.detail === 'string' && payload.detail.trim() !== '') {
        return payload.detail;
      }
    } catch (_error) {
      // Fall through to the generic HTTP status when the backend returns HTML
      // or an unexpected payload shape.
    }

    return `Request failed with status ${response.status}.`;
  }

  private async loadSessions() {
    this.isLoading = true;
    this.errorMessage = null;

    try {
      const params = new URLSearchParams();
      params.set('offset', this.offset.toString());
      params.set('limit', this.itemsPerPage.toString());

      if (this.appliedFirstPromptFilter) {
        params.set('firstPromptFilter', this.appliedFirstPromptFilter);
      }
      if (this.appliedLastPromptFilter) {
        params.set('lastPromptFilter', this.appliedLastPromptFilter);
      }

      const response = await fetch(
        `${EUPHONY_API_URL}codex-sessions/?${params.toString()}`
      );
      if (!response.ok) {
        throw new Error(await this.getErrorMessage(response));
      }

      const payload = (await response.json()) as CodexSessionListResponse;

      // If a filter shrinks the result set enough that the requested page no
      // longer exists, jump to the last valid page and refetch once so the UI
      // always shows real results instead of an empty middle state.
      const totalPageNum = Math.max(
        1,
        Math.ceil(payload.matchedCount / this.itemsPerPage)
      );
      if (payload.matchedCount > 0 && this.curPage > totalPageNum) {
        this.curPage = totalPageNum;
        this.updateURL();
        await this.loadSessions();
        return;
      }

      this.sessions = payload.data;
      this.totalSessions = payload.total;
      this.matchedSessions = payload.matchedCount;
      this.updateURL();
    } catch (error) {
      this.sessions = [];
      this.totalSessions = 0;
      this.matchedSessions = 0;
      this.errorMessage = String(error);
    } finally {
      this.isLoading = false;
    }
  }

  private buildSessionFileURL(session: CodexSessionSummary) {
    const query = new URLSearchParams({
      relative_path: session.relative_path
    });
    return new URL(
      `${EUPHONY_API_URL}codex-session-file/?${query.toString()}`,
      window.location.href
    ).toString();
  }

  private buildViewerURL(session: CodexSessionSummary) {
    const viewerURL = new URL('./', window.location.href);
    viewerURL.searchParams.set('path', this.buildSessionFileURL(session));
    return viewerURL.toString();
  }

  private formatTimestamp(timestamp: string | null) {
    if (!timestamp) {
      return 'Not available';
    }

    const parsed = new Date(timestamp);
    if (Number.isNaN(parsed.getTime())) {
      return timestamp;
    }

    return new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short'
    }).format(parsed);
  }

  private renderTimestamp(timestamp: string | null) {
    return html`
      <time datetime=${timestamp ?? ''}>${this.formatTimestamp(timestamp)}</time>
    `;
  }

  private renderPromptSummary(
    label: string,
    timestamp: string | null,
    prompt: string | null,
    emptyState = 'No user prompt captured.'
  ) {
    return html`
      <section class="info-card prompt-card">
        <div class="card-label">${label}</div>
        <div class="card-time">${this.renderTimestamp(timestamp)}</div>
        <div class="prompt-text">${prompt ?? emptyState}</div>
      </section>
    `;
  }

  private applyFilters(event: Event) {
    event.preventDefault();
    this.appliedFirstPromptFilter = this.firstPromptFilterInput.trim();
    this.appliedLastPromptFilter = this.lastPromptFilterInput.trim();
    this.curPage = 1;
    void this.loadSessions();
  }

  private resetFilters() {
    this.firstPromptFilterInput = '';
    this.lastPromptFilterInput = '';
    this.appliedFirstPromptFilter = '';
    this.appliedLastPromptFilter = '';
    this.curPage = 1;
    void this.loadSessions();
  }

  private pageClicked(event: CustomEvent<number>) {
    this.curPage = event.detail;
    void this.loadSessions();
  }

  private itemsPerPageChanged(event: CustomEvent<number>) {
    this.itemsPerPage = event.detail;
    this.curPage = 1;
    void this.loadSessions();
  }

  render() {
    return html`
      <div class="page-shell">
        <div class="page-content">
          <header class="hero">
            <div class="hero-copy">
              <p class="eyebrow">Local Codex Activity</p>
              <h1>Codex Sessions</h1>
              <p class="hero-text">
                Browse JSONL session logs from your local Codex history, filter
                by the first or last user prompt, and open any session in the
                main Euphony viewer.
              </p>
            </div>

            <div class="hero-actions">
              <a class="nav-link primary-link" href="./">Open Euphony Viewer</a>
              <p class="hero-note">
                Source: <code>~/.codex/sessions</code> or <code>CODEX_HOME</code>
                when it is configured.
              </p>
            </div>
          </header>

          <section class="toolbar-card">
            <form class="filters" @submit=${(event: Event) => this.applyFilters(event)}>
              <label class="filter-field">
                <span>Filter first prompt</span>
                <input
                  .value=${this.firstPromptFilterInput}
                  placeholder="e.g. bug fix"
                  @input=${(event: InputEvent) => {
                    const target = event.target as HTMLInputElement;
                    this.firstPromptFilterInput = target.value;
                  }}
                />
              </label>

              <label class="filter-field">
                <span>Filter last prompt</span>
                <input
                  .value=${this.lastPromptFilterInput}
                  placeholder="e.g. release notes"
                  @input=${(event: InputEvent) => {
                    const target = event.target as HTMLInputElement;
                    this.lastPromptFilterInput = target.value;
                  }}
                />
              </label>

              <div class="filter-actions">
                <button class="action-button primary-button" type="submit">
                  Apply filters
                </button>
                <button
                  class="action-button secondary-button"
                  type="button"
                  @click=${() => this.resetFilters()}
                >
                  Reset
                </button>
              </div>
            </form>

            <div class="results-row">
              <div class="results-copy">
                ${this.matchedSessions.toLocaleString()} matched session${this
                  .matchedSessions === 1
                  ? ''
                  : 's'}
                ${this.totalSessions !== this.matchedSessions
                  ? html`<span class="results-muted">
                      of ${this.totalSessions.toLocaleString()} total
                    </span>`
                  : ''}
              </div>
            </div>
          </section>

          ${this.isLoading
            ? html`
                <section class="status-card">
                  <div class="status-title">Loading sessions</div>
                  <div class="status-body">
                    Reading Codex session metadata from your local machine.
                  </div>
                </section>
              `
            : ''}

          ${!this.isLoading && this.errorMessage
            ? html`
                <section class="status-card error-card">
                  <div class="status-title">Could not load sessions</div>
                  <div class="status-body">${this.errorMessage}</div>
                </section>
              `
            : ''}

          ${!this.isLoading &&
          !this.errorMessage &&
          this.matchedSessions === 0
            ? html`
                <section class="status-card">
                  <div class="status-title">No matching sessions</div>
                  <div class="status-body">
                    Try clearing the prompt filters or create a new Codex
                    session and reload this page.
                  </div>
                </section>
              `
            : ''}

          ${!this.isLoading &&
          !this.errorMessage &&
          this.matchedSessions > 0
            ? html`
                <section class="session-list">
                  ${this.sessions.map(
                    session => html`
                      <article class="session-card">
                        <div class="session-header">
                          <div class="session-title-block">
                            <a
                              class="session-link"
                              href=${this.buildViewerURL(session)}
                            >
                              ${session.session_id ?? session.file_name}
                            </a>
                            <div class="session-subtitle">
                              <span>${session.relative_path}</span>
                              ${session.cwd
                                ? html`<span class="subtitle-separator">|</span>
                                    <span>${session.cwd}</span>`
                                : ''}
                            </div>
                          </div>

                          <div class="session-meta">
                            <div class="meta-label">Started</div>
                            <div class="meta-value">
                              ${this.renderTimestamp(session.started_at)}
                            </div>
                          </div>
                        </div>

                        <div class="info-grid">
                          ${this.renderPromptSummary(
                            'First user prompt',
                            session.first_prompt_time,
                            session.first_prompt
                          )}

                          ${this.renderPromptSummary(
                            'Last user prompt',
                            session.last_prompt_time,
                            session.last_prompt
                          )}

                          ${this.renderPromptSummary(
                            'Last response',
                            session.last_response_time,
                            session.last_response,
                            'No assistant response captured.'
                          )}
                        </div>
                      </article>
                    `
                  )}
                </section>
              `
            : ''}

          ${!this.isLoading && !this.errorMessage && this.matchedSessions > 0
            ? html`
                <footer class="pagination-row">
                  <nightjar-pagination
                    .curPage=${this.curPage}
                    .totalPageNum=${this.totalPageNum}
                    .itemsPerPage=${this.itemsPerPage}
                    .itemsPerPageOptions=${ITEMS_PER_PAGE_OPTIONS}
                    @page-clicked=${(event: CustomEvent<number>) =>
                      this.pageClicked(event)}
                    @items-per-page-changed=${(event: CustomEvent<number>) =>
                      this.itemsPerPageChanged(event)}
                  ></nightjar-pagination>
                </footer>
              `
            : ''}
        </div>
      </div>
    `;
  }

  static styles = [
    css`
      ${unsafeCSS(componentCSS)}
    `
  ];
}

declare global {
  interface HTMLElementTagNameMap {
    'euphony-codex-sessions-page': EuphonyCodexSessionsPage;
  }
}
