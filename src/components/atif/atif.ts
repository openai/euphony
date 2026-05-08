/**
 * <euphony-atif> — viewer for Harbor's ATIF (Agent Trajectory Interchange
 * Format) trajectories. Mirrors `<euphony-codex>`: parses the input via
 * `parseAtifTrajectory` and delegates rendering to `<euphony-conversation>`.
 */

import { css, html, LitElement, PropertyValues, unsafeCSS } from 'lit';
import { customElement, property, query, state } from 'lit/decorators.js';
import { ifDefined } from 'lit/directives/if-defined.js';
import type { Conversation } from '../../types/harmony-types';
import { parseAtifTrajectory } from '../../utils/atif-trajectory';
import type { EuphonyConversation } from '../conversation/conversation';
import type {
  FocusModeSettings,
  MessageLabelSettings
} from '../preference-window/preference-window';

import '../conversation/conversation';
import componentCSS from './atif.css?inline';

@customElement('euphony-atif')
export class EuphonyAtif extends LitElement {
  /** Raw ATIF trajectory as a JSON string. Used for declarative HTML usage. */
  @property({ type: String, attribute: 'trajectory-string' })
  trajectoryString = '';

  /**
   * Parsed ATIF trajectory data, matching the shape of `<euphony-codex>`'s
   * `sessionData` for parity. The local-data worker hands us a one-element
   * array wrapping the trajectory document; `refreshConversationFromTrajectory`
   * unwraps it before parsing.
   *
   * Future improvement: this could be tightened to
   * `[AtifTrajectory] | null` (or a dedicated runtime type) so call-site
   * type-checking catches malformed data instead of relying on
   * `parseAtifTrajectory` to no-op. Kept loose here to mirror the codex
   * component's public API exactly.
   */
  @property({ attribute: false })
  trajectoryData: unknown[] | null = null;

  @property({ type: String, attribute: 'sharing-url' })
  sharingURL: string | null = null;

  @property({ type: String, attribute: 'conversation-label' })
  conversationLabel = 'Trajectory';

  @property({ type: String, attribute: 'conversation-max-width' })
  conversationMaxWidth: string | null = null;

  @property({ type: String, attribute: 'conversation-style' })
  conversationStyle = '';

  @property({ type: Boolean, attribute: 'should-render-markdown' })
  shouldRenderMarkdown = false;

  @property({ type: Boolean, attribute: 'is-showing-metadata' })
  isShowingMetadata = false;

  @property({ type: Array, attribute: 'focus-mode-author' })
  focusModeAuthor: string[] = [];

  @property({ type: Array, attribute: 'focus-mode-recipient' })
  focusModeRecipient: string[] = [];

  @property({ type: Array, attribute: 'focus-mode-content-type' })
  focusModeContentType: string[] = [];

  @property({ type: Boolean, attribute: 'disable-markdown-button' })
  disableMarkdownButton = false;

  @property({ type: Boolean, attribute: 'disable-translation-button' })
  disableTranslationButton = false;

  @property({ type: Boolean, attribute: 'disable-share-button' })
  disableShareButton = false;

  @property({ type: Boolean, attribute: 'disable-metadata-button' })
  disableMetadataButton = false;

  @property({ type: Boolean, attribute: 'disable-message-metadata' })
  disableMessageMetadata = false;

  @property({ type: Boolean, attribute: 'disable-conversation-name' })
  disableConversationName = false;

  @property({ type: Boolean, attribute: 'disable-preference-button' })
  disablePreferenceButton = false;

  @property({ type: Boolean, attribute: 'disable-image-preview-window' })
  disableImagePreviewWindow = false;

  @property({ type: Boolean, attribute: 'disable-token-window' })
  disableTokenWindow = false;

  @property({ type: Boolean, attribute: 'disable-editing-mode-save-button' })
  disableEditingModeSaveButton = false;

  @property({ type: Boolean, attribute: 'disable-conversation-id-copy-button' })
  disableConversationIDCopyButton = false;

  @property({
    type: String,
    attribute: 'disable-download-convo-button-tooltip'
  })
  disableDownloadConvoButtonTooltip = '';

  @property({ type: String, attribute: 'disable-copy-convo-button-tooltip' })
  disableCopyConvoButtonTooltip = '';

  @property({ type: String, attribute: 'theme' })
  theme: 'auto' | 'light' | 'dark' = 'light';

  @state()
  conversation: Conversation | null = null;

  @state()
  parseError: string | null = null;

  @query('euphony-conversation')
  conversationComponent: EuphonyConversation | undefined;

  private refreshConversationFromTrajectory() {
    // Defensive `Array.isArray` mirrors `<euphony-codex>`'s `sessionData`
    // handling — runtime callers (declarative HTML, ad-hoc JS) may bypass the
    // typed property and pass non-array values, so we re-check at runtime.
    let raw: unknown = null;
    if (Array.isArray(this.trajectoryData) && this.trajectoryData.length > 0) {
      raw = this.trajectoryData[0] ?? null;
    } else if (this.trajectoryString !== '') {
      try {
        raw = JSON.parse(this.trajectoryString);
      } catch (_error) {
        raw = null;
      }
    }

    if (raw === null) {
      this.conversation = null;
      this.parseError = 'No ATIF trajectory data found.';
      return;
    }

    const parseResult = parseAtifTrajectory(raw);
    if (!parseResult) {
      this.conversation = null;
      this.parseError = 'Unsupported or malformed ATIF trajectory.';
      return;
    }

    this.conversation = parseResult.conversation;
    this.parseError = null;
  }

  willUpdate(changedProperties: PropertyValues<this>) {
    if (
      changedProperties.has('trajectoryString') ||
      changedProperties.has('trajectoryData')
    ) {
      this.refreshConversationFromTrajectory();
    }
  }

  render() {
    if (!this.conversation) {
      return html`
        <div class="empty-state">
          ${this.parseError ?? 'No ATIF trajectory to display.'}
        </div>
      `;
    }

    return html`
      <div class="atif-wrapper">
        <euphony-conversation
          .conversationData=${this.conversation}
          sharing-url=${ifDefined(this.sharingURL ?? undefined)}
          conversation-label=${this.conversationLabel}
          conversation-max-width=${ifDefined(
            this.conversationMaxWidth ?? undefined
          )}
          ?should-render-markdown=${this.shouldRenderMarkdown}
          ?is-showing-metadata=${this.isShowingMetadata}
          .focusModeAuthor=${this.focusModeAuthor}
          .focusModeRecipient=${this.focusModeRecipient}
          .focusModeContentType=${this.focusModeContentType}
          ?disable-markdown-button=${this.disableMarkdownButton}
          ?disable-translation-button=${this.disableTranslationButton}
          ?disable-share-button=${this.disableShareButton}
          ?disable-metadata-button=${this.disableMetadataButton}
          ?disable-message-metadata=${this.disableMessageMetadata}
          ?disable-conversation-name=${this.disableConversationName}
          ?disable-preference-button=${this.disablePreferenceButton}
          ?disable-image-preview-window=${this.disableImagePreviewWindow}
          ?disable-token-window=${this.disableTokenWindow}
          ?disable-editing-mode-save-button=${this.disableEditingModeSaveButton}
          ?disable-conversation-id-copy-button=${this
            .disableConversationIDCopyButton}
          disable-download-convo-button-tooltip=${ifDefined(
            this.disableDownloadConvoButtonTooltip || undefined
          )}
          disable-copy-convo-button-tooltip=${ifDefined(
            this.disableCopyConvoButtonTooltip || undefined
          )}
          theme=${this.theme}
          style=${this.conversationStyle}
        ></euphony-conversation>
      </div>
    `;
  }

  static styles = [
    css`
      ${unsafeCSS(componentCSS)}
    `
  ];

  preferenceWindowMessageLabelChanged(e: CustomEvent<MessageLabelSettings>) {
    this.conversationComponent?.preferenceWindowMessageLabelChanged(e);
  }

  preferenceWindowFocusModeSettingsChanged(e: CustomEvent<FocusModeSettings>) {
    this.conversationComponent?.preferenceWindowFocusModeSettingsChanged(e);
  }

  expandBlockContents() {
    this.conversationComponent?.expandBlockContents();
  }

  collapseBlockContents() {
    this.conversationComponent?.collapseBlockContents();
  }

  translationButtonClicked() {
    void this.conversationComponent?.translationButtonClicked();
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'euphony-atif': EuphonyAtif;
  }
}
