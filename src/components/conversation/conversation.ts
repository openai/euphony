import { autoUpdate } from '@floating-ui/dom';
import { downloadText } from '@xiaohk/utils';
import {
  css,
  html,
  LitElement,
  PropertyValues,
  TemplateResult,
  unsafeCSS
} from 'lit';
import {
  customElement,
  property,
  query,
  queryAsync,
  state
} from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import type {
  MessageSharingRequest,
  TranslatableConversation,
  TranslatableMessage,
  TranslateResponse,
  TranslationCompletedEventDetail,
  TranslationRequest
} from '../../types/common-types';
import {
  Conversation,
  DeveloperContentMessageContent,
  getContentFromContentOrString,
  getContentTypeFromContent,
  Message,
  Role,
  SystemContentMessageContent,
  TextMessageContent,
  tryGetContentTypeFromContent
} from '../../types/harmony-types';
import {
  getCustomLabelsFromMagicMetadata,
  getDeferredPromise,
  sharedCollapseBlockContents,
  sharedExpandBlockContents,
  styleToString,
  updateFloatPosition,
  updatePopperOverlay
} from '../../utils/utils';
import {
  EuphonyFloatingToolbar,
  FloatingToolbarButton
} from '../floating-toolbar/floating-toolbar';
import '../preference-window/preference-window';
import { EuphonyTokenWindow } from '../token-window/token-window';

import '@shoelace-style/shoelace/dist/components/copy-button/copy-button.js';
import '@shoelace-style/shoelace/dist/components/split-panel/split-panel.js';
import '../floating-toolbar/floating-toolbar';
import '../json-viewer/json-viewer';
import '../message-code/message-code';
import '../message-developer-content/message-developer-content';
import type { DeveloperContentEditPayload } from '../message-developer-content/message-developer-content';
import '../message-editor-popover/message-editor-popover';
import type { MessageEditorUserSetData } from '../message-editor-popover/message-editor-popover';
import '../message-hidden/message-hidden';
import '../message-system-content/message-system-content';
import type { SystemContentEditPayload } from '../message-system-content/message-system-content';
import '../message-text/message-text';
import '../message-unsupported/message-unsupported';
import '../token-window/token-window';

import shoelaceCSS from '@shoelace-style/shoelace/dist/themes/light.css?inline';
import iconArrow from '../../images/icon-arrow-right.svg?raw';
import iconAssistant from '../../images/icon-assistant.svg?raw';
import iconChevron from '../../images/icon-chevron-up-lg.svg?raw';
import iconCopy from '../../images/icon-copy.svg?raw';
import iconDownload from '../../images/icon-download.svg?raw';
import iconEyeOff from '../../images/icon-eye-off.svg?raw';
import iconEye from '../../images/icon-eye.svg?raw';
import iconLink from '../../images/icon-link.svg?raw';
import iconDeveloper from '../../images/icon-macbook.svg?raw';
import iconMetaData from '../../images/icon-marker-data.svg?raw';
import iconArrowRotate from '../../images/icon-paint.svg?raw';
import iconPen from '../../images/icon-pen-simple.svg?raw';
import iconPlusSquare from '../../images/icon-plus-square.svg?raw';
import iconSystem from '../../images/icon-settings.svg?raw';
import iconShare from '../../images/icon-share.svg?raw';
import iconTranslate from '../../images/icon-translate.svg?raw';
import iconTrash from '../../images/icon-trash.svg?raw';
import iconUser from '../../images/icon-user.svg?raw';
import iconTool from '../../images/icon-wrench.svg?raw';

import {
  FocusModeSettings,
  MessageLabelSettings
} from '../preference-window/preference-window';
import componentCSS from './conversation.css?inline';

const METADATA_MIN_HEIGHT = 500;
const TRANSLATION_TIMEOUT = 600000;
const MAX_MESSAGE_COUNT = 8000;
type CustomMessageLabel =
  | [number | string, string]
  | [number | string, string, string]
  | [number | string, string, string, string];

interface TranslateResponseWithPartIndex extends TranslateResponse {
  partIndex: number;
  partContentType: 'string';
}

/**
 * Conversation element.
 */
@customElement('euphony-conversation')
export class EuphonyConversation extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @property({ type: String, attribute: 'conversation-string' })
  conversationString = '';

  @property({ attribute: false })
  conversationData: TranslatableConversation | null = null;

  /**
   * The URL for the sharing button. If it's null, the copy URL button will be
   * hidden.
   */
  @property({ type: String, attribute: 'sharing-url' })
  sharingURL: string | null = null;

  /**
   * Optional URL to the current json/jsonl file (if any). This is used to
   * resolve some relative paths in the asset pointers in the conversation. For
   * example, `aquifer://foo` will be resolved to `dataFileURL/../assets/foo`.
   * We won't resolve relative paths if `dataFileURL` is not provided.
   */
  @property({ type: String, attribute: 'data-file-url' })
  dataFileURL: string | null = null;

  /**
   * This overrides the JSON content when user clicks copy JSON or download
   * JSON. It should never be set unless the content you want users to copy is
   * different form the Conversation data itself (e.g., we use it for
   * Comparison).
   */
  @property({ type: String, attribute: 'override-sharing-json-string' })
  overrideSharingJSONString: string | null = null;

  @property({ type: Boolean, attribute: 'should-render-markdown' })
  shouldRenderMarkdown = false;

  /**
   * We use DOMPurify to sanitize the markdown rendered HTML before displaying
   * it. This property is used to pass DOMPurify's allowed tags. If this is not
   * provided, we use the default allowed tags defined in dompurify-configs.ts
   */
  @property({ type: Array, attribute: 'markdown-allowed-tags' })
  markdownAllowedTags: string[] | null = null;

  /**
   * We use DOMPurify to sanitize the markdown rendered HTML before displaying
   * it. This property is used to pass DOMPurify's allowed attributes. If this
   * is not provided, we use the default allowed attributes defined in
   * dompurify-configs.ts
   */
  @property({ type: Array, attribute: 'markdown-allowed-attributes' })
  markdownAllowedAttributes: string[] | null = null;

  /**
   * The label shown before the conversation ID.
   */
  @property({ type: String, attribute: 'conversation-label' })
  conversationLabel = 'Conversation';

  @state()
  conversation: TranslatableConversation | null = null;

  // Editable settings
  @property({ type: Boolean, attribute: 'is-editable' })
  isEditable = false;

  /**
   * Focus mode settings.
   * Each of the three properties is an independent filter. If it is empty, we
   * do not apply any filter.
   */
  @property({ type: Array, attribute: 'focus-mode-author' })
  focusModeAuthor: string[] = [];

  @property({ type: Array, attribute: 'focus-mode-recipient' })
  focusModeRecipient: string[] = [];

  @property({ type: Array, attribute: 'focus-mode-content-type' })
  focusModeContentType: string[] = [];

  @state()
  focusModeExemptedMessageIndexes = new Set<number>();

  @state()
  deletedMessageIndexes = new Set<number>();

  @state()
  insertMessageMenuIndex: number | null = null;

  @state()
  showMessageEditorPopover = false;

  @state()
  editorFocusedMessage: Message | null = null;

  @state()
  editorFocusedMessageIndex: number | null = null;

  @property({ type: Boolean, attribute: 'is-convo-marked-for-deletion' })
  isConvoMarkedForDeletion = false;

  @state()
  hasMessageSharingURLEventListener = false;

  // Translation settings
  @state()
  hasTranslationEventListener = false;

  @state()
  isShowingTranslation = false;

  @state()
  isTranslating = false;

  @state()
  translationProgress = '';

  @state()
  translationSourceLanguage: string | null = null;

  /**
   * Custom labels. They will be shown on the header bar. Each item is a string
   * array with at most four items.
   * 1. ['value'] -> 'value'
   * 2. ['key', 'value'] -> 'key: value'
   * 3. ['key', 'value', 'tooltip text'] -> 'key: value' + tooltip text
   * 4. ['key', 'value', 'tooltip text', 'color'] -> 'key: value' + tooltip text
   *    + text color
   */
  @property({ type: Array, attribute: 'custom-labels' })
  customLabels: string[][] = [];

  /**
   * Custom message labels. They will be shown below the author icon. Each item
   * is a string array with at most four items. The first two items are
   * required, and the last two items are optional.
   * 1. [message index / 'message id', 'tooltip text']
   * 2. [message index / 'message id', 'tooltip text', 'color']
   * 3. [message index / 'message id', 'tooltip text', 'color', 'icon text']
   */
  @property({ type: Array, attribute: 'custom-message-labels' })
  customMessageLabels: CustomMessageLabel[] = [];

  // Cache the resolved labels so we do not repeatedly re-parse conversation
  // metadata on every render pass.
  private effectiveCustomLabels: string[][] = [];
  private effectiveCustomMessageLabels: CustomMessageLabel[] = [];

  private updateEffectiveCustomLabels() {
    const metadataLabels = this.conversation?.metadata
      ? getCustomLabelsFromMagicMetadata(this.conversation)
      : {
          customLabels: [],
          customMessageLabels: []
        };

    this.effectiveCustomLabels = [
      ...metadataLabels.customLabels,
      ...this.customLabels
    ];
    this.effectiveCustomMessageLabels = [
      ...metadataLabels.customMessageLabels,
      ...this.customMessageLabels
    ];
  }

  /**
   * Custom share buttons. They will be shown on in the floating bar when hover
   * over the share button. Each item is a string array with 3 items.
   * You can leave the svg string empty if you want to use the default icon.
   * ['name', 'url', 'svg string']
   */
  @property({ type: Array, attribute: 'custom-share-buttons' })
  customShareButtons: string[][] = [];

  baseTime: number | null;

  // Floating UIs
  @query('#popper-tooltip')
  popperTooltip: HTMLElement | undefined;

  @query('.message-metadata-overlay')
  messageMetadataOverlay: HTMLElement | undefined;

  isResizingMessageMetadata = false;

  @query('euphony-floating-toolbar.floating-toolbar-share')
  shareFloatingToolbar: EuphonyFloatingToolbar | undefined | null;

  @state()
  showShareFloatingToolbar = false;
  shareFloatingToolbarButtons: FloatingToolbarButton[] = [
    {
      name: 'copy-url',
      tooltip: 'Copy sharable URL',
      svgIcon: iconLink
    },
    {
      name: 'copy-json',
      tooltip: 'Copy conversation JSON',
      svgIcon: iconCopy
    },
    {
      name: 'download-json',
      tooltip: 'Download conversation JSON',
      svgIcon: iconDownload
    },
    {
      name: 'harmony-render',
      tooltip: 'Render conversation using a harmony renderer',
      svgIcon: iconArrowRotate
    }
  ];
  cleanupShareFloatingToolbarAutoUpdate: () => void = () => {};
  shareFloatingToolbarRepositionAdded = false;
  cleanupMessageEditorPopoverAutoUpdate: () => void = () => {};
  messageEditorPopoverRepositionAdded = false;
  cleanupInsertMessageMenuAutoUpdate: () => void = () => {};
  insertMessageMenuRepositionAdded = false;
  hasInsertMessageMenuOutsideClickListener = false;
  hasMessageEditorPopoverOutsideClickListener = false;

  @query('euphony-token-window')
  tokenWindowComponent: EuphonyTokenWindow | undefined;

  // Metadata
  @property({ type: Boolean, attribute: 'is-showing-metadata' })
  isShowingMetadata = false;

  @state()
  mouseoverMessage: Message | null = null;
  mouseoverMessageIndex: number | null = null;

  @state()
  isShowingMessageMetadata = false;

  // Styles
  @property({ type: Number, attribute: 'conversation-max-width' })
  conversationMaxWidth: number | null = null;

  @property({ type: Number, attribute: 'conversation-min-width' })
  conversationMinWidth: number | null = null;

  @property({ type: Boolean, attribute: 'disable-markdown-button' })
  disableMarkdownButton = false;

  @property({ type: Boolean, attribute: 'disable-translation-button' })
  disableTranslationButton = false;

  @property({ type: Boolean, attribute: 'disable-share-button' })
  disableShareButton = false;

  @property({ type: Boolean, attribute: 'disable-metadata-button' })
  disableMetadataButton = false;

  @property({ type: Boolean, attribute: 'disable-editing-mode-save-button' })
  disableEditingModeSaveButton = false;

  @property({ type: Boolean, attribute: 'disable-conversation-id-copy-button' })
  disableConversationIDCopyButton = false;

  @state()
  isShowingPreferenceWindow = false;

  euphonyStyleConfig: Record<string, string> = {};

  @property({ type: Boolean, attribute: 'disable-message-metadata' })
  disableMessageMetadata = false;

  @property({ type: Boolean, attribute: 'disable-conversation-name' })
  disableConversationName = false;

  @property({ type: Boolean, attribute: 'disable-preference-button' })
  disablePreferenceButton = false;

  @property({ type: Boolean, attribute: 'disable-token-window' })
  disableTokenWindow = false;

  @property({ type: String, attribute: 'theme' })
  theme: 'auto' | 'light' | 'dark' = 'light';

  // Do not set this property directly. Use `theme` instead.
  @property({ type: Boolean, attribute: 'is-dark-theme', reflect: true })
  isDarkTheme = false;

  // Debouncers
  toolbarTooltipDebouncer: number | null = null;
  shareFloatingToolbarDebouncer: number | null = null;
  @state()
  shareFloatingToolbarDisappearDebouncer: number | null = null;
  metadataDisappearDebouncer: number | null = null;
  metadataAppearDebouncer: number | null = null;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
    this.baseTime = null;
  }

  addEventListener(
    type:
      | keyof HTMLElementEventMap
      | 'translation-requested'
      | 'translation-completed'
      | 'conversation-metadata-button-toggled'
      | 'markdown-button-toggled'
      | 'editing-save-button-clicked'
      | 'fetch-message-sharing-url'
      | 'refresh-renderer-list-requested'
      | 'harmony-render-requested',
    listener: EventListener,
    options?: AddEventListenerOptions
  ): void {
    // Check if the parent adds the translation event listener. If not, we need
    // to hide the translation button.
    if (type === 'translation-requested') {
      this.hasTranslationEventListener = true;
    }

    if (type === 'fetch-message-sharing-url') {
      this.hasMessageSharingURLEventListener = true;
    }

    super.addEventListener(type, listener, options);
  }

  /**
   * This method is called when the DOM is added for the first time
   */
  firstUpdated() {
    // Listen to the theme change event
    window
      .matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', () => {
        if (
          window.matchMedia('(prefers-color-scheme: dark)').matches &&
          this.theme === 'auto'
        ) {
          this.isDarkTheme = true;
        } else {
          this.isDarkTheme = this.theme === 'dark';
        }
      });
  }

  /**
   * This method is called before new DOM is updated and rendered
   * @param changedProperties Property that has been changed
   */
  willUpdate(changedProperties: PropertyValues<this>) {
    if (
      changedProperties.has('conversationString') &&
      this.conversationString !== ''
    ) {
      // Convert the conversation string to JSON data
      this.conversation = parseConversationJSONString(this.conversationString);
      this.updateEffectiveCustomLabels();
      this.resetComponent();
      this.bootstrapEmptyConversationForEditorMode();
    }

    if (changedProperties.has('conversationData') && this.conversationData) {
      // Update the conversation data
      this.conversation = this.conversationData;
      this.updateEffectiveCustomLabels();
      this.resetComponent();
      this.bootstrapEmptyConversationForEditorMode();
    }

    if (
      changedProperties.has('customLabels') ||
      changedProperties.has('customMessageLabels')
    ) {
      this.updateEffectiveCustomLabels();
    }

    if (changedProperties.has('isEditable') && this.isEditable) {
      this.bootstrapEmptyConversationForEditorMode();
    }

    if (changedProperties.has('theme')) {
      if (
        window.matchMedia('(prefers-color-scheme: dark)').matches &&
        this.theme === 'auto'
      ) {
        this.isDarkTheme = true;
      } else {
        this.isDarkTheme = this.theme === 'dark';
      }
    }

    if (
      changedProperties.has('customShareButtons') &&
      this.customShareButtons.length > 0
    ) {
      // Add custom button to the floating toolbar
      for (const [i, button] of this.customShareButtons.entries()) {
        // Skip the button if it already exists
        if (
          this.shareFloatingToolbarButtons.filter(d => d.tooltip == button[0])
            .length > 0
        ) {
          continue;
        }

        this.shareFloatingToolbarButtons.push({
          name: `custom-button-${i}`,
          tooltip: button[0],
          svgIcon: button[2] === '' ? iconShare : button[2]
        });
      }
    }
  }

  updated(changedProperties: PropertyValues<this>) {
    // Update the min height of the messages based on the metadata's height
    if (
      changedProperties.has('isShowingMetadata') ||
      changedProperties.has('conversation')
    ) {
      const messagesElement =
        this.shadowRoot?.querySelector<HTMLElement>('.messages');
      const jsonViewerElement = this.shadowRoot?.querySelector<HTMLElement>(
        '.metadata euphony-json-viewer'
      );
      if (this.isShowingMetadata) {
        if (messagesElement && jsonViewerElement) {
          this.updateComplete.then(
            () => {
              const messagesRect = messagesElement.getBoundingClientRect();
              const jsonViewerRect = jsonViewerElement.getBoundingClientRect();

              if (messagesRect.height < jsonViewerRect.height) {
                const minHeight = Math.min(
                  METADATA_MIN_HEIGHT,
                  jsonViewerRect.height + 22
                );
                messagesElement.style.minHeight = `${minHeight}px`;
              } else {
                messagesElement.style.minHeight = 'auto';
              }
            },
            () => {}
          );
        }
      } else {
        if (messagesElement) {
          messagesElement.style.minHeight = 'auto';
        }
      }
    }

    // Automatically update the floating toolbar's position based on the share
    // button's position
    if (!this.shareFloatingToolbarRepositionAdded) {
      const shareButton =
        this.shadowRoot?.querySelector<HTMLElement>('.share-button');
      const floatingToolbar = this.shadowRoot?.querySelector<HTMLElement>(
        '.floating-toolbar-share'
      );
      if (shareButton && floatingToolbar) {
        this.cleanupShareFloatingToolbarAutoUpdate = autoUpdate(
          shareButton,
          floatingToolbar,
          () => {
            this.updateShareFloatingToolbarPosition(
              shareButton,
              floatingToolbar
            );
          }
        );
        this.shareFloatingToolbarRepositionAdded = true;
      }
    }

    // Keep the message editor popover positioned next to the active pen button.
    if (
      this.showMessageEditorPopover &&
      this.editorFocusedMessageIndex !== null
    ) {
      const editButton = this.shadowRoot?.querySelector<HTMLElement>(
        `.edit-button[data-message-index="${this.editorFocusedMessageIndex}"]`
      );
      const messageEditorPopover = this.shadowRoot?.querySelector<HTMLElement>(
        'euphony-message-editor-popover'
      );

      if (editButton && messageEditorPopover) {
        const shouldResetAutoUpdate =
          !this.messageEditorPopoverRepositionAdded ||
          changedProperties.has('editorFocusedMessageIndex') ||
          changedProperties.has('showMessageEditorPopover');

        if (shouldResetAutoUpdate) {
          this.cleanupMessageEditorPopoverAutoUpdate();
          this.cleanupMessageEditorPopoverAutoUpdate = autoUpdate(
            editButton,
            messageEditorPopover,
            () => {
              this.updateMessageEditorPopoverPosition(
                editButton,
                messageEditorPopover
              );
            }
          );
          this.messageEditorPopoverRepositionAdded = true;
        }

        this.updateMessageEditorPopoverPosition(
          editButton,
          messageEditorPopover
        );
      }
    } else if (this.messageEditorPopoverRepositionAdded) {
      this.cleanupMessageEditorPopoverAutoUpdate();
      this.messageEditorPopoverRepositionAdded = false;
    }

    // Keep the insert-message type menu positioned next to the active "+" button.
    if (this.insertMessageMenuIndex !== null) {
      const addButton = this.shadowRoot?.querySelector<HTMLElement>(
        `.add-button[data-message-index="${this.insertMessageMenuIndex}"]`
      );
      const insertMessageMenu = this.shadowRoot?.querySelector<HTMLElement>(
        '.add-message-type-menu'
      );

      if (addButton && insertMessageMenu) {
        const shouldResetAutoUpdate =
          !this.insertMessageMenuRepositionAdded ||
          changedProperties.has('insertMessageMenuIndex');

        if (shouldResetAutoUpdate) {
          this.cleanupInsertMessageMenuAutoUpdate();
          this.cleanupInsertMessageMenuAutoUpdate = autoUpdate(
            addButton,
            insertMessageMenu,
            () => {
              this.updateInsertMessageMenuPosition(
                addButton,
                insertMessageMenu
              );
            }
          );
          this.insertMessageMenuRepositionAdded = true;
        }

        this.updateInsertMessageMenuPosition(addButton, insertMessageMenu);
      }
    } else if (this.insertMessageMenuRepositionAdded) {
      this.cleanupInsertMessageMenuAutoUpdate();
      this.insertMessageMenuRepositionAdded = false;
    }

    if (changedProperties.has('insertMessageMenuIndex')) {
      if (
        this.insertMessageMenuIndex !== null &&
        !this.hasInsertMessageMenuOutsideClickListener
      ) {
        window.addEventListener(
          'pointerdown',
          this.insertMessageMenuWindowPointerDown
        );
        this.hasInsertMessageMenuOutsideClickListener = true;
      } else if (
        this.insertMessageMenuIndex === null &&
        this.hasInsertMessageMenuOutsideClickListener
      ) {
        window.removeEventListener(
          'pointerdown',
          this.insertMessageMenuWindowPointerDown
        );
        this.hasInsertMessageMenuOutsideClickListener = false;
      }
    }

    if (
      changedProperties.has('showMessageEditorPopover') ||
      changedProperties.has('editorFocusedMessageIndex')
    ) {
      const shouldListenForOutsideClicks =
        this.showMessageEditorPopover &&
        this.editorFocusedMessageIndex !== null;

      if (
        shouldListenForOutsideClicks &&
        !this.hasMessageEditorPopoverOutsideClickListener
      ) {
        window.addEventListener(
          'pointerdown',
          this.messageEditorPopoverWindowPointerDown
        );
        this.hasMessageEditorPopoverOutsideClickListener = true;
      } else if (
        !shouldListenForOutsideClicks &&
        this.hasMessageEditorPopoverOutsideClickListener
      ) {
        window.removeEventListener(
          'pointerdown',
          this.messageEditorPopoverWindowPointerDown
        );
        this.hasMessageEditorPopoverOutsideClickListener = false;
      }
    }
  }

  //==========================================================================||
  //                              Custom Methods                              ||
  //==========================================================================||
  async initData() {}

  refreshBaseTime() {
    this.baseTime = null;

    if (!this.conversation) {
      return;
    }

    if (this.conversation.create_time) {
      this.baseTime = this.conversation.create_time;
      return;
    }

    for (const message of this.conversation.messages) {
      if (message.create_time) {
        this.baseTime = message.create_time;
        return;
      }
    }
  }

  resetComponent() {
    this.refreshBaseTime();

    // Reset translation state so progress/results from the previous
    // conversation never bleed into the newly loaded one.
    this.isShowingTranslation = false;
    this.isTranslating = false;
    this.translationProgress = '';
    this.translationSourceLanguage = null;

    // Reset message-level editor/filter state because indices and message
    // object references are conversation-specific.
    this.deletedMessageIndexes = new Set<number>();
    this.focusModeExemptedMessageIndexes = new Set<number>();
    this.closeInsertMessageMenu();
    this.closeMessageEditorPopover();
    this.mouseoverMessage = null;
    this.mouseoverMessageIndex = null;
    this.isShowingMessageMetadata = false;
    this.isResizingMessageMetadata = false;

    // Cancel delayed metadata UI callbacks tied to the previous conversation.
    if (this.metadataDisappearDebouncer !== null) {
      clearTimeout(this.metadataDisappearDebouncer);
      this.metadataDisappearDebouncer = null;
    }
    if (this.metadataAppearDebouncer !== null) {
      clearTimeout(this.metadataAppearDebouncer);
      this.metadataAppearDebouncer = null;
    }

    // Remove floating-ui auto-update hooks that were anchored to prior DOM.
    if (this.messageEditorPopoverRepositionAdded) {
      this.cleanupMessageEditorPopoverAutoUpdate();
      this.messageEditorPopoverRepositionAdded = false;
    }
    if (this.insertMessageMenuRepositionAdded) {
      this.cleanupInsertMessageMenuAutoUpdate();
      this.insertMessageMenuRepositionAdded = false;
    }
  }

  /**
   * Return the edited conversation data after filtering deleted messages.
   * This is the single source of truth for all editor-mode exports.
   */
  getEditedConversationData(): Conversation | null {
    if (this.conversation === null) {
      throw new Error('Conversation is not set');
    }

    if (this.isConvoMarkedForDeletion) {
      return null;
    }

    const conversationClone = structuredClone(this.conversation);
    conversationClone.messages = conversationClone.messages.filter(
      (_, i) => !this.deletedMessageIndexes.has(i)
    );

    return conversationClone;
  }

  /**
   * Serialize the current conversation to a JSON string. It will ignore the
   * deleted messages.
   */
  serializeConversation(indent: number | null = null) {
    const editedConversation = this.getEditedConversationData();
    if (editedConversation === null) {
      return 'null';
    }

    let result = '';
    if (indent) {
      result = JSON.stringify(editedConversation, null, indent);
    } else {
      result = JSON.stringify(editedConversation);
    }
    return result;
  }

  updateShareFloatingToolbarPosition(
    shareButton: HTMLElement,
    floatingToolbar: HTMLElement
  ) {
    // Update the floating toolbar's position based on the share button's
    // position
    updateFloatPosition(shareButton, floatingToolbar, 'top', 3);
  }

  updateInsertMessageMenuPosition(
    addButton: HTMLElement,
    insertMessageMenu: HTMLElement
  ) {
    updateFloatPosition(addButton, insertMessageMenu, 'right', 6);
  }

  updateMessageEditorPopoverPosition(
    editButton: HTMLElement,
    messageEditorPopover: HTMLElement
  ) {
    updateFloatPosition(editButton, messageEditorPopover, 'right', 10);
  }

  /**
   * In editor mode, conversations with zero messages need a temporary deleted
   * placeholder so the existing per-message add controls still have an anchor.
   * The placeholder stays filtered out of all exports unless the user restores
   * or inserts real content.
   */
  bootstrapEmptyConversationForEditorMode() {
    if (!this.isEditable || this.conversation === null) {
      return;
    }
    if (this.conversation.messages.length > 0) {
      return;
    }

    this.conversation.messages.push({
      role: Role.User,
      content: [{ text: '' }]
    });
    this.deletedMessageIndexes.add(0);
  }

  getMessageByIndex = (messageIndex: number) => {
    const element = this.shadowRoot?.querySelector<HTMLElement>(
      `#message-${messageIndex}`
    );
    return element;
  };

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  async translationButtonClicked() {
    if (this.conversation === null) {
      return;
    }

    // Dispatch an event to let parent know that a user clicks the translation button
    // This is different from the translation-requested event. Parents are not
    // expected to handle this event.
    const event = new CustomEvent<void>('translation-button-clicked', {
      bubbles: true,
      composed: true
    });
    this.dispatchEvent(event);

    if (this.isShowingTranslation) {
      this.isShowingTranslation = false;
      return;
    } else {
      if (this.conversation.translatedMessages !== undefined) {
        // Skip the work if this conversation has already been translated
        this.isShowingTranslation = true;
        return;
      }
    }

    this.isTranslating = true;

    // Step 1: Translate all the text content in message using API
    const translatedMessages = structuredClone(
      this.conversation.messages
    ) as TranslatableMessage[];

    /**
     * Translate messages. This function will only modify `translatedMessages`
     * @param message The message to sample translatable text
     * @param messageIndex The index of the message in translatedMessages
     * @returns The translation response
     */
    const updateMessageWithTranslation = async (
      message: TranslatableMessage,
      messageIndex: number
    ) => {
      interface TranslatablePart {
        text: string;
        index: number;
        type: 'string';
      }

      // A message can contain multiple translatable parts. We will send
      // multiple requests to the API for each message
      const translatableParts: TranslatablePart[] = [];
      const contentType = getContentTypeFromContent(message.content);
      const content = getContentFromContentOrString(message.content);

      switch (contentType) {
        case 'text': {
          const typeContent = content as TextMessageContent;
          translatableParts.push({
            text: typeContent.text,
            index: 0,
            type: 'string'
          });
          break;
        }
        case 'developer': {
          const typeContent = content as DeveloperContentMessageContent;
          if (typeContent.instructions) {
            const instructions: string[] = [typeContent.instructions];

            if (instructions.length > 0) {
              for (const [i, instruction] of instructions.entries()) {
                translatableParts.push({
                  text: instruction,
                  index: i,
                  type: 'string'
                });
              }
            }
          }
          break;
        }

        default:
          throw new Error(
            `Unsupported message content type for message: ${contentType}`
          );
      }

      const translatePromises: Promise<TranslateResponseWithPartIndex>[] = [];
      for (const part of translatableParts) {
        // Create deferred promises that we will send to the parent
        const { promise, resolve, reject } =
          Promise.withResolvers<TranslateResponse>();

        // Wrap the promise with the part index
        const wrappedPromise: Promise<TranslateResponseWithPartIndex> = promise
          .then(response => {
            return {
              ...response,
              partIndex: part.index,
              partContentType: part.type
            };
          })
          .catch((error: unknown) => {
            console.error(
              'Translation failed for a conversation part, falling back to original text.',
              error
            );
            return {
              translation: part.text,
              is_translated: false,
              language: 'Failed',
              has_command: false,
              partIndex: part.index,
              partContentType: part.type
            };
          });

        // Set a timer: if the promise is not resolved in N seconds, reject it
        const promiseTimer = window.setTimeout(() => {
          reject('Timeout');
        }, TRANSLATION_TIMEOUT);

        // Wrap the resolve and reject to clear the timer
        const wrappedResolve = (value: TranslateResponse) => {
          clearTimeout(promiseTimer);
          resolve(value);
        };

        const wrappedReject = (reason?: string) => {
          clearTimeout(promiseTimer);
          reject(reason);
        };

        // Emit the text, resolver, and rejecter to the parent
        const event = new CustomEvent<TranslationRequest>(
          'translation-requested',
          {
            bubbles: true,
            composed: true,
            detail: {
              text: part.text,
              resolve: wrappedResolve,
              reject: wrappedReject
            }
          }
        );

        // Dispatch the event to the parent
        this.dispatchEvent(event);

        // Add the promise to the list of promises
        translatePromises.push(wrappedPromise);
      }

      const translateResponses = await Promise.all(translatePromises);

      if (translateResponses.some(response => response.is_translated)) {
        const translatedMessage = translatedMessages[messageIndex];
        const contentType = getContentTypeFromContent(
          translatedMessage.content
        );
        translatedMessage.isTranslated = true;
        for (const [_, response] of translateResponses.entries()) {
          const content = getContentFromContentOrString(
            translatedMessage.content
          );
          if (response.is_translated) {
            switch (contentType) {
              case 'text': {
                // Text message
                if ('text' in content) {
                  (content as TextMessageContent).text = response.translation;
                } else if ('content' in content) {
                  (content as { content: string }).content =
                    response.translation;
                }
                break;
              }

              case 'developer': {
                // System content message
                const typedContent = content as DeveloperContentMessageContent;
                if (typedContent.instructions) {
                  typedContent.instructions = response.translation;
                }
                break;
              }
              default:
                throw new Error(
                  `Unsupported message content type for translated message: ${contentType}`
                );
            }
          }
        }
      }

      return translateResponses;
    };

    // Wait for all translations to complete
    const promises: Promise<TranslateResponse[]>[] = [];

    for (const [i, message] of this.conversation.messages.entries()) {
      // We only translate [text messages, developer messages]
      const contentType = getContentTypeFromContent(message.content);
      if (contentType === 'text' || contentType === 'developer') {
        // Normalize string content into a text object before translating so
        // downstream updates mutate the translated message clone.
        const content = getContentFromContentOrString(message.content);
        const translatableMessage: TranslatableMessage =
          contentType === 'text'
            ? {
                ...translatedMessages[i],
                content: [content as TextMessageContent],
                isTranslated: false
              }
            : {
                ...translatedMessages[i],
                content: [content as DeveloperContentMessageContent],
                isTranslated: false
              };
        translatedMessages[i] = translatableMessage;
        promises.push(updateMessageWithTranslation(translatableMessage, i));
      }
    }

    // Show the translation progress as each promise resolves
    let resolvedCount = 0;
    const total = promises.length;
    const responses = await Promise.all(
      promises.map(p =>
        p.then(res => {
          resolvedCount++;
          this.translationProgress =
            total > 0 ? `(${Math.round((resolvedCount / total) * 100)}%)` : '';
          return res;
        })
      )
    );

    // Step 2: Create a new field translatedMessages based on the clone of
    // the original messages
    let hasForeignLanguage = false;
    for (const responseArray of responses) {
      for (const response of responseArray) {
        if (response.is_translated) {
          hasForeignLanguage = true;
          // Update the source language
          this.translationSourceLanguage = response.language;
          break;
        }
      }
      if (hasForeignLanguage) {
        break;
      }
    }

    // Step 3: Update the view
    this.isTranslating = false;
    this.translationProgress = '';
    if (hasForeignLanguage) {
      this.conversation.translatedMessages = translatedMessages;
      this.isShowingTranslation = true;
      const event = new CustomEvent<TranslationCompletedEventDetail>(
        'translation-completed',
        {
          bubbles: true,
          composed: true,
          detail: { translatedMessages }
        }
      );
      this.dispatchEvent(event);
    } else {
      this.conversation.translatedMessages = undefined;
    }
  }

  /**
   * MouseEnter Event handler for all the buttons in the toolbar
   * @param e Mouse event
   */
  toolButtonMouseEnter(
    e: MouseEvent,
    type:
      | 'markdown'
      | 'translate'
      | 'share'
      | 'metadata'
      | 'delete'
      | 'add'
      | 'edit'
      | 'reorder-up'
      | 'reorder-down'
      | 'custom-label'
      | 'preference'
      | 'message-share',
    maybeTooltipText?: string
  ) {
    e.stopPropagation();
    e.preventDefault();

    if (!this.popperTooltip) {
      console.error('Popper tooltip not initialized.');
      return;
    }

    const anchor = e.currentTarget as HTMLElement;

    if (this.toolbarTooltipDebouncer) {
      clearTimeout(this.toolbarTooltipDebouncer);
    }

    this.toolbarTooltipDebouncer = window.setTimeout(() => {
      // Update the content
      const labelElement = this.popperTooltip!.querySelector('.popper-label');
      let message = 'Button';
      switch (type) {
        case 'markdown': {
          message = 'Markdown rendering';
          break;
        }

        case 'translate': {
          message = 'Translate the conversation';
          break;
        }

        case 'share': {
          message = 'Copy a sharable URL';
          break;
        }

        case 'metadata': {
          message = 'Show conversation metadata';
          break;
        }

        case 'delete': {
          message = 'Delete this message';
          break;
        }

        case 'add': {
          message = 'Insert a new message';
          break;
        }

        case 'edit': {
          message = 'Focus editable fields';
          break;
        }

        case 'reorder-up': {
          message = 'Move message up';
          break;
        }

        case 'reorder-down': {
          message = 'Move message down';
          break;
        }

        case 'preference': {
          message = 'Customize display';
          break;
        }

        case 'custom-label': {
          if (maybeTooltipText === undefined) {
            console.error('maybeTooltipText is not set');
            return;
          }
          message = maybeTooltipText;
          break;
        }

        case 'message-share': {
          message = 'Copy a sharable URL for this message';
          break;
        }

        default: {
          break;
        }
      }
      labelElement!.textContent = message;
      updatePopperOverlay(this.popperTooltip!, anchor, 'top', true, 7);
      this.popperTooltip!.classList.remove('hidden');
    }, 500);
  }

  /**
   * MouseLeave Event handler for all the buttons in the toolbar
   * @param e Mouse event
   */
  toolButtonMouseLeave(useTransition = true) {
    if (!this.popperTooltip) {
      console.error('popperTooltip are not initialized yet.');
      return;
    }

    if (this.toolbarTooltipDebouncer) {
      clearTimeout(this.toolbarTooltipDebouncer);
      this.toolbarTooltipDebouncer = null;
    }

    if (useTransition) {
      this.popperTooltip.classList.add('hidden');
    } else {
      this.popperTooltip.classList.add('no-transition');
      this.popperTooltip.classList.add('hidden');
      setTimeout(() => {
        this.popperTooltip!.classList.remove('no-transition');
      }, 150);
    }
  }

  shareButtonMouseEnter() {
    if (this.shareFloatingToolbarDebouncer) {
      clearTimeout(this.shareFloatingToolbarDebouncer);
      this.shareFloatingToolbarDebouncer = null;
    }

    if (this.shareFloatingToolbarDisappearDebouncer) {
      clearTimeout(this.shareFloatingToolbarDisappearDebouncer);
      this.shareFloatingToolbarDisappearDebouncer = null;
    }

    this.shareFloatingToolbarDebouncer = window.setTimeout(() => {
      this.showShareFloatingToolbar = true;
    }, 500);
  }

  shareButtonMouseLeave() {
    if (this.shareFloatingToolbarDebouncer) {
      clearTimeout(this.shareFloatingToolbarDebouncer);
      this.shareFloatingToolbarDebouncer = null;
    }

    this.shareFloatingToolbarDisappearDebouncer = window.setTimeout(() => {
      this.showShareFloatingToolbar = false;
    }, 600);
  }

  async shareFloatingToolbarButtonClicked(e: CustomEvent<string>) {
    if (!this.shareFloatingToolbar) {
      console.error('Share floating toolbar not initialized');
      return;
    }

    const buttonName = e.detail;
    switch (buttonName) {
      case 'copy-url': {
        if (!this.sharingURL) {
          console.error('Sharing URL is not set');
          return;
        }
        await navigator.clipboard.writeText(this.sharingURL);
        this.shareFloatingToolbar.updateCurrentTooltip('Copied');
        break;
      }

      case 'copy-json': {
        let jsonString = this.serializeConversation(2);
        if (this.overrideSharingJSONString) {
          jsonString = this.overrideSharingJSONString;
        }
        await navigator.clipboard.writeText(jsonString);
        this.shareFloatingToolbar.updateCurrentTooltip('Copied');
        break;
      }

      case 'download-json': {
        let jsonString = this.serializeConversation(2);
        if (this.overrideSharingJSONString) {
          jsonString = this.overrideSharingJSONString;
        }
        const fileName = 'euphony-conversation.json';
        downloadText(jsonString, null, fileName);
        this.shareFloatingToolbar.updateCurrentTooltip('Downloaded');
        break;
      }

      case 'harmony-render': {
        if (!this.tokenWindowComponent) {
          console.error('Token window component not initialized');
          return;
        }
        // Get the conversation string
        const conversationString = this.serializeConversation();

        // In library mode, we show the token window from the convo element
        if (!this.disableTokenWindow) {
          this.tokenWindowComponent.show(conversationString);
        } else {
          // When the conversation is hosted inside the app shell, the shared token
          // window lives on the app element instead of here.
          const event = new CustomEvent<string>(
            'harmony-render-button-clicked',
            {
              bubbles: true,
              composed: true,
              detail: conversationString
            }
          );
          this.dispatchEvent(event);
        }
        break;
      }

      default: {
        if (buttonName.includes('custom-button')) {
          const customButtonIndex = parseInt(buttonName.split('-')[2]);
          if (customButtonIndex >= this.customShareButtons.length) {
            console.error(
              'Custom button index is out of range:',
              customButtonIndex
            );
            return;
          }
          const url = this.customShareButtons[customButtonIndex][1];
          if (url) {
            window.open(url, '_blank');
          }
        } else {
          console.error('Unsupported button name:', buttonName);
        }
        break;
      }
    }
  }

  metadataButtonClicked() {
    this.toolButtonMouseLeave(false);

    // Notify the parent
    const event = new CustomEvent<boolean>(
      'conversation-metadata-button-toggled',
      {
        bubbles: true,
        composed: true,
        detail: !this.isShowingMetadata
      }
    );
    this.isShowingMetadata = !this.isShowingMetadata;
    this.dispatchEvent(event);
  }

  markdownButtonClicked() {
    // Notify the parent
    const event = new CustomEvent<boolean>('markdown-button-toggled', {
      bubbles: true,
      composed: true,
      detail: !this.shouldRenderMarkdown
    });
    this.shouldRenderMarkdown = !this.shouldRenderMarkdown;
    this.dispatchEvent(event);
  }

  /**
   * Send the current conversation string to the parent
   */
  editingSaveButtonClicked() {
    const curConversationString = this.serializeConversation();
    const event = new CustomEvent<string>('editing-save-button-clicked', {
      bubbles: true,
      composed: true,
      detail: curConversationString
    });
    this.dispatchEvent(event);
  }

  private swapDeletedMessageIndexes(a: number, b: number) {
    const next = new Set(this.deletedMessageIndexes);
    const hasA = next.has(a);
    const hasB = next.has(b);
    if (hasA) {
      next.delete(a);
    }
    if (hasB) {
      next.delete(b);
    }
    if (hasA) {
      next.add(b);
    }
    if (hasB) {
      next.add(a);
    }
    this.deletedMessageIndexes = next;
  }

  private shiftDeletedIndexesAfterInsert(insertedIndex: number) {
    const next = new Set<number>();
    for (const index of this.deletedMessageIndexes) {
      next.add(index >= insertedIndex ? index + 1 : index);
    }
    this.deletedMessageIndexes = next;
  }

  reorderUpButtonClicked(messageIndex: number) {
    if (!this.conversation || messageIndex <= 0) {
      return;
    }
    const messages = this.conversation.messages;
    [messages[messageIndex - 1], messages[messageIndex]] = [
      messages[messageIndex],
      messages[messageIndex - 1]
    ];
    this.swapDeletedMessageIndexes(messageIndex - 1, messageIndex);
    this.closeMessageEditorPopover();
    this.requestUpdate();
  }

  reorderDownButtonClicked(messageIndex: number) {
    if (
      !this.conversation ||
      messageIndex >= this.conversation.messages.length - 1
    ) {
      return;
    }
    const messages = this.conversation.messages;
    [messages[messageIndex], messages[messageIndex + 1]] = [
      messages[messageIndex + 1],
      messages[messageIndex]
    ];
    this.swapDeletedMessageIndexes(messageIndex, messageIndex + 1);
    this.closeMessageEditorPopover();
    this.requestUpdate();
  }

  createEmptyMessageForContentType(
    referenceMessage: Message | undefined,
    contentType: 'text' | 'system' | 'developer'
  ): Message {
    // Keep text insertions aligned with nearby message role, but force
    // system/developer roles to match the selected structured content type.
    if (contentType === 'system') {
      const today = new Date().toISOString().slice(0, 10);
      return {
        role: Role.System,
        content: [
          {
            model_identity:
              'You are ChatGPT, a large language model trained by OpenAI.',
            conversation_start_date: today,
            knowledge_cutoff: '2024-06',
            channel_config: {
              valid_channels: ['analysis', 'commentary', 'final'],
              channel_required: true
            }
          }
        ]
      };
    }

    if (contentType === 'developer') {
      return {
        role: Role.Developer,
        content: [{ instructions: '' }]
      };
    }

    return {
      role: referenceMessage?.role ?? Role.User,
      content: [{ text: '' }]
    };
  }

  async insertMessageAfterIndex(
    messageIndex: number,
    contentType: 'text' | 'system' | 'developer'
  ) {
    if (!this.conversation) {
      return;
    }
    const referenceMessage = this.conversation.messages[messageIndex];
    const newMessage = this.createEmptyMessageForContentType(
      referenceMessage,
      contentType
    );
    const insertIndex = messageIndex + 1;
    this.conversation.messages.splice(insertIndex, 0, newMessage);
    this.shiftDeletedIndexesAfterInsert(insertIndex);
    this.closeMessageEditorPopover();
    this.insertMessageMenuIndex = null;
    this.requestUpdate();
    await this.updateComplete;
    this.focusEditableFieldsForMessage(insertIndex);
  }

  closeInsertMessageMenu() {
    this.insertMessageMenuIndex = null;
  }

  insertMessageMenuWindowPointerDown = (e: Event) => {
    if (this.insertMessageMenuIndex === null) {
      return;
    }

    const menu = this.shadowRoot?.querySelector<HTMLElement>(
      '.add-message-type-menu'
    );
    const addButton = this.shadowRoot?.querySelector<HTMLElement>(
      `.add-button[data-message-index="${this.insertMessageMenuIndex}"]`
    );
    const eventPath = e.composedPath();

    if (
      (menu && eventPath.includes(menu)) ||
      (addButton && eventPath.includes(addButton))
    ) {
      return;
    }

    this.closeInsertMessageMenu();
    this.requestUpdate();
  };

  messageEditorAddMessageButtonClicked(messageIndex: number) {
    if (this.insertMessageMenuIndex === messageIndex) {
      this.closeInsertMessageMenu();
      return;
    }

    // Keep message action overlays mutually exclusive.
    this.closeMessageEditorPopover();
    this.insertMessageMenuIndex = messageIndex;
    this.requestUpdate();
  }

  closeMessageEditorPopover() {
    this.showMessageEditorPopover = false;
    this.editorFocusedMessage = null;
    this.editorFocusedMessageIndex = null;
  }

  messageEditorEditButtonClicked(messageIndex: number) {
    if (!this.conversation) {
      return;
    }

    if (
      this.showMessageEditorPopover &&
      this.editorFocusedMessageIndex === messageIndex
    ) {
      this.closeMessageEditorPopover();
      this.requestUpdate();
      return;
    }

    this.closeInsertMessageMenu();
    this.editorFocusedMessageIndex = messageIndex;
    this.editorFocusedMessage =
      this.conversation.messages[messageIndex] ?? null;
    this.showMessageEditorPopover = !!this.editorFocusedMessage;
    this.messageEditorPopoverRepositionAdded = false;
    this.requestUpdate();
  }

  messageEditorPopoverSaveButtonClicked(
    e: CustomEvent<MessageEditorUserSetData>
  ) {
    if (!this.editorFocusedMessage) {
      return;
    }

    const userSetData = e.detail;
    this.editorFocusedMessage.role = userSetData.role;
    this.editorFocusedMessage.name = userSetData.name;
    this.editorFocusedMessage.recipient = userSetData.recipient;
    this.editorFocusedMessage.channel = userSetData.channel;

    this.closeMessageEditorPopover();
    this.requestUpdate();
  }

  messageEditorPopoverCancelButtonClicked() {
    this.closeMessageEditorPopover();
    this.requestUpdate();
  }

  messageEditorPopoverWindowPointerDown = (e: Event) => {
    if (
      !this.showMessageEditorPopover ||
      this.editorFocusedMessageIndex === null
    ) {
      return;
    }

    const popover = this.shadowRoot?.querySelector<HTMLElement>(
      'euphony-message-editor-popover'
    );
    const editButton = this.shadowRoot?.querySelector<HTMLElement>(
      `.edit-button[data-message-index="${this.editorFocusedMessageIndex}"]`
    );
    const eventPath = e.composedPath();

    if (
      (popover && eventPath.includes(popover)) ||
      (editButton && eventPath.includes(editButton))
    ) {
      return;
    }

    this.closeMessageEditorPopover();
    this.requestUpdate();
  };

  focusEditableFieldsForMessage(messageIndex: number) {
    const messageElement = this.shadowRoot?.querySelector<HTMLElement>(
      `#message-${messageIndex}`
    );
    if (!messageElement) {
      return;
    }
    const editable = messageElement.querySelector<HTMLElement>(
      '[contenteditable="true"]'
    );
    if (!editable) {
      return;
    }
    editable.focus();
    const selection = window.getSelection();
    if (!selection) {
      return;
    }
    const range = document.createRange();
    range.selectNodeContents(editable);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  preferenceButtonClicked() {
    this.isShowingPreferenceWindow = !this.isShowingPreferenceWindow;
  }

  messageInfoMouseEnter(e: MouseEvent, message: Message, messageIndex: number) {
    if (this.disableMessageMetadata) {
      return;
    }

    if (!this.messageMetadataOverlay) {
      console.error('Message metadata overlay not initialized.');
      return;
    }

    if (this.metadataDisappearDebouncer) {
      clearTimeout(this.metadataDisappearDebouncer);
      this.metadataDisappearDebouncer = null;
    }

    if (this.metadataAppearDebouncer) {
      clearTimeout(this.metadataAppearDebouncer);
      this.metadataAppearDebouncer = null;
    }

    const anchor = e.currentTarget as HTMLElement;
    anchor.classList.add('is-hovered');
    const timeoutDelay = this.isShowingMessageMetadata ? 0 : 300;
    this.messageMetadataOverlay.scrollTo({ top: 0, behavior: 'instant' });

    this.metadataAppearDebouncer = window.setTimeout(() => {
      this.mouseoverMessage = message;
      this.mouseoverMessageIndex = messageIndex;
      updatePopperOverlay(
        this.messageMetadataOverlay!,
        anchor,
        'left',
        true,
        7
      );
      this.isShowingMessageMetadata = true;
    }, timeoutDelay);
  }

  messageInfoMouseLeave() {
    if (!this.messageMetadataOverlay) {
      console.error('Message metadata overlay not initialized.');
      return;
    }

    if (this.metadataAppearDebouncer) {
      clearTimeout(this.metadataAppearDebouncer);
      this.metadataAppearDebouncer = null;
    }

    const messageInfoElement = this.shadowRoot!.querySelector(
      '.message-info.is-hovered'
    )!;
    messageInfoElement.classList.remove('is-hovered');

    // Set a timeout here so user can move the mouse to the overlay
    this.metadataDisappearDebouncer = window.setTimeout(() => {
      this.isShowingMessageMetadata = false;
    }, 500);
  }

  metadataOverlayMouseEnter() {
    if (!this.messageMetadataOverlay) {
      console.error('Message metadata overlay not initialized.');
      return;
    }

    if (this.metadataDisappearDebouncer) {
      clearTimeout(this.metadataDisappearDebouncer);
      this.metadataDisappearDebouncer = null;
    }

    // Also highlight the message info
    const messageInfoElement = this.shadowRoot!.querySelector(
      `#message-info-${this.mouseoverMessageIndex}`
    )!;
    messageInfoElement.classList.add('is-hovered');

    this.isShowingMessageMetadata = true;
  }

  metadataOverlayMouseLeave() {
    if (!this.messageMetadataOverlay) {
      console.error('Message metadata overlay not initialized.');
      return;
    }

    if (this.isResizingMessageMetadata) {
      return;
    }

    const messageInfoElement = this.shadowRoot!.querySelector(
      '.message-info.is-hovered'
    )!;
    messageInfoElement.classList.remove('is-hovered');

    // Set a timeout here so user can move the mouse back
    this.metadataDisappearDebouncer = window.setTimeout(() => {
      this.isShowingMessageMetadata = false;
    }, 500);
  }

  metadataOverlayShareButtonClicked(e: MouseEvent, messageIndex: number) {
    if (!this.sharingURL) {
      console.error('Sharing URL is not set');
      return;
    }
    const anchor = e.currentTarget as HTMLElement;

    // Notify the host to get the message's sharing URL
    const { promise, resolve, reject } = getDeferredPromise<string>(1000);

    promise
      .then(async (url: string) => {
        await navigator.clipboard.writeText(url);
        const labelElement = this.popperTooltip!.querySelector('.popper-label');
        labelElement!.textContent = 'Copied';
        updatePopperOverlay(this.popperTooltip!, anchor, 'top', true, 7);
      })
      .catch(() => {});

    const event = new CustomEvent<MessageSharingRequest>(
      'fetch-message-sharing-url',
      {
        bubbles: true,
        composed: true,
        detail: {
          messageIndex,
          resolve,
          reject
        }
      }
    );
    this.dispatchEvent(event);
  }

  /**
   * Prevent mouse leave when user drag the overlay to resize it
   * Note this handler is not called on Safari. For some reason, Safari doesn't
   * fire mousedown event when user clicks the resize handle. It doesn't fire it
   * on window or document as well when user clicks the resize handle :(
   * WebKit but: https://bugs.webkit.org/show_bug.cgi?id=280956
   * @param e Mouse event
   */
  metadataMouseDown = () => {
    if (!this.messageMetadataOverlay) {
      console.error('Message metadata overlay not initialized.');
      return;
    }

    // Fix the width and height and remove max width and max height
    const bbox = this.messageMetadataOverlay.getBoundingClientRect();
    this.messageMetadataOverlay.style.width = `${bbox.width}px`;
    this.messageMetadataOverlay.style.height = `${bbox.height}px`;
    this.messageMetadataOverlay.style.maxHeight = 'unset';
    this.messageMetadataOverlay.style.maxWidth = 'unset';

    this.isResizingMessageMetadata = true;

    const mouseUpHandler = () => {
      this.isResizingMessageMetadata = false;
      window.removeEventListener('mouseup', mouseUpHandler);
    };

    window.addEventListener('mouseup', mouseUpHandler);
  };

  //==========================================================================||
  //                             Private Helpers                              ||
  //==========================================================================||
  loadKatexScript() {
    // Hack to load external script from web component
    // https://stackoverflow.com/questions/55686830/litelement-load-external-script
    // Katex is too large to be bundled with the app, so we have to do this :(
    if (!('katex' in window)) {
      const script = document.createElement('script');
      script.src =
        'https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js';
      script.defer = true;
      script.integrity =
        'sha384-Rma6DA2IPUwhNxmrB/7S3Tno0YY7sFu9WSYMCuulLhIqYSGZ2gKCJWIqhBWqMQfh';
      script.crossOrigin = 'anonymous';

      // Check if the stylesheet link is already present in the document head
      if (
        !document.querySelector(
          'link[href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css"]'
        )
      ) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href =
          'https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css';
        link.integrity =
          'sha384-zh0CIslj+VczCZtlzBcjt5ppRcsAmDnRem7ESsYwWwg3m/OaJ2l4x7YBZl9Kxxib';
        link.crossOrigin = 'anonymous';
        document.head.appendChild(link);
      }

      return script;
    }
  }

  preferenceWindowMaxMessageHeightChanged(e: CustomEvent<string>) {
    const newHeight = e.detail;
    this.euphonyStyleConfig['--max-message-height'] = newHeight;
    this.requestUpdate();
  }

  preferenceWindowFocusModeSettingsChanged(e: CustomEvent<FocusModeSettings>) {
    const focusModeSettings = e.detail;
    this.focusModeAuthor = [...focusModeSettings.author];
    this.focusModeRecipient = [...focusModeSettings.recipient];
    this.focusModeContentType = [...focusModeSettings.contentType];
  }

  preferenceWindowMessageLabelChanged(e: CustomEvent<MessageLabelSettings>) {
    const messageLabelSettings = e.detail;
    if (messageLabelSettings.absoluteTimestamp) {
      this.euphonyStyleConfig['--message-label-absolute-timestamp-display'] =
        'block';
      this.euphonyStyleConfig['--message-label-relative-timestamp-display'] =
        'none';
    } else {
      delete this.euphonyStyleConfig[
        '--message-label-absolute-timestamp-display'
      ];
      delete this.euphonyStyleConfig[
        '--message-label-relative-timestamp-display'
      ];
    }

    this.requestUpdate();
  }

  async allChildrenUpdateComplete() {
    await this.updateComplete;

    const childrenTags = ['euphony-message-text'];
    const promises: Promise<boolean>[] = [];

    for (const tag of childrenTags) {
      const elements = this.shadowRoot?.querySelectorAll<LitElement>(tag);
      if (elements) {
        elements.forEach(element => {
          promises.push(element.updateComplete);
        });
      }
    }

    await Promise.all(promises);
  }

  relativeTimestampFormatter(creationTime: number): string {
    if (this.baseTime === null) {
      console.error('Base time is not set');
      return '';
    }

    const t = Math.max(0, Math.floor(creationTime) - this.baseTime);
    let rem = t;
    const s: string[] = [];

    const units: [string, number][] = [
      ['d', 3600 * 24],
      ['h', 3600],
      ['m', 60],
      ['s', 1]
    ];

    for (const [unit, divisor] of units) {
      if (t < divisor && divisor > 1) {
        continue;
      }
      const v = Math.floor(rem / divisor);
      rem = rem % divisor;
      s.push(`${v}${unit}`);
    }

    return s.join(' ');
  }

  absoluteTimestampFormatter(creationTime: number): string {
    const date = new Date(creationTime * 1000);
    const dateString = `${date.getFullYear()}-${(date.getMonth() + 1)
      .toString()
      .padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')}`;
    const timeString = `${date.getHours().toString().padStart(2, '0')}:${date
      .getMinutes()
      .toString()
      .padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}`;
    return `${dateString} ${timeString}`;
  }

  getAuthorIcon(role: Role) {
    let authorIcon = html`<span class="role-icon"
      >${unsafeHTML(iconUser)}</span
    >`;

    switch (role) {
      case Role.Assistant: {
        authorIcon = html`<span class="role-icon svg-icon"
          >${unsafeHTML(iconAssistant)}</span
        >`;
        break;
      }

      case Role.User: {
        authorIcon = html`<span class="role-icon svg-icon"
          >${unsafeHTML(iconUser)}</span
        >`;
        break;
      }

      case Role.System: {
        authorIcon = html`<span class="role-icon svg-icon"
          >${unsafeHTML(iconSystem)}</span
        >`;
        break;
      }

      case Role.Tool: {
        authorIcon = html`<span class="role-icon svg-icon"
          >${unsafeHTML(iconTool)}</span
        >`;
        break;
      }

      case Role.Developer: {
        authorIcon = html`<span class="role-icon svg-icon"
          >${unsafeHTML(iconDeveloper)}</span
        >`;
        break;
      }

      default: {
        console.warn('Unsupported role:', role);
      }
    }
    return authorIcon;
  }

  /**
   * Check if the message is hidden by focus mode settings.
   * @param message Message to check
   * @returns True if the message is hidden, false otherwise
   */
  isMessageHiddenByFocusMode(message: Message, messageIndex: number) {
    if (this.focusModeExemptedMessageIndexes.has(messageIndex)) {
      return false;
    }

    if (this.focusModeAuthor.length > 0) {
      if (!this.focusModeAuthor.includes(message.role)) {
        return true;
      }
    }

    if (this.focusModeRecipient.length > 0) {
      if (!this.focusModeRecipient.includes(message.recipient ?? '')) {
        return true;
      }
    }

    if (this.focusModeContentType.length > 0) {
      const contentType = tryGetContentTypeFromContent(message.content);
      if (
        contentType === null ||
        !this.focusModeContentType.includes(contentType)
      ) {
        return true;
      }
    }

    return false;
  }

  getMessageContentTemplate(message: Message, i: number) {
    if (this.conversation === null) {
      throw new Error('Conversation is not set');
    }
    const messageElementID = `message-${i}`;

    // Check if the message is hidden by focus mode settings. If so, we override
    // an in-line style to hide the message.
    const isMessageHiddenByFocusMode = this.isMessageHiddenByFocusMode(
      message,
      i
    );
    const inlineStyle = isMessageHiddenByFocusMode ? 'display: none;' : '';

    let template = html``;
    const contentType = tryGetContentTypeFromContent(message.content);
    switch (contentType) {
      case 'text': {
        template = html`
          <euphony-message-text
            .message=${message}
            id=${messageElementID}
            style=${inlineStyle}
            ?shouldRenderMarkdown=${this.shouldRenderMarkdown}
            .markdownAllowedTags=${this.markdownAllowedTags}
            .markdownAllowedAttributes=${this.markdownAllowedAttributes}
            ?isEditable=${this.isEditable}
            ?isTranslation=${this.isShowingTranslation &&
            this.conversation.translatedMessages &&
            (message as TranslatableMessage).isTranslated}
            @message-text-changed=${(e: CustomEvent<string>) => {
              const editableMessage = this.conversation!.messages[i];
              if (typeof editableMessage.content === 'string') {
                editableMessage.content = e.detail;
                return;
              }

              const messageContent = editableMessage.content[0] as
                | TextMessageContent
                | { content: string }
                | undefined;
              if (!messageContent) {
                editableMessage.content = [{ text: e.detail }];
                return;
              }

              if ('text' in messageContent) {
                messageContent.text = e.detail;
              } else {
                messageContent.content = e.detail;
              }
            }}
          ></euphony-message-text>
        `;
        break;
      }

      case 'code': {
        template = html`
          <euphony-message-code
            .message=${message}
            id=${messageElementID}
            style=${inlineStyle}
          ></euphony-message-code>
        `;
        break;
      }

      case 'system': {
        template = html`
          <euphony-message-system-content
            .message=${message}
            id=${messageElementID}
            style=${inlineStyle}
            ?shouldRenderMarkdown=${this.shouldRenderMarkdown}
            .markdownAllowedTags=${this.markdownAllowedTags}
            .markdownAllowedAttributes=${this.markdownAllowedAttributes}
            ?isEditable=${this.isEditable}
            ?isTranslation=${this.isShowingTranslation &&
            this.conversation.translatedMessages &&
            (message as TranslatableMessage).isTranslated}
            .dataFileURL=${this.dataFileURL}
            @message-system-content-changed=${(
              e: CustomEvent<SystemContentEditPayload>
            ) => {
              const messageContent = this.conversation!.messages[i]
                .content[0] as SystemContentMessageContent;
              const { location, newContent } = e.detail;
              switch (location) {
                case 'model_identity':
                  messageContent.model_identity = newContent;
                  break;
                case 'conversation_start_date':
                  messageContent.conversation_start_date = newContent;
                  break;
                case 'knowledge_cutoff':
                  messageContent.knowledge_cutoff = newContent;
                  break;
                case 'valid_channels':
                  if (messageContent.channel_config) {
                    messageContent.channel_config.valid_channels = newContent
                      .split(',')
                      .map(channel => channel.trim())
                      .filter(Boolean);
                  }
                  break;
                case 'channel_required':
                  if (messageContent.channel_config) {
                    messageContent.channel_config.channel_required =
                      newContent.trim().toLowerCase() === 'true';
                  }
                  break;
                default:
                  console.warn('Unsupported system edit location:', location);
              }
            }}
          ></euphony-message-system-content>
        `;
        break;
      }

      case 'developer': {
        template = html`
          <euphony-message-developer-content
            .message=${message}
            id=${messageElementID}
            style=${inlineStyle}
            ?shouldRenderMarkdown=${this.shouldRenderMarkdown}
            .markdownAllowedTags=${this.markdownAllowedTags}
            .markdownAllowedAttributes=${this.markdownAllowedAttributes}
            ?isEditable=${this.isEditable}
            @message-developer-content-changed=${(
              e: CustomEvent<DeveloperContentEditPayload>
            ) => {
              const messageContent = this.conversation!.messages[i]
                .content[0] as DeveloperContentMessageContent;
              const { location, index, newContent } = e.detail;

              switch (location) {
                case 'instruction':
                  messageContent.instructions = newContent;
                  break;
                case 'tool_namespace_name':
                  if (messageContent.tools && typeof index === 'string') {
                    messageContent.tools[index].name = newContent;
                  }
                  break;
                case 'tool_namespace_description':
                  if (messageContent.tools && typeof index === 'string') {
                    messageContent.tools[index].description = newContent;
                  }
                  break;
                default:
                  console.warn(
                    'Unsupported developer edit location:',
                    location
                  );
              }
            }}
          ></euphony-message-developer-content>
        `;
        break;
      }

      default: {
        console.error('Unsupported message content type:', contentType);
        template = html`
          <euphony-message-unsupported
            .message=${message}
            id=${messageElementID}
            style=${inlineStyle}
          ></euphony-message-unsupported>
        `;
        break;
      }
    }

    // Add message action controls in editor mode.
    if (this.isEditable) {
      template = html`<div
        class="editable-message"
        style=${inlineStyle}
        ?is-deleted=${this.deletedMessageIndexes.has(i)}
      >
        ${template}
        <div class="action-group">
          <div class="action-item">
            <button
              class="svg-icon reorder-up-button"
              ?disabled=${i === 0}
              @mouseenter=${(e: MouseEvent) => {
                this.toolButtonMouseEnter(e, 'reorder-up');
              }}
              @mouseleave=${() => {
                this.toolButtonMouseLeave();
              }}
              @click=${() => {
                this.reorderUpButtonClicked(i);
              }}
            >
              ${unsafeHTML(iconChevron)}
            </button>

            <button
              class="svg-icon add-button"
              data-message-index=${i}
              ?is-activated=${this.insertMessageMenuIndex === i}
              @mouseenter=${(e: MouseEvent) => {
                this.toolButtonMouseEnter(e, 'add');
              }}
              @mouseleave=${() => {
                this.toolButtonMouseLeave();
              }}
              @click=${() => {
                this.toolButtonMouseLeave();
                this.messageEditorAddMessageButtonClicked(i);
              }}
            >
              ${unsafeHTML(iconPlusSquare)}
            </button>

            <button
              class="svg-icon reorder-down-button"
              ?disabled=${i === this.conversation.messages.length - 1}
              @mouseenter=${(e: MouseEvent) => {
                this.toolButtonMouseEnter(e, 'reorder-down');
              }}
              @mouseleave=${() => {
                this.toolButtonMouseLeave();
              }}
              @click=${() => {
                this.reorderDownButtonClicked(i);
              }}
            >
              ${unsafeHTML(iconChevron)}
            </button>
          </div>

          <div class="action-item">
            <button
              class="svg-icon delete-button action-item"
              @mouseenter=${(e: MouseEvent) => {
                this.toolButtonMouseEnter(e, 'delete');
              }}
              @mouseleave=${() => {
                this.toolButtonMouseLeave();
              }}
              @click=${() => {
                if (this.deletedMessageIndexes.has(i)) {
                  this.deletedMessageIndexes.delete(i);
                } else {
                  this.deletedMessageIndexes.add(i);
                }
                this.requestUpdate();
              }}
            >
              ${unsafeHTML(iconTrash)}
            </button>

            <button
              class="svg-icon edit-button action-item"
              data-message-index=${i}
              ?is-activated=${this.showMessageEditorPopover &&
              this.editorFocusedMessageIndex === i}
              @mouseenter=${(e: MouseEvent) => {
                this.toolButtonMouseEnter(e, 'edit');
              }}
              @mouseleave=${() => {
                this.toolButtonMouseLeave();
              }}
              @click=${() => {
                this.toolButtonMouseLeave();
                this.messageEditorEditButtonClicked(i);
              }}
            >
              ${unsafeHTML(iconPen)}
            </button>
          </div>
        </div>
      </div>`;
    }
    return template;
  }

  renderTextWithWordBreaks(text: string): TemplateResult[] {
    // Keep each text segment escaped by Lit while preserving the original
    // wrapping behavior after common identifier separators.
    return text.split(/([_.-])/g).map(part => {
      if (part === '_' || part === '-' || part === '.') {
        return html`${part}<wbr>`;
      }
      return html`${part}`;
    });
  }

  getMessageMetadataInfo(message: Message) {
    const formatTimestamp = (timestamp: number): string => {
      const date = new Date(timestamp * 1000);

      const options: Intl.DateTimeFormatOptions = {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      };

      return date.toLocaleString('en-US', options);
    };

    let template = html``;

    if (message.name) {
      template = html`${template}
        <span class="message-metadata-info-tag"
          >author: ${message.name}</span
        > `;
    }

    if (message.create_time) {
      template = html`${template}
        <span class="message-metadata-info-tag"
          >created: ${formatTimestamp(message.create_time)}</span
        > `;
    }

    if (message.recipient) {
      template = html`${template}
        <span class="message-metadata-info-tag"
          >recipient: ${message.recipient}</span
        > `;
    }

    if (message.channel) {
      template = html`${template}
        <span class="message-metadata-info-tag"
          >channel: ${message.channel}</span
        > `;
    }

    return template;
  }

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    let header = html``;
    let messageElements = html``;
    let metadataElement = html``;

    if (this.conversation) {
      // Add a translation source label if needed
      let translationSourceLabel = html``;

      if (this.isShowingTranslation && this.translationSourceLanguage) {
        translationSourceLabel = html`<div class="translation-label">
          Translated from ${this.translationSourceLanguage}
        </div>`;
      }

      // Add a sharing button (allow copy & download JSON by default)
      // Ff the the sharing URL is set, also allow copying the URL
      const sharingButton = html`
        <button
          class="icon svg-icon share-button"
          ?is-hidden=${this.disableShareButton}
          ?is-activated=${this.showShareFloatingToolbar}
          @mouseenter=${() => {
            this.shareButtonMouseEnter();
          }}
          @mouseleave=${() => {
            this.shareButtonMouseLeave();
          }}
        >
          ${unsafeHTML(iconShare)}
        </button>
      `;

      // Compile custom labels
      let customLabelsContent = html``;
      for (const label of this.effectiveCustomLabels) {
        let mouseenterHandler = (_: MouseEvent) => {};
        let mouseleaveHandler = () => {};
        const customLabelStyleConfig: Record<string, string> = {};

        if (label.length >= 4) {
          customLabelStyleConfig['--custom-label-color'] = `${label[3]};`;
        }

        if (label.length >= 3 && label[2] !== '') {
          mouseenterHandler = (e: MouseEvent) => {
            this.toolButtonMouseEnter(e, 'custom-label', label[2]);
          };
          mouseleaveHandler = () => {
            this.toolButtonMouseLeave();
          };
        }

        let innerContent = html``;
        if (label.length >= 2 && label[1] !== '') {
          innerContent = html`
            <div class="label-name">${label[0]}:</div>
            <div class="label-text">${label[1]}</div>
          `;
        } else {
          innerContent = html` <div class="label-name">${label[0]}</div> `;
        }

        customLabelsContent = html`${customLabelsContent}
          <div
            class="custom-label"
            style=${styleToString(customLabelStyleConfig)}
            @mouseenter=${mouseenterHandler}
            @mouseleave=${mouseleaveHandler}
          >
            ${innerContent}
          </div>`;
      }

      // Save button in editing mode
      let saveButton = html``;
      if (this.isEditable && !this.disableEditingModeSaveButton) {
        saveButton = html`<button
          class="text-button"
          @click=${() => {
            this.editingSaveButtonClicked();
          }}
        >
          Save
        </button>`;
      }

      header = html`<div
        class="header"
        ?is-showing-metadata=${this.isShowingMetadata}
      >
        <div
          class="label-group"
          ?is-hidden=${this.disableConversationName}
          ?no-show=${this.disableConversationName &&
          this.disableShareButton &&
          this.disableMarkdownButton &&
          this.disableTranslationButton &&
          this.disableMetadataButton}
        >
          <div class="conversation-label-group">
            <span class="conversation-label">${this.conversationLabel}:</span>
            <span class="conversation-id" title=${this.conversation.id ?? ''}
              >${this.conversation.id?.slice(0, 8) ?? ''}</span
            >
            <sl-copy-button
              value=${this.conversation.id ?? ''}
              size="small"
              copy-label="Copy conversation ID"
              hoist
              ?is-hidden=${this.disableConversationIDCopyButton}
            >
            </sl-copy-button>
          </div>

          <div class="loader-container" ?is-loading=${this.isTranslating}>
            <div class="loader-label">
              Translating ${this.translationProgress}
            </div>
            <div class="loader"></div>
          </div>
          ${saveButton} ${translationSourceLabel} ${customLabelsContent}
        </div>

        <div class="action-group">
          <button
            class="icon svg-icon preference-button"
            ?is-active=${this.isShowingPreferenceWindow}
            ?is-hidden=${this.disablePreferenceButton}
            @click=${() => {
              this.preferenceButtonClicked();
            }}
            @mouseenter=${(e: MouseEvent) => {
              this.toolButtonMouseEnter(e, 'preference');
            }}
            @mouseleave=${() => {
              this.toolButtonMouseLeave();
            }}
          >
            ${unsafeHTML(iconSystem)}
          </button>

          <button
            class="icon svg-icon markdown-button"
            ?is-active=${this.shouldRenderMarkdown}
            ?is-hidden=${this.isEditable || this.disableMarkdownButton}
            @click=${() => {
              this.markdownButtonClicked();
            }}
            @mouseenter=${(e: MouseEvent) => {
              this.toolButtonMouseEnter(e, 'markdown');
            }}
            @mouseleave=${() => {
              this.toolButtonMouseLeave();
            }}
          >
            ${this.shouldRenderMarkdown
              ? unsafeHTML(iconEyeOff)
              : unsafeHTML(iconEye)}
          </button>

          <button
            class="icon svg-icon translate-button"
            ?disabled=${this.isTranslating}
            ?is-active=${this.isShowingTranslation || this.isTranslating}
            ?is-hidden=${this.isEditable ||
            this.disableTranslationButton ||
            !this.hasTranslationEventListener}
            @mouseenter=${(e: MouseEvent) => {
              this.toolButtonMouseEnter(e, 'translate');
            }}
            @mouseleave=${() => {
              this.toolButtonMouseLeave();
            }}
            @click=${() => {
              this.translationButtonClicked().then(
                () => {},
                () => {}
              );
            }}
          >
            ${unsafeHTML(iconTranslate)}
          </button>

          <button
            class="icon svg-icon metadata-button"
            ?is-active=${this.isShowingMetadata}
            ?is-hidden=${this.disableMetadataButton}
            @click=${() => {
              this.metadataButtonClicked();
            }}
            @mouseenter=${(e: MouseEvent) => {
              this.toolButtonMouseEnter(e, 'metadata');
            }}
            @mouseleave=${() => {
              this.toolButtonMouseLeave();
            }}
          >
            ${unsafeHTML(iconMetaData)}
          </button>

          ${sharingButton}
        </div>
      </div>`;

      let curMessages = this.conversation.messages;

      if (this.isShowingTranslation && this.conversation.translatedMessages) {
        curMessages = this.conversation.translatedMessages;
      }

      if (curMessages.length > MAX_MESSAGE_COUNT) {
        // Insert a message to the front before slicing to 3000
        const insertedMessage: Message = {
          role: Role.Tool,
          name: 'Euphony',
          content: [
            {
              text: `This conversation is truncated to ${MAX_MESSAGE_COUNT} messages from the bottom (total: ${curMessages.length}).`
            }
          ],
          recipient: 'all',
          channel: undefined,
          metadata: {}
        };
        curMessages = [
          insertedMessage,
          ...curMessages.slice(0, MAX_MESSAGE_COUNT),
          insertedMessage
        ];
      }

      for (const [i, message] of curMessages.entries()) {
        // Get the role of the author and select the icon
        const role = message.role;
        const authorIcon = this.getAuthorIcon(role);

        // Show the author's name if it is set
        let authorName = html``;
        if (message.name) {
          authorName = html`<span class="label label-text"
            ><span class="name-text" title=${message.name}
              >${this.renderTextWithWordBreaks(message.name)}</span
            ></span
          >`;
        }

        let timestampLabel = html``;
        if (this.baseTime !== null && message.create_time && i > 0) {
          timestampLabel = html`<div
            class="label label-text label-relative-timestamp"
          >
            ${this.relativeTimestampFormatter(message.create_time)}
          </div>`;
        }

        let absoluteTimestampLabel = html``;
        if (message.create_time) {
          absoluteTimestampLabel = html`<div
            class="label label-text label-absolute-timestamp"
          >
            ${this.absoluteTimestampFormatter(message.create_time)}
          </div>`;
        }

        // Show the recipient label if there is a special one
        const arrow = html` <span class="arrow svg-icon"
          >${unsafeHTML(iconArrow)}
        </span>`;
        let recipientLabel = html``;
        const normalRecipients = new Set(['all']);
        if (message.recipient && !normalRecipients.has(message.recipient)) {
          recipientLabel = html`
            <span class="label label-text">
              ${arrow}<span class="recipient-text" title=${message.recipient}
                >${this.renderTextWithWordBreaks(message.recipient)}</span
              ></span
            >
          `;
        }

        // Show the channel if it is set
        let channelLabel = html``;
        if (message.channel !== undefined && message.channel !== null) {
          channelLabel = html`
            <span class="label label-text channel">
              ${arrow}<span class="channel-text">${message.channel}</span></span
            >
          `;
        }

        // Add custom message labels
        let customLabels = html``;
        const customMessageLabelStyleConfig: Record<string, string> = {};

        // Filter custom message labels based on the message index or message id
        const curCustomMessageLabels = this.effectiveCustomMessageLabels.filter(
          label => {
            if (typeof label[0] === 'number') {
              return label[0] === i;
            }
            return false;
          }
        );

        if (curCustomMessageLabels.length > 0) {
          // Build the custom labels content
          const labelTemplates: TemplateResult[] = [];
          for (const label of curCustomMessageLabels) {
            const mouseenterHandler = (e: MouseEvent) => {
              this.toolButtonMouseEnter(e, 'custom-label', label[1]);
            };
            const mouseleaveHandler = () => {
              this.toolButtonMouseLeave();
            };
            const customLabelStyleConfig: Record<string, string> = {};

            if (label.length >= 3) {
              customLabelStyleConfig['--custom-label-color'] = `${label[2]};`;
            }

            customLabelsContent = html`<div
              class="custom-label"
              style=${styleToString(customLabelStyleConfig)}
              @mouseenter=${mouseenterHandler}
              @mouseleave=${mouseleaveHandler}
            >
              ${label[3]}
            </div>`;
            labelTemplates.push(customLabelsContent);
          }

          customLabels = html`<div class="custom-labels">
            ${labelTemplates}
          </div>`;

          // Update the message content style if there is a custom label
          customMessageLabelStyleConfig['--message-content-border'] =
            `3px solid ${curCustomMessageLabels[0][2]};`;
          customMessageLabelStyleConfig['--message-content-border-left'] =
            `3px solid ${curCustomMessageLabels[0][2]};`;
          customMessageLabelStyleConfig['--message-content-border-radius'] =
            '4px;';
          customMessageLabelStyleConfig['--conv-background-color'] = `color-mix(
            in lab, ${curCustomMessageLabels[0][2]} 7%, transparent 100%);`;
        }

        // Get the appropriate message content element based on the content type
        const messageContentTemplate = this.getMessageContentTemplate(
          message,
          i
        );

        // Check if the message is hidden by focus mode settings. If so, we show
        // a hidden message element instead.
        const isMessageHiddenByFocusMode = this.isMessageHiddenByFocusMode(
          message,
          i
        );
        const messageHiddenTemplate = isMessageHiddenByFocusMode
          ? html`<euphony-message-hidden
              .message=${message}
              @hidden-message-clicked=${() => {
                this.focusModeExemptedMessageIndexes = new Set(
                  this.focusModeExemptedMessageIndexes
                ).add(i);
              }}
            ></euphony-message-hidden>`
          : html``;

        messageElements = html`${messageElements}
          <div
            class="message"
            ?is-user=${role === Role.User}
            ?is-assistant=${role === Role.Assistant}
            style=${styleToString(customMessageLabelStyleConfig)}
          >
            <div
              class="message-info"
              id=${`message-info-${i}`}
              tabindex=${1}
              @mouseenter=${(e: MouseEvent) => {
                this.messageInfoMouseEnter(e, message, i);
              }}
              @mouseleave=${() => {
                this.messageInfoMouseLeave();
              }}
            >
              <div class="author">${authorIcon}</div>
              ${timestampLabel} ${absoluteTimestampLabel} ${authorName}
              ${recipientLabel} ${channelLabel} ${customLabels}
            </div>

            ${messageContentTemplate} ${messageHiddenTemplate}
          </div> `;
      }

      // Show the metadata
      metadataElement = html`<euphony-json-viewer
        .data=${this.conversation.metadata}
        ?is-dark-theme=${this.isDarkTheme}
      >
      </euphony-json-viewer>`;
    }

    // Tooltips
    const tooltipTemplate = html`
      <div
        id="popper-tooltip"
        class="popper-tooltip hidden"
        role="tooltip"
        @click=${(e: MouseEvent) => {
          e.stopPropagation();
        }}
      >
        <div class="popper-content">
          <span class="popper-label">Hello</span>
        </div>
        <div class="popper-arrow"></div>
      </div>
    `;

    // Floating toolbar
    let curShareFloatingToolbarButtons = structuredClone(
      this.shareFloatingToolbarButtons
    );
    if (!this.sharingURL) {
      curShareFloatingToolbarButtons = curShareFloatingToolbarButtons.filter(
        button => button.name !== 'copy-url'
      );
    }
    const shareFloatingToolbar = html`
      <euphony-floating-toolbar
        ?is-hidden=${!this.showShareFloatingToolbar}
        .buttons=${curShareFloatingToolbarButtons}
        disappearTimeout=${this.shareFloatingToolbarDisappearDebouncer ?? -1}
        class="floating-toolbar-share"
        @mouseleave=${() => {
          this.shareButtonMouseLeave();
        }}
        @button-clicked=${(e: CustomEvent<string>) => {
          this.shareFloatingToolbarButtonClicked(e).then(
            () => {},
            () => {}
          );
        }}
      ></euphony-floating-toolbar>
    `;

    let messageEditorPopoverTemplate = html``;
    if (this.showMessageEditorPopover && this.editorFocusedMessage) {
      messageEditorPopoverTemplate = html`
        <euphony-message-editor-popover
          .message=${this.editorFocusedMessage}
          @save-button-clicked=${(e: CustomEvent<MessageEditorUserSetData>) => {
            this.messageEditorPopoverSaveButtonClicked(e);
          }}
          @cancel-button-clicked=${() => {
            this.messageEditorPopoverCancelButtonClicked();
          }}
        ></euphony-message-editor-popover>
      `;
    }

    let addMessageTypeMenuTemplate = html``;
    if (this.insertMessageMenuIndex !== null) {
      addMessageTypeMenuTemplate = html`
        <div
          class="add-message-type-menu"
          @click=${(e: MouseEvent) => {
            e.stopPropagation();
          }}
        >
          <button
            class="add-message-type-menu-item"
            @click=${() => {
              void this.insertMessageAfterIndex(
                this.insertMessageMenuIndex!,
                'text'
              );
            }}
          >
            Text
          </button>
          <button
            class="add-message-type-menu-item"
            @click=${() => {
              void this.insertMessageAfterIndex(
                this.insertMessageMenuIndex!,
                'system'
              );
            }}
          >
            System
          </button>
          <button
            class="add-message-type-menu-item"
            @click=${() => {
              void this.insertMessageAfterIndex(
                this.insertMessageMenuIndex!,
                'developer'
              );
            }}
          >
            Developer
          </button>
        </div>
      `;
    }

    // Overlay JSONL Viewer for message metadata
    let headerContent = html``;
    let messageSharingButton = html``;
    if (this.mouseoverMessage && this.mouseoverMessageIndex) {
      const role = this.mouseoverMessage.role;
      const roleName = role.charAt(0).toUpperCase() + role.slice(1);
      const contentType = getContentTypeFromContent(
        this.mouseoverMessage.content
      );
      const contentTypeName =
        contentType.charAt(0).toUpperCase() + contentType.slice(1);
      headerContent = html`<span
        >${roleName} ${contentTypeName} Metadata</span
      >`;

      // Build message-level sharing button
      messageSharingButton = html`
        <button
          class="icon svg-icon message-share-button"
          ?is-hidden=${!this.hasMessageSharingURLEventListener}
          @mouseenter=${(e: MouseEvent) => {
            this.toolButtonMouseEnter(e, 'message-share');
          }}
          @mouseleave=${() => {
            this.toolButtonMouseLeave();
          }}
          @click=${(e: MouseEvent) => {
            this.metadataOverlayShareButtonClicked(
              e,
              this.mouseoverMessageIndex!
            );
          }}
        >
          ${unsafeHTML(iconLink)}
        </button>
      `;
    }

    let messageMetadataInfo = html``;
    if (this.mouseoverMessage) {
      messageMetadataInfo = this.getMessageMetadataInfo(this.mouseoverMessage);
    }

    const messageMetadata = html` <div
      class="message-metadata-overlay"
      ?is-hidden=${!this.isShowingMessageMetadata}
      role="tooltip"
      tabindex="0"
      @mousedown=${() => {
        this.metadataMouseDown();
      }}
      @mouseenter=${() => {
        this.metadataOverlayMouseEnter();
      }}
      @mouseleave=${() => {
        this.metadataOverlayMouseLeave();
      }}
    >
      <div class="metadata-header">
        <div class="metadata-header-name">${headerContent}</div>
        <div class="metadata-header-share-button">${messageSharingButton}</div>
      </div>
      <div class="metadata-info">${messageMetadataInfo}</div>
      <euphony-json-viewer
        .data=${this.mouseoverMessage?.metadata ?? null}
        ?is-dark-theme=${this.isDarkTheme}
      >
      </euphony-json-viewer>
      <div class="popper-arrow"></div>
    </div>`;

    let style = styleToString(this.euphonyStyleConfig);
    let splitPanelStyle = '--min: 100px;';

    if (this.conversationMaxWidth) {
      style += `--conversation-max-width: ${this.conversationMaxWidth}px;`;
      splitPanelStyle += `--max: ${this.conversationMaxWidth}px;`;
    }

    if (this.conversationMinWidth) {
      style += `--conversation-min-width: ${this.conversationMinWidth}px;`;
      splitPanelStyle += `--min: ${this.conversationMinWidth}px;`;
    }

    // Compile the content template based on whether to show metadata
    let contentTemplate = html``;

    if (this.isShowingMetadata) {
      contentTemplate = html`
        <sl-split-panel position="60" style=${splitPanelStyle}>
          <div
            class="messages"
            is-showing-metadata=${this.isShowingMessageMetadata}
            slot="start"
          >
            ${messageElements}
          </div>
          <div
            class="metadata"
            is-showing-metadata=${this.isShowingMessageMetadata}
            slot="end"
          >
            <div class="metadata-header">Conversation Metadata</div>
            ${metadataElement}
          </div>
          <div class="my-divider" slot="divider"></div>
        </sl-split-panel>
      `;
    } else {
      contentTemplate = html` <div class="messages">${messageElements}</div> `;
    }

    // Preference window
    const preferenceWindowTemplate = html`
      <euphony-preference-window
        ?is-hidden=${this.disablePreferenceButton ||
        !this.isShowingPreferenceWindow}
        .enabledOptions=${{
          maxMessageHeight: true,
          gridView: false,
          expandAndCollapseAll: true,
          advanced: true,
          messageLabel: true,
          focusMode: true
        }}
        ?is-dark-theme=${this.isDarkTheme}
        @preference-window-close-clicked=${() => {
          this.isShowingPreferenceWindow = false;
        }}
        @max-message-height-changed=${(e: CustomEvent<string>) => {
          this.preferenceWindowMaxMessageHeightChanged(e);
        }}
        @message-label-changed=${(e: CustomEvent<MessageLabelSettings>) => {
          this.preferenceWindowMessageLabelChanged(e);
        }}
        @expand-all-clicked=${() => {
          this.expandBlockContents();
        }}
        @collapse-all-clicked=${() => {
          this.collapseBlockContents();
        }}
        @translate-all-clicked=${() => {
          void this.translationButtonClicked();
        }}
        @focus-mode-settings-changed=${(e: CustomEvent<FocusModeSettings>) => {
          this.preferenceWindowFocusModeSettingsChanged(e);
        }}
      ></euphony-preference-window>
    `;

    // Harmony render window
    const tokenWindowTemplate = html`
      <euphony-token-window
        ?is-hidden=${this.disableTokenWindow}
      ></euphony-token-window>
    `;

    return html`
      ${tooltipTemplate} ${messageMetadata} ${shareFloatingToolbar}
      ${messageEditorPopoverTemplate} ${addMessageTypeMenuTemplate}
      ${preferenceWindowTemplate} ${tokenWindowTemplate}
      <div
        class="conversation"
        tabindex="0"
        style=${style}
        ?is-dark-theme=${this.isDarkTheme}
      >
        ${header}
        <div class="content">${contentTemplate}</div>
      </div>
      ${this.loadKatexScript()}
    `;
  }

  static styles = [
    css`
      ${unsafeCSS(shoelaceCSS)}
      ${unsafeCSS(componentCSS)}
    `
  ];

  // Place holder so that one can call it on all euphony elements
  expandBlockContents() {
    sharedExpandBlockContents(this);
  }

  // Place holder so that one can call it on all euphony elements
  collapseBlockContents() {
    sharedCollapseBlockContents(this);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'euphony-conversation': EuphonyConversation;
  }
}

export const parseConversationJSONString = (
  conversationString: string
): Conversation | null => {
  try {
    return JSON.parse(conversationString) as Conversation;
  } catch (e) {
    console.error(e);
    console.error(
      'Error parsing conversation JSON string:',
      conversationString
    );
    return null;
  }
};
