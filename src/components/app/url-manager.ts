import { EuphonyApp } from './app';

export class URLManager {
  private app: EuphonyApp;

  constructor(app: EuphonyApp) {
    this.app = app;
  }

  /**
   * Update the URL based on the current configs.
   */
  updateURL() {
    // Maintain the current hash
    const currentHash = window.location.hash;
    const params = new URLSearchParams(window.location.search);

    // Pagination
    params.set('page', this.app.curPage.toString());
    params.set('limit', this.app.itemsPerPage.toString());

    // Markdown
    if (this.app.globalShouldRenderMarkdown) {
      params.set('markdown', 'true');
    } else {
      params.delete('markdown');
    }

    // Metadata
    if (this.app.globalIsShowingMetadata) {
      params.set('metadata', 'true');
    } else {
      params.delete('metadata');
    }

    // JMESPath query
    if (this.app.jmespathQuery) {
      params.set('jmespath', this.app.jmespathQuery);
    } else {
      params.delete('jmespath');
    }

    if (this.app.isGridView) {
      params.set('grid', this.app.gridViewColumnWidth.toString());
    } else {
      params.delete('grid');
    }

    // Editor mode
    if (this.app.isEditorMode) {
      params.set('editor', 'true');
    } else {
      params.delete('editor');
    }

    // Frontend only mode
    if (this.app.isFrontendOnlyMode) {
      params.set('frontend-only', 'true');
    } else {
      params.delete('frontend-only');
    }

    const paramsString = params.toString();

    history.pushState({}, '', `?${paramsString}${currentHash}`);
  }

  /**
   * Update the configs based on the current URL.
   */
  updateConfigsFromURL() {
    const params = new URLSearchParams(window.location.search);

    // Pagination
    const pageParam = params.get('page');
    if (pageParam) {
      this.app.curPage = parseInt(pageParam);
    }

    const limitParam = params.get('limit');
    if (limitParam) {
      this.app.itemsPerPage = parseInt(limitParam);
    }

    // Markdown
    const markdownParam = params.get('markdown');
    if (markdownParam) {
      this.app.globalShouldRenderMarkdown = markdownParam === 'true';
    }

    // Conversation metadata
    const conversationMetadataParam = params.get('metadata');
    if (conversationMetadataParam) {
      this.app.globalIsShowingMetadata = conversationMetadataParam === 'true';
    }

    // Editor mode
    const editorParam = params.get('editor');
    if (editorParam) {
      this.app.isEditorMode = editorParam === 'true';
      this.app.itemsPerPage = 100000000;
    }

    // JMESPath query
    const jmespathParam = params.get('jmespath');
    if (jmespathParam) {
      this.app.jmespathQuery = jmespathParam;
    }

    // Grid mode
    const gridViewParam = params.get('grid');
    if (gridViewParam) {
      this.app.isGridView = true;
      this.app.gridViewColumnWidth = parseInt(gridViewParam);
      this.app.appStyleConfig['--app-grid-view-column-width'] =
        `${parseInt(gridViewParam)}px`;
    }

    // Frontend only mode
    const frontendOnlyParam = params.get('frontend-only');
    if (frontendOnlyParam) {
      this.app.isFrontendOnlyMode = frontendOnlyParam === 'true';
    }

    // TODO: when adding new params, remember to update getShareURL() as well
  }

  /**
   * Get the share URL for a conversation.
   * @param conversationID The ID of the conversation to share.
   * @returns The share URL for the conversation.
   */
  getShareURL = (conversationID: number, blobPath: string | null) => {
    // Note: we can't just read the current URL, because getSharedURL might be
    // called before the URL update.
    let url = '';
    const params = new URLSearchParams();
    if (blobPath !== null && blobPath !== '') {
      params.set('path', blobPath);
      params.set('page', this.app.curPage.toString());
      params.set('limit', this.app.itemsPerPage.toString());
    }
    if (this.app.globalIsShowingMetadata) {
      params.set('metadata', 'true');
    }
    if (this.app.globalShouldRenderMarkdown) {
      params.set('markdown', 'true');
    }
    if (this.app.jmespathQuery) {
      params.set('jmespath', this.app.jmespathQuery);
    }
    if (this.app.isGridView) {
      params.set('grid', this.app.gridViewColumnWidth.toString());
    }

    params.set('index', conversationID.toString());
    url = `${window.location.origin}${window.location.pathname}?${params.toString()}`;

    return url;
  };

  /**
   * Get the share URL for a message.
   * @param conversationID The ID of the conversation to share.
   * @returns The share URL for the message.
   */
  getMessageShareURL = (
    conversationID: number,
    messageIndex: number,
    blobPath: string | null
  ) => {
    // Note: we can't just read the current URL, because getSharedURL might be
    // called before the URL update.
    let url = '';
    const params = new URLSearchParams();
    if (blobPath !== null && blobPath !== '') {
      params.set('path', blobPath);
      params.set('page', this.app.curPage.toString());
      params.set('limit', this.app.itemsPerPage.toString());
    }
    if (this.app.globalIsShowingMetadata) {
      params.set('metadata', 'true');
    }
    if (this.app.globalShouldRenderMarkdown) {
      params.set('markdown', 'true');
    }
    if (this.app.jmespathQuery) {
      params.set('jmespath', this.app.jmespathQuery);
    }
    if (this.app.isGridView) {
      params.set('grid', this.app.gridViewColumnWidth.toString());
    }

    params.set('index', conversationID.toString());
    params.set('subindex', messageIndex.toString());
    url = `${window.location.origin}${window.location.pathname}?${params.toString()}`;

    return url;
  };
}
