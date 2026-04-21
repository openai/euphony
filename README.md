# Euphony: Harmony Chat and Codex Session Viewer

[![Build](https://github.com/openai/euphony/actions/workflows/build.yml/badge.svg)](https://github.com/openai/euphony/actions/workflows/build.yml)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/openai/euphony/blob/main/LICENSE)

Visualize harmony chat conversations and Codex sessions in your browser 🎵

<table>
  <tr>
    <td colspan="3"><a href="https://openai.github.io/euphony/"><img src='https://github.com/user-attachments/assets/cfdccb43-a63f-4495-8718-63efde8a1a11' width="100%"></a></td>
  </tr>
  <tr></tr>
  <tr>
     <td><a href="https://openai.github.io/euphony/">🎵 Euphony Demo</a></td>
     <td><a href="https://openai.github.io/euphony/?path=https://huggingface.co/datasets/xiaohk/x-datasets/resolve/main/health-bench-hard-20250508.jsonl">💬 Harmony Chat Example</a></td>
     <td><a href="https://openai.github.io/euphony/?path=https://huggingface.co/datasets/victor/codex-sample-session/resolve/main/codex-session-hi-2026-03-16.jsonl">🤖 Codex Session Example</a></td>
  </tr>
</table>

[Harmony](https://github.com/openai/harmony) conversations and [Codex session
logs](https://developers.openai.com/codex/cli/features) are useful across
training, evaluation, and agent workflows, but they are often difficult to
inspect. Euphony addresses that gap by providing **portable**, **customizable**,
and **modular** _Web Components_ for visualizing
structured chat data and Codex sessions in the browser.

## Features

| Feature                     | What it does                                                                                                               |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| Harmony conversation viewer | Renders Harmony conversations with support for different message types and metadata.                                       |
| Codex session viewer        | Detects Codex session JSONL files, converts them into a conversation, and renders them in the same viewer.                 |
| Flexible loading            | Loads data from the clipboard, local `.json` or `.jsonl` files, or public HTTP(S) JSON/JSONL URLs.                         |
| Markdown and HTML rendering | Renders markdown in message content, including formulas and optional HTML blocks.                                          |
| Translation                 | Translates non-English text into English in normal mode or frontend-only mode with a user-provided OpenAI API key.         |
| Metadata inspection         | Exposes conversation-level and message-level metadata directly in the UI.                                                  |
| Filtering and focus mode    | Filters datasets with JMESPath and narrows visible messages by role, recipient, or content type.                           |
| Grid and editor modes       | Supports dataset skimming in grid view and direct JSONL editing in editor mode.                                            |
| Harmony token rendering     | Shows Harmony renderer output, token IDs, decoded tokens, and rendered display strings.                                    |
| Embeddable web components   | Ships reusable custom elements for integrating the viewer into other web apps in any framework (e.g., React, Svelte, Vue). |

## Get Started

There are two ways you can use Euphony.

1. Use the [standalone app](https://openai.github.io/euphony/) to view Harmony JSON/JSONL data and Codex session JSONL files.
2. Use the Euphony JavaScript library to render Harmony data in your own web app.

### Use [Euphony](https://openai.github.io/euphony/) to View My Data

1. Load data from one of the supported sources:
   1. Paste JSON or JSONL from the clipboard
   2. Choose a local `.json` or `.jsonl` file
   3. Provide a public HTTP(S) URL that serves JSON or JSONL (e.g., Hugging Face)
2. Euphony automatically detects and renders the input:
   1. If the JSONL is a list of conversations → Euphony renders all conversations
   2. If the JSONL is a Codex session file → Euphony renders a structured Codex session timeline
   3. If the conversation is stored at some top-level field → Euphony renders all conversations and treat other top-level fields as each conversation’s metadata
   4. Else → Euphony renders the data as raw JSON objects

### Integrate Euphony into My Web App

#### Web Component API

To use Euphony web components, first build the Euphony library:

```
pnpm install
pnpm run build:library
```

The main library entrypoint is built at: `./lib/euphony.js`.

Then, you can import the JS file. To show a viewer of Harmony data, you can add the web component to your HTML file (or React, Svelte, Vue components):

```html
<euphony-conversation conversation-string="HARMONY_CONVERSATION_JSON_STRING">
</euphony-conversation>
```

Alternatively, you can pass parsed JSON object directly into the `euphony-conversation` `HTMLElement` from Javascript.

```tsx
const harmonyConversation: Conversation;
const myEuphonyConvoElement = document.querySelector('euphony-conversation');
myEuphonyConvoElement.conversationJSON = harmonyConversation;
```

Euphony web components are customizable. For example, to style these components, you can override CSS variables in your stylesheets.

```css
--euphony-padding-v: 10px;
--euphony-padding-h: 15px;
--euphony-font-color: hsl(0, 0%, 12.94%);
--euphony-font-color-secondary: hsl(0, 0%, 61.96%);
--euphony-user-color: hsl(122, 43.43%, 38.82%);
--euphony-assistant-color: hsl(282, 67.88%, 37.84%);
--euphony-conv-background-color: hsl(0, 0%, 96.08%);
```

## Frontend-only and Backend-assisted Modes

Euphony can run in two modes:

| Mode                  | What it is for                                                                                                                                                    |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frontend-only mode    | Recommended for static or external hosting. URL loading happens in the browser, and translation uses a user-provided OpenAI API key when needed.                  |
| Backend-assisted mode | Optional for local development. The FastAPI server supports remote JSON/JSONL loading, backend translation, and Harmony rendering. Useful to load large datasets. |

The backend server is optional. It should only be used locally. Frontend-only mode is controlled with the Vite env variable `VITE_EUPHONY_FRONTEND_ONLY`.

- Set `VITE_EUPHONY_FRONTEND_ONLY=true` to force frontend-only mode.
- Set `VITE_EUPHONY_FRONTEND_ONLY=false` to use the local backend-assisted mode.

The current backend includes a remote URL fetch path for loading JSON and JSONL data. If you host that backend on an external server, it can introduce SSRF risk because the server may be tricked into fetching attacker-controlled URLs. Therefore, only use the backend-assisted mode for local development.

## Development

To develop Euphony locally, install Node.js and a package manager such as
[pnpm](https://pnpm.io/).

Start the backend server:

```bash
pnpm install
uvicorn fastapi-main:app --app-dir server --host 127.0.0.1 --port 8020 --reload
```

Start the frontend development server:

```bash
pnpm run dev
```

To force frontend-only mode with Vite:

```bash
VITE_EUPHONY_FRONTEND_ONLY=true pnpm run dev
```

Visit [http://localhost:3000/](http://localhost:3000/), you should see Euphony running in your browser!

To build the static frontend:

```bash
pnpm run build
python -m http.server -d ./dist
```

To build a frontend-only static bundle:

```bash
VITE_EUPHONY_FRONTEND_ONLY=true pnpm run build
python -m http.server -d ./dist
```

## License

Euphony is released under the Apache 2.0 license. See [`LICENSE`](./LICENSE).

The file [`src/css/prism-coldark-auto.css`](./src/css/prism-coldark-auto.css) is derived
from an upstream Prism theme ([GitHub](https://github.com/ArmandPhilippot/coldark-prism))
and remains subject to the MIT license terms of that upstream work.
