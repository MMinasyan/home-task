"use strict";

// Regression check for the chat client's delayed-POST buffer handling, run
// directly: `node tests/test_browser.js`. It loads the real app.js into a vm
// context with only the DOM/EventSource/fetch methods the script consumes.

const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SOURCE = fs.readFileSync(
  path.join(__dirname, "..", "src", "agent_qa", "static", "app.js"),
  "utf8",
);

function makeElement() {
  return {
    textContent: "",
    className: "",
    value: "",
    disabled: false,
    children: [],
    listeners: {},
    append(...nodes) {
      this.children.push(...nodes);
    },
    addEventListener(type, handler) {
      (this.listeners[type] ??= []).push(handler);
    },
  };
}

function makePage() {
  const elements = {};
  const sources = [];
  const pending = [];
  let uuids = 0;

  const document = {
    getElementById(id) {
      return (elements[id] ??= makeElement());
    },
    createElement(tag) {
      return makeElement();
    },
    createTextNode(text) {
      return { text };
    },
  };

  class EventSource {
    constructor() {
      this.listeners = {};
      sources.push(this);
    }

    close() {}

    addEventListener(type, handler) {
      (this.listeners[type] ??= []).push(handler);
    }

    emit(type, data) {
      for (const handler of this.listeners[type] ?? []) {
        handler({ data: JSON.stringify(data) });
      }
    }
  }

  function fetch(url) {
    return new Promise((resolve) => pending.push({ url, resolve }));
  }

  const page = {
    elements,
    sources,
    pending,
    question: null,
    conversation: null,
  };

  const sandbox = {
    document,
    EventSource,
    fetch,
    crypto: { randomUUID: () => `id-${++uuids}` },
    setTimeout,
  };
  vm.runInNewContext(SOURCE, sandbox, { filename: "app.js" });

  page.question = elements.question;
  page.conversation = elements.conversation;
  page.dispatch = (target, type, event) => {
    for (const handler of target.listeners[type] ?? []) handler(event);
  };
  page.respond = async (url, status, body) => {
    const held = pending.find((item) => item.url === url);
    assert.notEqual(held, undefined, `no held request for ${url}`);
    pending.splice(pending.indexOf(held), 1);
    held.resolve({ ok: status >= 200 && status < 300, status, json: async () => body });
    await new Promise(setImmediate);
  };
  page.displayedAnswer = () => {
    const lines = page.conversation.children;
    if (lines.length === 0) return "";
    const last = lines[lines.length - 1];
    const text = last.children
      .map((node) => node.textContent ?? node.text)
      .join("");
    return text.replace(/^(You|Agent) /, "");
  };
  return page;
}

async function connectedPage() {
  const page = makePage();
  await page.respond("/api/session", 200, { messages: [], turn: null, outcome: null });
  const stream = page.sources[0];
  assert.notEqual(stream, undefined, "EventSource opened after bootstrap");
  stream.onopen();
  await page.respond("/api/session", 200, { messages: [], turn: null, outcome: null });
  return page;
}

// A held successful POST reply must not wipe buffered text that SSE already
// delivered and a snapshot already confirmed while the reply was in flight.
async function confirmedTextSurvivesDelayedReply() {
  const page = await connectedPage();
  page.question.value = "q";
  page.dispatch(page.elements.composer, "submit", { preventDefault() {} });
  await new Promise(setImmediate);
  assert.equal(page.pending.filter((item) => item.url === "/api/message").length, 1);

  page.sources[0].emit("delta", { turn_id: "T1", text: "Hel" });
  await page.respond("/api/session", 200, {
    messages: [{ role: "user", text: "q" }],
    turn: { id: "T1" },
    outcome: null,
  });
  assert.equal(page.displayedAnswer(), "Hel");

  await page.respond("/api/message", 200, { turn_id: "T1" });
  assert.equal(
    page.displayedAnswer(),
    "Hel",
    "the delayed POST reply must not wipe confirmed buffered text",
  );

  // The subsequent numbered snapshot stays authoritative.
  await page.respond("/api/session", 200, {
    messages: [
      { role: "user", text: "q" },
      { role: "assistant", text: "Hello" },
    ],
    turn: null,
    outcome: "success",
  });
  assert.equal(page.displayedAnswer(), "Hello");
  assert.equal(page.elements.send.disabled, false);
}

// A newer buffered turn identity must survive an older delayed POST reply.
async function newerIdentitySurvivesOlderReply() {
  const page = await connectedPage();
  page.question.value = "q1";
  page.dispatch(page.elements.composer, "submit", { preventDefault() {} });
  await new Promise(setImmediate);

  // Another session writer's newer turn streams in before the delayed reply.
  page.sources[0].emit("delta", { turn_id: "T2", text: "B" });
  await page.respond("/api/session", 200, {
    messages: [{ role: "user", text: "q2" }],
    turn: { id: "T2" },
    outcome: null,
  });
  assert.equal(page.displayedAnswer(), "B");

  await page.respond("/api/message", 200, { turn_id: "T1" });
  assert.equal(
    page.displayedAnswer(),
    "B",
    "the older delayed reply must not overwrite the newer buffered identity",
  );
}

// While the first snapshot response is unresolved, the exact disabled
// predicate (snapshot === null, no EventSource) must already lock the composer.
async function initialComposerDisabledWhileSnapshotUnresolved() {
  const page = makePage();
  assert.equal(
    page.question.disabled,
    true,
    "textarea must be disabled while the first snapshot is unresolved",
  );
  assert.equal(
    page.elements.send.disabled,
    true,
    "Send must be disabled while the first snapshot is unresolved",
  );
  await page.respond("/api/session", 200, { messages: [], turn: null, outcome: null });
  const stream = page.sources[0];
  assert.notEqual(stream, undefined, "EventSource opened after bootstrap");
  assert.equal(
    page.elements.send.disabled,
    false,
    "the resolved bootstrap must enable the composer",
  );
}

(async () => {
  const failures = [];
  for (const [name, check] of [
    ["confirmed text survives delayed reply", confirmedTextSurvivesDelayedReply],
    ["newer identity survives older reply", newerIdentitySurvivesOlderReply],
    [
      "composer disabled while first snapshot unresolved",
      initialComposerDisabledWhileSnapshotUnresolved,
    ],
  ]) {
    try {
      await check();
      console.log(`ok: ${name}`);
    } catch (error) {
      failures.push(name);
      console.error(`failed: ${name}: ${error.message}`);
    }
  }
  if (failures.length > 0) process.exit(1);
  console.log("browser regression checks passed");
})();