"use strict";

// Browser chat client over the session/message/events API: one authoritative
// snapshot, one transient answer buffer keyed by turn ID, one EventSource, one
// reconnect timer, and one monotonically increasing snapshot-request counter.

const conversation = document.getElementById("conversation");
const statusLine = document.getElementById("status");
const question = document.getElementById("question");
const send = document.getElementById("send");
const composer = document.getElementById("composer");

let snapshot = null; // last committed session revision, kept during recovery
let bufferedTurn = null; // transient answer identity awaiting confirmation
let bufferedText = "";
let discoveryStarted = false; // one discovery read per buffered identity
let postPending = false;
let retained = null; // {id, text} kept after an uncertain POST for retry
let source = null; // null means connection state unknown
let timer = null; // at most one reconnect timer
let fetchNumber = 0;
let notice = "";

function disabled() {
  const buffered = bufferedTurn !== null;
  const confirmed =
    snapshot !== null && snapshot.turn !== null && snapshot.turn.id === bufferedTurn;
  return (
    postPending ||
    snapshot === null ||
    source === null ||
    snapshot.turn !== null ||
    (buffered && !confirmed)
  );
}

function render() {
  conversation.textContent = "";
  if (snapshot !== null) {
    for (const message of snapshot.messages) {
      conversation.append(answerLine(message.role, message.text));
    }
    if (
      bufferedTurn !== null &&
      snapshot.turn !== null &&
      snapshot.turn.id === bufferedTurn &&
      bufferedText !== ""
    ) {
      conversation.append(answerLine("assistant", bufferedText));
    }
  }
  if (notice !== "") {
    statusLine.textContent = notice;
  } else if (snapshot !== null && snapshot.turn !== null) {
    statusLine.textContent = "running";
  } else if (snapshot !== null) {
    statusLine.textContent = snapshot.outcome === null ? "" : snapshot.outcome;
  } else {
    statusLine.textContent = "";
  }
  const locked = disabled();
  question.disabled = locked;
  send.disabled = locked;
}

function answerLine(role, text) {
  const line = document.createElement("p");
  line.className = "message " + role;
  const who = document.createElement("strong");
  who.textContent = role === "user" ? "You" : "Agent";
  line.append(who, document.createTextNode(" " + text));
  return line;
}

function applySnapshot(next) {
  snapshot = next;
  if (
    bufferedTurn !== null &&
    (snapshot.turn === null || snapshot.turn.id !== bufferedTurn)
  ) {
    bufferedTurn = null;
    bufferedText = "";
    discoveryStarted = false;
  }
}

function scheduleReconnect() {
  if (timer === null) {
    timer = setTimeout(() => {
      timer = null;
      fetchSnapshot();
    }, 1000);
  }
}

function recover() {
  if (source !== null) {
    source.close();
    source = null;
  }
  scheduleReconnect();
}

async function fetchSnapshot() {
  const number = ++fetchNumber;
  let body = null;
  try {
    const response = await fetch("/api/session");
    if (!response.ok) throw new Error("snapshot rejected");
    body = await response.json();
  } catch (error) {
    if (number !== fetchNumber) {
      // Superseded while there is still no current EventSource: keep the
      // same close/reconnect cycle instead of fetching again right away.
      if (source === null) scheduleReconnect();
      return;
    }
    notice = "reconnecting";
    recover();
    render();
    return;
  }
  if (number !== fetchNumber) return;
  notice = "";
  applySnapshot(body);
  connect();
  render();
}

function connect() {
  if (source !== null) return;
  const stream = new EventSource("/api/events");
  const live = () => source === stream;
  stream.onopen = () => {
    // The server subscribed before these headers; read the guarantee snapshot.
    if (live()) fetchSnapshot();
  };
  stream.onerror = () => {
    if (live()) {
      notice = "reconnecting";
      recover();
      render();
    }
  };
  stream.addEventListener("delta", (event) => {
    if (live()) acceptDelta(event);
  });
  stream.addEventListener("turn_end", () => {
    // A terminal event starts a new numbered fetch, invalidating older ones.
    if (live()) fetchSnapshot();
  });
  source = stream;
}

function acceptDelta(event) {
  const data = JSON.parse(event.data);
  if (data.turn_id !== bufferedTurn) {
    bufferedTurn = data.turn_id;
    bufferedText = "";
    discoveryStarted = false;
  }
  bufferedText += data.text;
  const confirmed =
    snapshot !== null && snapshot.turn !== null && snapshot.turn.id === bufferedTurn;
  if (!confirmed && !discoveryStarted) {
    discoveryStarted = true;
    fetchSnapshot();
  }
  render();
}

async function submit() {
  if (disabled()) return;
  const text = question.value;
  const id =
    retained !== null && retained.text === text ? retained.id : crypto.randomUUID();
  retained = { id, text };
  postPending = true;
  render();
  let body = null;
  try {
    const response = await fetch("/api/message", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ request_id: id, text }),
    });
    if (!response.ok) throw new Error("message rejected");
    body = await response.json();
  } catch (error) {
    postPending = false;
    notice = "Submission could not be confirmed.";
    fetchSnapshot();
    render();
    return;
  }
  postPending = false;
  retained = null;
  question.value = "";
  // SSE and discovery snapshots may have buffered text (or a newer turn may
  // have started) while this reply was delayed; only an empty buffer takes
  // this reply's identity, and the numbered snapshot stays authoritative.
  if (bufferedTurn === null) {
    bufferedTurn = body.turn_id;
    bufferedText = "";
    discoveryStarted = false;
  }
  fetchSnapshot();
  render();
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submit();
});

render(); // apply the disabled predicate while the first snapshot is unresolved
fetchSnapshot();
