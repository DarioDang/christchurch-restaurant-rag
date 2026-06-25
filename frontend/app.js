const API_BASE = 'http://localhost:8000'; // change when deploying to Render

let conversation = [];
let isStreaming = false;

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('composer');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const statsEl = document.getElementById('stats');

init();

async function init() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const data = await res.json();
    statsEl.textContent = `${data.restaurants} restaurants · ${data.reviews.toLocaleString()} reviews`;
  } catch (e) {
    statsEl.textContent = '';
  }
  loadExamples();
}

async function loadExamples() {
  try {
    const res = await fetch(`${API_BASE}/api/examples`);
    const data = await res.json();
    renderExamples(data.examples);
  } catch (e) {
    // examples are a nice-to-have; fail silently
  }
}

function renderExamples(examples) {
  const wrap = document.createElement('div');
  wrap.className = 'examples';
  wrap.innerHTML = `<p class="examples__label">Try asking</p>`;
  examples.forEach((q) => {
    const btn = document.createElement('button');
    btn.className = 'examples__chip';
    btn.type = 'button';
    btn.textContent = q;
    btn.addEventListener('click', () => sendMessage(q));
    wrap.appendChild(btn);
  });
  chatEl.appendChild(wrap);
}

formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;
  inputEl.value = '';
  sendMessage(text);
});

function sendMessage(text) {
  clearExamples();
  appendUserBubble(text);
  conversation.push({ role: 'user', content: text });
  streamAssistantReply();
}

function clearExamples() {
  const ex = chatEl.querySelector('.examples');
  if (ex) ex.remove();
}

function appendUserBubble(text) {
  const bubble = document.createElement('div');
  bubble.className = 'msg msg--user';
  bubble.textContent = text;
  chatEl.appendChild(bubble);
  scrollToBottom();
}

function createAssistantBubble() {
  const bubble = document.createElement('div');
  bubble.className = 'msg msg--assistant';
  bubble.innerHTML = '<span class="msg__cursor"></span>';
  chatEl.appendChild(bubble);
  scrollToBottom();
  return bubble;
}

function createTraceStrip(query) {
  const strip = document.createElement('div');
  strip.className = 'trace';
  strip.innerHTML = `
    <span class="trace__dot"></span>
    <span>searching</span>
    <span class="trace__query">"${escapeHtml(query)}"</span>
  `;
  chatEl.appendChild(strip);
  scrollToBottom();
  return strip;
}

function finishTraceStrip(strip, mode, count) {
  strip.classList.add('trace--done');
  strip.innerHTML = `
    <span class="trace__dot trace__dot--done"></span>
    <span>found</span>
    <span class="trace__query">${count} result${count === 1 ? '' : 's'} · ${mode}</span>
  `;
}

function setStreaming(state) {
  isStreaming = state;
  sendBtn.disabled = state;
  inputEl.disabled = state;
}

async function streamAssistantReply() {
  setStreaming(true);
  let assistantBubble = null;
  let traceStrip = null;
  let buffer = '';

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversation }),
    });

    if (!res.ok) {
      appendSystemError('The server could not process that request. Please try again.');
      setStreaming(false);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (!rawEvent.startsWith('data: ')) continue;
        handleEvent(JSON.parse(rawEvent.slice(6)));
      }
    }
  } catch (e) {
    appendSystemError('Lost connection while generating a response.');
  } finally {
    setStreaming(false);
  }

  function handleEvent(event) {
    if (event.type === 'tool_call_start') {
      traceStrip = createTraceStrip(event.query);
    } else if (event.type === 'tool_call_end') {
      if (traceStrip) finishTraceStrip(traceStrip, event.mode, event.result_count);
    } else if (event.type === 'text_delta') {
      if (!assistantBubble) assistantBubble = createAssistantBubble();
      const cursor = assistantBubble.querySelector('.msg__cursor');
      assistantBubble.insertBefore(document.createTextNode(event.content), cursor);
      scrollToBottom();
    } else if (event.type === 'error') {
      appendSystemError(event.detail || 'Something went wrong.');
    } else if (event.type === 'done') {
      conversation = event.messages;
      const cursor = assistantBubble?.querySelector('.msg__cursor');
      if (cursor) cursor.remove();
    }
  }
}

function appendSystemError(text) {
  const el = document.createElement('div');
  el.className = 'msg msg--error';
  el.textContent = text;
  chatEl.appendChild(el);
  scrollToBottom();
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}