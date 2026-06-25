const API_BASE = 'http://localhost:8000'; // change when deploying to Render

let conversation = [];
let isStreaming = false;
let userLocation = { lat: null, lon: null, enabled: false };
let maxDistanceKm = 5;
let lastUserQuery = null;
let autoLocationAttempted = false;

const headerEl = document.getElementById('header');
const chatEl = document.getElementById('chat');
const jumpBtn = document.getElementById('jump-btn');
const formEl = document.getElementById('composer');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const statsEl = document.getElementById('stats');

const locationBtn = document.getElementById('location-btn');
const locationPanel = document.getElementById('location-panel');
const locationToggle = document.getElementById('location-toggle');
const locationDistanceWrap = document.getElementById('location-distance');
const distanceSlider = document.getElementById('distance-slider');
const distanceValue = document.getElementById('distance-value');
const locationStatus = document.getElementById('location-status');

const CUISINES = ['Chinese', 'Japanese', 'Italian', 'Thai', 'Indian', 'Vietnamese', 'Korean', 'American'];
const SERVICES = [
  { label: 'Delivery', query: 'Restaurants with delivery' },
  { label: 'Takeout', query: 'Restaurants with takeout' },
  { label: 'Budget-friendly', query: 'Cheap restaurants' },
  { label: 'Open late', query: 'Restaurants open late' },
];

init();

async function init() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const data = await res.json();
    statsEl.classList.remove('skeleton');
    statsEl.textContent = `${data.restaurants} restaurants · ${data.reviews.toLocaleString()} reviews`;
  } catch (e) {
    statsEl.classList.remove('skeleton');
    statsEl.textContent = '';
  }
  loadSuggestions();
}

// ============================================
// SCROLL HANDLING
// ============================================

function isNearBottom() {
  return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 100;
}

function showJumpButton() {
  jumpBtn.hidden = false;
}
function hideJumpButton() {
  jumpBtn.hidden = true;
}

function appendToChat(el) {
  const wasNearBottom = isNearBottom();
  chatEl.appendChild(el);
  if (wasNearBottom) {
    chatEl.scrollTop = chatEl.scrollHeight;
  } else {
    showJumpButton();
  }
}

function maybeAutoScroll() {
  if (isNearBottom()) chatEl.scrollTop = chatEl.scrollHeight;
}

jumpBtn.addEventListener('click', () => {
  chatEl.scrollTop = chatEl.scrollHeight;
  hideJumpButton();
});

chatEl.addEventListener('scroll', () => {
  headerEl.classList.toggle('header--scrolled', chatEl.scrollTop > 4);
  if (isNearBottom()) hideJumpButton();
});

function formatTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

// ============================================
// MARKDOWN (minimal, safe subset — bold only)
// ============================================

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderMarkdownInline(raw) {
  const escaped = escapeHtml(raw);
  return escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

// ============================================
// SUGGESTIONS
// ============================================

async function loadSuggestions() {
  const [examplesRes, popularRes] = await Promise.allSettled([
    fetch(`${API_BASE}/api/examples`).then((r) => r.json()),
    fetch(`${API_BASE}/api/popular`).then((r) => r.json()),
  ]);

  const wrap = document.createElement('div');
  wrap.className = 'suggestions';
  wrap.innerHTML = `<p class="suggestions__intro">Ask about restaurants, hours, or find something nearby.</p>`;

  if (examplesRes.status === 'fulfilled') {
    wrap.appendChild(buildChipSection('Try asking', examplesRes.value.examples, (q) => q));
  }

  if (popularRes.status === 'fulfilled' && popularRes.value.trending?.length) {
    const label = popularRes.value.show_count ? 'Popular right now' : 'Popular categories';
    const queries = popularRes.value.trending.map((t) => t.example_query);
    wrap.appendChild(buildChipSection(label, queries, (q) => q));
  }

  wrap.appendChild(buildChipSection('By cuisine', CUISINES, (c) => `Best ${c} restaurants`));
  wrap.appendChild(
    buildChipSection('By service', SERVICES.map((s) => s.label), (label) =>
      SERVICES.find((s) => s.label === label).query
    )
  );

  chatEl.appendChild(wrap);
}

function buildChipSection(label, items, toQuery) {
  const section = document.createElement('div');
  section.className = 'suggestions__section';
  section.innerHTML = `<p class="suggestions__label">${label}</p>`;
  const row = document.createElement('div');
  row.className = 'suggestions__row';
  items.forEach((item) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'examples__chip';
    btn.textContent = item;
    btn.addEventListener('click', () => sendMessage(toQuery(item)));
    row.appendChild(btn);
  });
  section.appendChild(row);
  return section;
}

function clearExamples() {
  const ex = chatEl.querySelector('.suggestions');
  if (ex) ex.remove();
}

// ============================================
// LOCATION PANEL
// ============================================

function openLocationPanel() {
  locationPanel.hidden = false;
  requestAnimationFrame(() => locationPanel.classList.add('location__panel--visible'));
  locationBtn.setAttribute('aria-expanded', 'true');
}

function attemptAutoLocation() {
  acquireLocation()
    .then(() => {
      appendSystemNote('📍 Using your location to search again…');
      if (lastUserQuery) {
        setTimeout(() => sendMessage(lastUserQuery, { isRetry: true }), 500);
      }
    })
    .catch(() => {
    });
}

function closeLocationPanel() {
  locationPanel.classList.remove('location__panel--visible');
  locationBtn.setAttribute('aria-expanded', 'false');
  setTimeout(() => { locationPanel.hidden = true; }, 160);
}

locationBtn.addEventListener('click', (e) => {
  e.stopPropagation(); 
  if (locationPanel.hidden) openLocationPanel();
  else closeLocationPanel();
});

document.addEventListener('click', (e) => {
  if (!locationPanel.hidden && !locationPanel.contains(e.target) && !locationBtn.contains(e.target)) {
    closeLocationPanel();
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !locationPanel.hidden) {
    closeLocationPanel();
    locationBtn.focus();
  }
});

function showDistanceControl() {
  locationDistanceWrap.hidden = false;
  requestAnimationFrame(() => locationDistanceWrap.classList.add('location__distance--visible'));
}

function hideDistanceControl() {
  locationDistanceWrap.classList.remove('location__distance--visible');
  setTimeout(() => { locationDistanceWrap.hidden = true; }, 160);
}

locationToggle.addEventListener('change', () => {
  if (locationToggle.checked) {
    locationStatus.textContent = 'Requesting location…';
    acquireLocation().catch(() => { locationToggle.checked = false; });
  } else {
    userLocation = { lat: null, lon: null, enabled: false };
    hideDistanceControl();
    locationStatus.textContent = '';
  }
});

distanceSlider.addEventListener('input', () => {
  maxDistanceKm = Number(distanceSlider.value);
  distanceValue.textContent = maxDistanceKm;
});

// ============================================
// COMPOSER (auto-resize + Enter/Shift+Enter)
// ============================================

function autoResizeInput() {
  inputEl.style.height = 'auto';
  const contentHeight = inputEl.scrollHeight;
  const capped = Math.min(contentHeight, 120);
  inputEl.style.height = capped + 'px';
  inputEl.style.overflowY = contentHeight > 120 ? 'auto' : 'hidden';
}

inputEl.addEventListener('input', autoResizeInput);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;
  inputEl.value = '';
  autoResizeInput();
  sendMessage(text);
});

// ============================================
// CHAT
// ============================================

function sendMessage(text, { isRetry = false } = {}) {
  if (!isRetry) {
    lastUserQuery = text;
    autoLocationAttempted = false;
  }
  clearExamples();
  appendUserBubble(text);
  conversation.push({ role: 'user', content: text });
  streamAssistantReply();
}

function acquireLocation() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation not supported'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        userLocation = { lat: pos.coords.latitude, lon: pos.coords.longitude, enabled: true };
        locationToggle.checked = true;
        showDistanceControl();
        locationStatus.textContent = `Location set (±${Math.round(pos.coords.accuracy)}m accuracy)`;
        resolve(userLocation);
      },
      (err) => {
        locationStatus.textContent = 'Could not get your location. Check browser permissions.';
        reject(err);
      }
    );
  });
}

function appendSystemNote(text) {
  const el = document.createElement('div');
  el.className = 'msg msg--system';
  el.textContent = text;
  appendToChat(el);
}

function appendUserBubble(text) {
  const group = document.createElement('div');
  group.className = 'msg-group msg-group--user';
  const bubble = document.createElement('div');
  bubble.className = 'msg msg--user';
  bubble.textContent = text;
  const time = document.createElement('span');
  time.className = 'msg__time';
  time.textContent = formatTime();
  group.appendChild(bubble);
  group.appendChild(time);
  appendToChat(group);
}

function showThinkingIndicator() {
  const el = document.createElement('div');
  el.className = 'thinking';
  el.id = 'thinking-indicator';
  el.innerHTML = '<span class="thinking__dot"></span><span class="thinking__dot"></span><span class="thinking__dot"></span>';
  appendToChat(el);
  return el;
}

function removeThinkingIndicator() {
  const el = document.getElementById('thinking-indicator');
  if (el) el.remove();
}

function createAssistantBubble() {
  const group = document.createElement('div');
  group.className = 'msg-group msg-group--assistant';
  const bubble = document.createElement('div');
  bubble.className = 'msg msg--assistant';
  bubble.innerHTML = '<span class="msg__cursor"></span>';
  group.appendChild(bubble);
  appendToChat(group);
  return bubble;
}

function renderAssistantText(bubble, raw) {
  bubble.innerHTML = renderMarkdownInline(raw) + '<span class="msg__cursor"></span>';
}

function createTraceStrip(query) {
  const strip = document.createElement('div');
  strip.className = 'trace';
  strip.innerHTML = `
    <span class="trace__dot"></span>
    <span>searching</span>
    <span class="trace__query">"${escapeHtml(query)}"</span>
  `;
  appendToChat(strip);
  return strip;
}

function finishTraceStrip(strip, mode, count) {
  strip.classList.add('trace--done');
  strip.innerHTML = `
    <span class="trace__dot trace__dot--done"></span>
    <span>found</span>
    <span class="trace__query">${count} result${count === 1 ? '' : 's'} · ${mode}</span>
  `;
  maybeAutoScroll();
}

function setStreaming(state) {
  isStreaming = state;
  sendBtn.disabled = state;
  inputEl.disabled = state;
}

async function streamAssistantReply() {
  setStreaming(true);
  showThinkingIndicator();

  let assistantBubble = null;
  let assistantRawText = '';
  let traceStrip = null;
  let buffer = '';
  let firstEventReceived = false;
  let pendingLocationRetry = false;

  function clearThinkingOnce() {
    if (!firstEventReceived) {
      firstEventReceived = true;
      removeThinkingIndicator();
    }
  }

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: conversation,
        user_lat: userLocation.enabled ? userLocation.lat : null,
        user_lon: userLocation.enabled ? userLocation.lon : null,
        max_distance_km: maxDistanceKm,
      }),
    });

    if (!res.ok) {
      removeThinkingIndicator();
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
    removeThinkingIndicator();
    appendSystemError('Lost connection while generating a response.');
  } finally {
    setStreaming(false);
  }

  function handleEvent(event) {
    clearThinkingOnce();

    if (event.type === 'tool_call_start') {
      traceStrip = createTraceStrip(event.query);
    } else if (event.type === 'tool_call_end') {
      if (traceStrip) finishTraceStrip(traceStrip, event.mode, event.result_count);
      if (event.mode === 'location_required' && !userLocation.enabled && !autoLocationAttempted) {
        autoLocationAttempted = true;
        pendingLocationRetry = true;
      }
    } else if (event.type === 'text_delta') {
      if (!assistantBubble) assistantBubble = createAssistantBubble();
      assistantRawText += event.content;
      renderAssistantText(assistantBubble, assistantRawText);
      maybeAutoScroll();
    } else if (event.type === 'error') {
      appendSystemError(event.detail || 'Something went wrong.');
    } else if (event.type === 'done') {
      conversation = event.messages;
      if (assistantBubble) {
        renderAssistantText(assistantBubble, assistantRawText);
        const cursor = assistantBubble.querySelector('.msg__cursor');
        if (cursor) cursor.remove();
        assistantBubble.classList.add('msg--complete');
        const group = assistantBubble.parentElement;
        const time = document.createElement('span');
        time.className = 'msg__time';
        time.textContent = formatTime();
        group.appendChild(time);
      }
    }
  }

  if (pendingLocationRetry) {
    pendingLocationRetry = false;
    attemptAutoLocation();
  }
}

function appendSystemError(text) {
  const el = document.createElement('div');
  el.className = 'msg msg--error';
  el.textContent = text;
  appendToChat(el);
}