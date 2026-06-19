/* ═══════════════════════════════════════════════
   UIT Academic Policies Chatbot — app.js
   ═══════════════════════════════════════════════ */

// ── Configure marked.js ──
marked.setOptions({
  breaks: true,
  gfm: true,
});

// ── State ──
let sessionId   = getOrCreateSessionId();
let isLoading   = false;
let sidebarOpen = window.innerWidth > 680;

// ── DOM refs ──
const messagesArea   = document.getElementById('messages-area');
const messageList    = document.getElementById('message-list');
const welcomeScreen  = document.getElementById('welcome-screen');
const typingIndicator = document.getElementById('typing-indicator');
const chatInput      = document.getElementById('chat-input');
const sendBtn        = document.getElementById('send-btn');
const sessionDisplay = document.getElementById('session-display');
const articleInput   = document.getElementById('article-input');
const articleResult  = document.getElementById('article-result');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const toast          = document.getElementById('toast');

// ── Init ──
(function init() {
  sessionDisplay.textContent = sessionId.slice(-8);
  updateSidebarState();
  checkIndexStatus();
  setupInputListeners();
})();

// ── Session management ──
function getOrCreateSessionId() {
  let id = sessionStorage.getItem('uit_session_id');
  if (!id) {
    id = 'session-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7);
    sessionStorage.setItem('uit_session_id', id);
  }
  return id;
}

function startNewChat() {
  sessionStorage.removeItem('uit_session_id');
  sessionId = getOrCreateSessionId();
  sessionDisplay.textContent = sessionId.slice(-8);
  messageList.innerHTML = '';
  welcomeScreen.style.display = '';
  showToast('Bắt đầu cuộc hội thoại mới', 'success');
}

// ── Health / index status check ──
async function checkIndexStatus() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.index_ready) {
      statusDot.className = 'status-dot ready';
      statusText.textContent = 'Index sẵn sàng';
    } else {
      statusDot.className = 'status-dot error';
      statusText.textContent = 'Chưa có index — chạy make build-index';
    }
  } catch {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Không kết nối được';
  }
}

// ── Input listeners ──
function setupInputListeners() {
  chatInput.addEventListener('input', () => {
    // Auto-resize textarea
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    // Enable/disable send button
    sendBtn.disabled = !chatInput.value.trim();
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled && !isLoading) handleSend();
    }
  });

  articleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleArticleSearch();
  });
}

// ── Send message ──
async function handleSend() {
  const question = chatInput.value.trim();
  if (!question || isLoading) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;

  hideWelcome();
  appendUserMessage(question);
  showTyping();
  scrollToBottom();

  isLoading = true;
  try {
    const res = await fetch('/api/v1/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hideTyping();
    appendAssistantMessage(data.answer, data.sources);
  } catch (err) {
    hideTyping();
    appendAssistantMessage(
      `⚠️ Có lỗi xảy ra: ${err.message}. Vui lòng thử lại.`,
      ''
    );
    showToast(err.message, 'error');
  } finally {
    isLoading = false;
    scrollToBottom();
  }
}

// ── Suggested question ──
function askSuggestion(btn) {
  const text = btn.textContent.replace(/^[\p{Emoji}\s]+/u, '').trim();
  chatInput.value = text;
  chatInput.dispatchEvent(new Event('input'));
  handleSend();
}

// ── Article search ──
async function handleArticleSearch() {
  const val = articleInput.value.trim();
  if (!val) return;

  articleResult.classList.remove('hidden');
  articleResult.innerHTML = `<span style="color:var(--text-3)">Đang tìm Điều ${val}...</span>`;

  try {
    const res = await fetch(`/api/v1/rag/search?article=${encodeURIComponent(val)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    articleResult.innerHTML = marked.parse(data.content || `Không tìm thấy Điều ${val}.`);
  } catch (err) {
    articleResult.innerHTML = `<span style="color:var(--error)">Lỗi: ${err.message}</span>`;
  }
}

// ── DOM helpers ──
function hideWelcome() {
  if (welcomeScreen) {
    welcomeScreen.style.display = 'none';
  }
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="message-avatar user">Bạn</div>
    <div class="message-bubble-wrap">
      <div class="message-bubble user">${escapeHtml(text)}</div>
      <div class="message-time">${formatTime()}</div>
    </div>`;
  messageList.appendChild(row);
}

function appendAssistantMessage(answer, sources) {
  const row = document.createElement('div');
  row.className = 'message-row assistant';

  const sourceHtml = buildSourcesHtml(sources);

  row.innerHTML = `
    <div class="message-avatar assistant">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <rect width="14" height="14" rx="4" fill="url(#mlg${Date.now()})"/>
        <path d="M3 5h8M3 8h5" stroke="#fff" stroke-width="1.2" stroke-linecap="round"/>
        <defs>
          <linearGradient id="mlg${Date.now()}" x1="0" y1="0" x2="14" y2="14">
            <stop offset="0%" stop-color="#4f6ef7"/>
            <stop offset="100%" stop-color="#7c3aed"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    <div class="message-bubble-wrap">
      <div class="message-bubble assistant">${marked.parse(answer)}</div>
      ${sourceHtml}
      <div class="message-time">${formatTime()}</div>
    </div>`;

  messageList.appendChild(row);

  // Wire up toggle
  const toggle = row.querySelector('.sources-toggle');
  const content = row.querySelector('.sources-content');
  if (toggle && content) {
    toggle.addEventListener('click', () => {
      const isOpen = content.classList.toggle('open');
      toggle.classList.toggle('open', isOpen);
    });
  }
}

function buildSourcesHtml(sources) {
  if (!sources || !sources.trim()) return '';
  return `
    <button class="sources-toggle">
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
        <path d="M3 5l4-3v6L3 5z" fill="currentColor"/>
      </svg>
      Nguồn tham khảo
    </button>
    <div class="sources-content">${marked.parse(sources)}</div>`;
}

function showTyping() {
  typingIndicator.classList.remove('hidden');
  scrollToBottom();
}
function hideTyping() {
  typingIndicator.classList.add('hidden');
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesArea.scrollTop = messagesArea.scrollHeight;
  });
}

// ── Sidebar toggle ──
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (window.innerWidth <= 680) {
    sidebar.classList.toggle('open');
  } else {
    sidebarOpen = !sidebarOpen;
    updateSidebarState();
  }
}

function updateSidebarState() {
  const sidebar = document.getElementById('sidebar');
  if (window.innerWidth > 680) {
    sidebar.classList.toggle('collapsed', !sidebarOpen);
  }
}

// ── Toast ──
let toastTimer = null;
function showToast(msg, type = '') {
  toast.textContent = msg;
  toast.className = `toast${type ? ' ' + type : ''}`;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
}

// ── Utilities ──
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatTime() {
  return new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}
