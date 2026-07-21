const $ = selector => document.querySelector(selector);
const STORAGE_KEY = 'rwkv-search-chats-v2';
let chats = loadChats();
let currentChatId = localStorage.getItem('rwkv-search-current') || null;
// Search is a one-shot override for the next message.  The default remains
// Router-controlled Auto, so stable knowledge stays in ordinary RWKV chat.
let searchEnabled = false;
let runtimeModel = {enabled: false, ready: false, label: 'RWKV'};
let requestRunning = false;
let activeRequest = null;

const STATE_LABELS = {
  queued: '请求已提交',
  routing: '正在判断是否需要搜索',
  discovering: '正在发现相关来源',
  fetching: '正在抓取网页正文',
  ranking: '正在筛选可靠证据',
  generating: '正在生成回答',
  completed: '已完成',
  failed: '请求失败',
  cancelled: '已停止生成'
};

function loadChats() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(value) ? value : [];
  } catch (_) { return []; }
}
function saveChats() {
  chats = chats.slice(0, 50);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  if (currentChatId) localStorage.setItem('rwkv-search-current', currentChatId);
  else localStorage.removeItem('rwkv-search-current');
}
function id() { return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`; }
function currentChat() { return chats.find(chat => chat.id === currentChatId) || null; }
function escapeHTML(value = '') {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function safeURL(value = '') {
  try {
    const url = new URL(value, location.origin);
    return ['http:', 'https:'].includes(url.protocol) ? escapeHTML(url.href) : '#';
  } catch (_) { return '#'; }
}
function renderMarkdown(value = '') {
  let text = escapeHTML(value);
  const fences = [];
  text = text.replace(/```[^\n]*\n?([\s\S]*?)```/g, (_, code) => {
    const token = `@@CODE${fences.length}@@`;
    fences.push(`<pre><code>${code.trim()}</code></pre>`);
    return token;
  });
  text = text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\[S(\d+)\]/g, '<button class="citation" data-cite="S$1">S$1</button>');
  const blocks = text.split(/\n{2,}/).map(block => {
    const lines = block.split('\n');
    if (lines.every(line => /^\s*[-*]\s+/.test(line))) {
      return `<ul>${lines.map(line => `<li>${line.replace(/^\s*[-*]\s+/, '')}</li>`).join('')}</ul>`;
    }
    return block.startsWith('@@CODE') ? block : `<p>${block.replace(/\n/g, '<br>')}</p>`;
  });
  text = blocks.join('');
  fences.forEach((fence, index) => { text = text.replace(`<p>@@CODE${index}@@</p>`, fence).replace(`@@CODE${index}@@`, fence); });
  return text;
}
function formatTime(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString([], {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'});
}

function createChat(firstMessage = '') {
  const chat = {id: id(), title: firstMessage.slice(0, 28) || '新对话', createdAt: Date.now(), updatedAt: Date.now(), messages: []};
  chats.unshift(chat);
  currentChatId = chat.id;
  saveChats();
  return chat;
}
function newChat() {
  currentChatId = null;
  saveChats();
  render();
  $('#queryInput').focus();
  closeSidebar();
}
function selectChat(chatId) {
  currentChatId = chatId;
  saveChats();
  render();
  closeSidebar();
}
function deleteChat(chatId) {
  chats = chats.filter(chat => chat.id !== chatId);
  if (currentChatId === chatId) currentChatId = chats[0]?.id || null;
  saveChats();
  render();
}

function renderHistory() {
  const history = $('#chatHistory');
  if (!chats.length) {
    history.innerHTML = '<div class="history-label">还没有对话</div>';
    return;
  }
  history.innerHTML = chats.map(chat => `<button class="history-item ${chat.id === currentChatId ? 'active' : ''}" type="button" data-chat="${escapeHTML(chat.id)}">
    <span class="history-title">${escapeHTML(chat.title)}</span>
    <span class="history-delete" role="button" aria-label="删除对话" data-delete="${escapeHTML(chat.id)}">×</span>
  </button>`).join('');
}
function sourceHTML(source) {
  const sourceId = source.id || `D${source.document_id || ''}`;
  const sourceKind = source.source_type === 'finewiki' ? '<span class="source-kind">FineWiki</span>' : '';
  return `<a class="source-card" data-source-id="${escapeHTML(sourceId)}" href="${safeURL(source.url)}" target="_blank" rel="noopener noreferrer">
    <strong><span class="source-id">${escapeHTML(sourceId)}</span>${sourceKind}${escapeHTML(source.title || source.url || '来源')}</strong>
    <p>${escapeHTML(source.snippet || source.text || '')}</p>
  </a>`;
}
function assistantHTML(message) {
  const inProgress = message.pending || ['queued', 'routing', 'discovering', 'fetching', 'ranking', 'generating'].includes(message.state);
  if (inProgress && !message.content) {
    return `<article class="message assistant" data-message="${message.id}">
      <div class="assistant-avatar">R</div><div class="assistant-content"><div class="assistant-name">RWKV Search</div><div class="thinking">${escapeHTML(message.status || STATE_LABELS[message.state] || '正在思考')}</div>${message.progress ? `<div class="request-progress">${escapeHTML(message.progress)}</div>` : ''}</div>
    </article>`;
  }
  const sources = message.sources || [];
  const meta = [];
  if (message.searchEnabled) meta.push('FineWiki + 联网搜索');
  if (message.model?.used) {
    meta.push(`${message.model.new_tokens || 0} tokens`);
    meta.push(`${Math.round(message.model.latency_ms || 0)} ms`);
    if (message.model.structured === false) meta.push('自由对话');
    if (message.model.repaired) meta.push('已修复结构');
  } else if (message.model?.error) meta.push('模型已降级');
  if (message.usage?.total_ms) meta.push(`总耗时 ${Math.round(message.usage.total_ms)} ms`);
  if (message.usage?.evidence_sources) meta.push(`${message.usage.evidence_sources} 条证据`);
  if (message.dataTime) meta.push(`数据 ${formatTime(message.dataTime)}`);
  const fallbackLabel = message.route?.intent === 'chat' ? '对话降级' : '抽取降级';
  const modelChip = message.model?.used ? '<span class="model-chip">RWKV 生成</span>' : (message.model?.error ? `<span class="model-chip">${fallbackLabel}</span>` : '');
  const stateChip = message.state === 'cancelled' ? '<span class="state-chip cancelled">已停止</span>' : (message.state === 'failed' ? '<span class="state-chip failed">失败</span>' : '');
  return `<article class="message assistant" data-message="${message.id}">
    <div class="assistant-avatar">R</div>
    <div class="assistant-content">
      <div class="assistant-name">RWKV Search ${modelChip}${stateChip}</div>
      <div class="answer-text">${renderMarkdown(message.content || '')}</div>
      ${sources.length ? `<details class="sources"><summary>${sources.length} 个参考来源</summary><div class="source-grid">${sources.map(sourceHTML).join('')}</div></details>` : ''}
      <div class="message-tools"><button type="button" data-copy="${message.id}">复制</button>${meta.length ? `<span>${escapeHTML(meta.join(' · '))}</span>` : ''}</div>
    </div>
  </article>`;
}
function renderMessages() {
  const chat = currentChat();
  const messages = chat?.messages || [];
  $('#emptyChat').classList.toggle('hidden', messages.length > 0);
  $('#messageList').innerHTML = messages.map(message => message.role === 'user'
    ? `<article class="message user"><div class="user-bubble">${escapeHTML(message.content)}</div></article>`
    : assistantHTML(message)).join('');
  $('#currentChatTitle').textContent = chat?.title || '新对话';
  if (messages.length) requestAnimationFrame(() => { $('#conversation').scrollTop = $('#conversation').scrollHeight; });
}
function renderSearchToggle() {
  const button = $('#searchToggle');
  button.classList.toggle('active', searchEnabled);
  button.setAttribute('aria-pressed', String(searchEnabled));
  button.setAttribute('aria-label', searchEnabled ? '取消本次搜索' : '仅下一条消息使用搜索');
  button.textContent = searchEnabled ? '⌁ 本次搜索' : '⌁ 搜索';
}
function render() { renderHistory(); renderMessages(); renderSearchToggle(); }

function parseSSEBlock(block) {
  const lines = block.split('\n').filter(item => item.startsWith('data:'));
  if (!lines.length) return null;
  try { return JSON.parse(lines.map(line => line.slice(5).trimStart()).join('\n')); } catch (_) { return null; }
}
function updateAssistant(chatId, messageId, mutate) {
  const chat = chats.find(item => item.id === chatId);
  const message = chat?.messages.find(item => item.id === messageId);
  if (!message) return;
  mutate(message, chat);
  chat.updatedAt = Date.now();
  saveChats();
  if (currentChatId === chatId) renderMessages();
}
async function ask(prefill = '') {
  const input = $('#queryInput');
  const query = (prefill || input.value).trim();
  if (!query || requestRunning) return;
  const forceSearch = searchEnabled;
  let chat = currentChat();
  if (!chat) chat = createChat(query);
  const history = chat.messages
    .filter(message => !message.pending && ['user', 'assistant'].includes(message.role))
    .filter(message => message.role !== 'assistant' || (
      message.state !== 'failed' &&
      message.state !== 'cancelled' &&
      message.model?.used !== false &&
      !message.insufficientEvidence
    ))
    .map(message => ({role: message.role, content: message.content}))
    .slice(-10);
  const userMessage = {id: id(), role: 'user', content: query, createdAt: Date.now()};
  const assistantMessage = {id: id(), role: 'assistant', content: '', pending: true, state: 'queued', status: STATE_LABELS.queued, sources: [], searchEnabled: forceSearch, createdAt: Date.now()};
  chat.messages.push(userMessage, assistantMessage);
  chat.updatedAt = Date.now();
  if (chat.messages.filter(message => message.role === 'user').length === 1) chat.title = query.slice(0, 28);
  const chatId = chat.id;
  const assistantId = assistantMessage.id;
  input.value = '';
  input.style.height = 'auto';
  requestRunning = true;
  const requestId = `req_${id()}`;
  const controller = new AbortController();
  activeRequest = {requestId, chatId, assistantId, controller};
  // Consume the manual override before starting the request.  Subsequent
  // messages immediately return to Auto even while this request is running.
  searchEnabled = false;
  setComposerRunning(true);
  saveChats();
  render();

  try {
    const modeFields = requestModeFields(forceSearch);
    const response = await fetch('/api/v1/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: controller.signal,
      body: JSON.stringify({
        schema_version: '1.0',
        request_id: requestId,
        conversation_id: chatId,
        message_id: assistantId,
        query,
        history,
        ...modeFields,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Shanghai',
        locale: navigator.language || 'zh-CN'
      })
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';
      for (const block of blocks) {
        const event = parseSSEBlock(block);
        if (event) handleEvent(chatId, assistantId, event);
      }
      if (done) break;
    }
  } catch (error) {
    updateAssistant(chatId, assistantId, message => {
      message.pending = false;
      if (error.name === 'AbortError') {
        message.state = 'cancelled';
        message.status = STATE_LABELS.cancelled;
        if (!message.content) message.content = '已停止生成。';
      } else {
        message.state = 'failed';
        message.content = `请求失败：${error.message || String(error)}`;
      }
    });
  } finally {
    requestRunning = false;
    if (activeRequest?.requestId === requestId) activeRequest = null;
    setComposerRunning(false);
    $('#queryInput').focus();
  }
}
function requestModeFields(forceSearch = false) {
  return {
    search_mode: forceSearch ? 'always' : 'auto',
    research_depth: 'fast',
    source_scope: 'auto',
    use_finewiki: forceSearch
  };
}
function setComposerRunning(running) {
  const button = $('#askButton');
  button.disabled = false;
  button.classList.toggle('stop', running);
  button.textContent = running ? '■' : '↑';
  button.setAttribute('aria-label', running ? '停止生成' : '发送');
}
async function cancelActiveRequest() {
  const current = activeRequest;
  if (!current) return;
  updateAssistant(current.chatId, current.assistantId, message => {
    message.status = '正在停止生成';
  });
  try {
    await fetch(`/api/v1/requests/${encodeURIComponent(current.requestId)}/cancel`, {
      method: 'POST',
      keepalive: true
    });
  } catch (_) {}
  current.controller.abort();
}
function handleEvent(chatId, assistantId, event) {
  updateAssistant(chatId, assistantId, message => {
    if (event.type === 'request_started') {
      message.requestId = event.request_id;
      message.finewikiEnabled = Boolean(event.accepted?.use_finewiki);
      message.state = event.state || 'queued';
      message.status = STATE_LABELS[message.state];
    } else if (event.type === 'route') {
      message.route = event.route;
      message.state = 'routing';
      if (!(event.route.tools || []).length) message.status = `${runtimeModel.model || runtimeModel.label || 'RWKV'} 正在思考`;
      else message.status = event.route.depth === 'multi' ? '正在规划多轮检索' : '正在搜索相关资料';
    } else if (event.type === 'search_plan') {
      message.state = 'discovering';
      message.plan = event.plan || null;
      message.status = event.plan?.depth === 'multi' ? '正在执行多轮搜索' : '正在搜索相关资料';
    } else if (event.type === 'discovery_progress') {
      message.state = 'discovering';
      message.progress = event.progress?.message || `已发现 ${event.progress?.candidate_count || 0} 个候选来源`;
      message.status = '正在发现相关来源';
    } else if (event.type === 'fetch_progress') {
      message.state = 'fetching';
      message.progress = event.progress?.message || `已抓取 ${event.progress?.completed || 0}/${event.progress?.total || 0} 个网页`;
      message.status = STATE_LABELS.fetching;
    } else if (event.type === 'evidence_ready') {
      message.state = 'ranking';
      message.sources = event.sources || message.sources;
      message.progress = `保留 ${event.evidence_count || 0} 条可引用证据`;
      message.status = runtimeModel.ready ? `${runtimeModel.model || runtimeModel.label || 'RWKV'} 正在生成回答` : '正在整理答案';
    } else if (event.type === 'generation_started') {
      message.state = 'generating';
      message.status = STATE_LABELS.generating;
    } else if (event.type === 'answer_delta') {
      message.state = 'generating';
      message.content += event.delta || '';
    } else if (event.type === 'answer_final') {
      const answer = event.answer || {};
      message.pending = false;
      message.state = 'completed';
      message.content = answer.content || message.content || '';
      message.citations = answer.citations || [];
      message.dataTime = answer.data_time || '';
      message.insufficientEvidence = Boolean(answer.insufficient_evidence);
      message.needsClarification = Boolean(answer.needs_clarification);
      message.sources = event.sources || message.sources;
      message.usage = event.usage || null;
      message.model = event.model || null;
    } else if (event.type === 'warning') {
      const warning = event.warning || {};
      if (warning.code === 'REQUEST_CANCELLED') {
        message.pending = false;
        message.state = 'cancelled';
        message.status = STATE_LABELS.cancelled;
        if (!message.content) message.content = '已停止生成。';
      } else {
        message.progress = warning.message || '部分搜索来源暂时不可用，正在使用可用结果';
      }
    } else if (event.type === 'error') {
      message.pending = false;
      message.state = 'failed';
      message.content = `请求失败：${event.error?.message || '未知错误'}`;
    } else if (event.type === 'done' && message.pending) {
      message.pending = false;
      message.state = event.state || (message.content ? 'completed' : 'failed');
      if (!message.content && message.state === 'failed') message.content = '请求未能生成回答。';
    }
  });
}

function openSidebar() { $('#sidebar').classList.add('open'); $('#sidebarOverlay').classList.add('open'); }
function closeSidebar() { $('#sidebar').classList.remove('open'); $('#sidebarOverlay').classList.remove('open'); }

$('#newChatButton').addEventListener('click', newChat);
$('#topNewChat').addEventListener('click', newChat);
$('#openSidebar').addEventListener('click', openSidebar);
$('#closeSidebar').addEventListener('click', closeSidebar);
$('#sidebarOverlay').addEventListener('click', closeSidebar);
$('#chatHistory').addEventListener('click', event => {
  const deleteButton = event.target.closest('[data-delete]');
  if (deleteButton) { event.stopPropagation(); deleteChat(deleteButton.dataset.delete); return; }
  const item = event.target.closest('[data-chat]');
  if (item) selectChat(item.dataset.chat);
});
$('#queryInput').addEventListener('input', event => {
  event.target.style.height = 'auto';
  event.target.style.height = `${Math.min(160, event.target.scrollHeight)}px`;
});
$('#queryInput').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); ask(); }
});
$('#askButton').addEventListener('click', () => requestRunning ? cancelActiveRequest() : ask());
$('#searchToggle').addEventListener('click', () => {
  searchEnabled = !searchEnabled;
  renderSearchToggle();
});
$('.suggestions').addEventListener('click', event => {
  const button = event.target.closest('[data-question]');
  if (button) ask(button.dataset.question);
});
$('#messageList').addEventListener('click', async event => {
  const copy = event.target.closest('[data-copy]');
  if (copy) {
    const message = currentChat()?.messages.find(item => item.id === copy.dataset.copy);
    if (message?.content) { await navigator.clipboard.writeText(message.content); copy.textContent = '已复制'; setTimeout(() => { copy.textContent = '复制'; }, 1200); }
    return;
  }
  const citation = event.target.closest('[data-cite]');
  if (citation) {
    const article = citation.closest('[data-message]');
    const message = currentChat()?.messages.find(item => item.id === article?.dataset.message);
    const source = message?.sources?.find(item => (item.id || `D${item.document_id || ''}`) === citation.dataset.cite);
    if (source?.url) window.open(source.url, '_blank', 'noopener,noreferrer');
  }
});

const sourceDialog = $('#sourceDialog');
$('#openSourceButton').addEventListener('click', () => sourceDialog.showModal());
$('#crawlButton').addEventListener('click', async () => {
  const url = $('#seedUrl').value.trim();
  if (!url) return;
  $('#crawlStatus').textContent = '提交中…';
  try {
    const response = await fetch('/api/crawl', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({urls:[url], max_pages:Number($('#seedLimit').value || 50)})});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    $('#crawlStatus').textContent = '已加入抓取队列';
    setTimeout(() => sourceDialog.close(), 800);
  } catch (error) { $('#crawlStatus').textContent = `失败：${error.message}`; }
});

async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('offline');
    const data = await response.json();
    runtimeModel = data.model || runtimeModel;
    const model = $('#modelStatus');
    model.classList.remove('online', 'degraded');
    if (runtimeModel.ready) {
      model.classList.add('online');
      model.querySelector('strong').textContent = runtimeModel.label || 'RWKV';
      model.querySelector('small').textContent = `${runtimeModel.device || 'ready'} · ${runtimeModel.dtype || ''}`;
      $('#runtimeLabel').textContent = `${runtimeModel.label || 'RWKV'} · 本地搜索`;
    } else if (runtimeModel.enabled) {
      model.classList.add('degraded');
      model.querySelector('strong').textContent = 'RWKV 加载失败';
      model.querySelector('small').textContent = '已启用抽取式降级';
    } else {
      model.querySelector('strong').textContent = 'RWKV 未启用';
      model.querySelector('small').textContent = '当前使用抽取式回答';
    }
    const index = $('#indexStatus');
    index.classList.add('online');
    const web = data.realtime_search || {};
    const finewiki = data.shadow_search || {};
    index.querySelector('strong').textContent = finewiki.ready ? 'FineWiki 已就绪' : '本地索引';
    const searchParts = [];
    if (finewiki.ready) searchParts.push('FineWiki 全量索引');
    if (web.enabled) searchParts.push(web.ready ? '实时联网就绪' : '实时联网待命');
    searchParts.push(`${data.stats?.documents || 0} 个本地文档`);
    index.querySelector('small').textContent = searchParts.join(' · ');
  } catch (_) {
    $('#modelStatus').classList.add('degraded');
    $('#modelStatus small').textContent = '服务器连接失败';
    $('#indexStatus small').textContent = '离线';
  }
}

if (!chats.some(chat => chat.id === currentChatId)) currentChatId = chats[0]?.id || null;
render();
loadHealth();
$('#queryInput').focus();
