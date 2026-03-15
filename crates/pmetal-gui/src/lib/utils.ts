/**
 * Shared utility functions used across multiple pages.
 */

export function formatEta(seconds: number | null): string {
  if (seconds === null || seconds <= 0) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

export function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, exp)).toFixed(1)} ${units[exp]}`;
}

export function getStatusBadgeClass(status: string): string {
  switch (status) {
    case 'running':
    case 'training':
    case 'loading_models':
    case 'generating_signals':
      return 'badge-primary';
    case 'completed': return 'badge-success';
    case 'failed': return 'badge-danger';
    case 'cancelled': return 'badge-warning';
    default: return 'badge-neutral';
  }
}

export function runProgress(step: number, totalSteps: number): number {
  if (!totalSteps) return 0;
  return Math.min(100, (step / totalSteps) * 100);
}

/**
 * Render basic markdown to HTML. Handles:
 * - **bold**, *italic*, `code`, ```code blocks```
 * - Headers (# ## ###)
 * - Unordered lists (- or *)
 * - Ordered lists (1.)
 * - Newlines
 */
export function renderMarkdown(text: string): string {
  if (!text) return '';

  // Escape HTML entities first to prevent XSS
  const escape = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Split into lines for block-level processing
  const lines = text.split('\n');
  const result: string[] = [];
  let inCodeBlock = false;
  let codeBlockLang = '';
  let codeBlockLines: string[] = [];

  for (const line of lines) {
    // Code block fence
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        // Close code block
        result.push(
          `<pre class="code-block"><code class="language-${escape(codeBlockLang)}">${codeBlockLines
            .map(escape)
            .join('\n')}</code></pre>`
        );
        codeBlockLines = [];
        codeBlockLang = '';
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
        codeBlockLang = line.slice(3).trim();
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      continue;
    }

    // Headings
    const h3 = line.match(/^### (.+)$/);
    if (h3) { result.push(`<h3 class="md-h3">${applyInline(escape(h3[1]))}</h3>`); continue; }
    const h2 = line.match(/^## (.+)$/);
    if (h2) { result.push(`<h2 class="md-h2">${applyInline(escape(h2[1]))}</h2>`); continue; }
    const h1 = line.match(/^# (.+)$/);
    if (h1) { result.push(`<h1 class="md-h1">${applyInline(escape(h1[1]))}</h1>`); continue; }

    // Unordered list
    const ul = line.match(/^[\-\*] (.+)$/);
    if (ul) { result.push(`<li class="md-li">${applyInline(escape(ul[1]))}</li>`); continue; }

    // Ordered list
    const ol = line.match(/^\d+\. (.+)$/);
    if (ol) { result.push(`<li class="md-li">${applyInline(escape(ol[1]))}</li>`); continue; }

    // Blank line → paragraph break
    if (line.trim() === '') {
      result.push('<br>');
      continue;
    }

    result.push(`<p class="md-p">${applyInline(escape(line))}</p>`);
  }

  // Close any unclosed code block
  if (inCodeBlock && codeBlockLines.length > 0) {
    result.push(
      `<pre class="code-block"><code>${codeBlockLines.map(escape).join('\n')}</code></pre>`
    );
  }

  return result.join('');
}

function applyInline(text: string): string {
  // Bold: **text**
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic: *text*
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Inline code: `text`
  text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  return text;
}
