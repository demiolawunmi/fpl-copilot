import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import type {
  CopilotBlendSubmitRequest,
  CopilotHybridResultPayload,
} from '../../api/backend';
import { isApiError, postCopilotChat } from '../../api/backend';
import { DashboardCard, DashboardHeader } from '@/components/ui/primitives';

interface Props {
  hybridPayload: CopilotHybridResultPayload | null;
  /** Last blend job input (from live submit or loaded snapshot) — required for live chat context. */
  blendInput?: CopilotBlendSubmitRequest | null;
  applyPhase?: 'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'failed';
}

const WELCOME =
  "👋 Hi! I'm your FPL Copilot. After you apply a model blend, you can ask follow-up questions here — answers use your saved blend context and the live LLM.";

const AskCopilotChat = ({ hybridPayload, blendInput = null, applyPhase = 'idle' }: Props) => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([
    { role: 'assistant', content: WELCOME },
  ]);

  useEffect(() => {
    setMessages([{ role: 'assistant', content: WELCOME }]);
  }, [hybridPayload?.correlation_id]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    if (!hybridPayload || !blendInput) {
      toast.warning('No blend context', {
        description: 'Apply a model blend in the AI Sandbox first (or load a saved session).',
        duration: 5000,
      });
      return;
    }

    if (applyPhase === 'submitting' || applyPhase === 'queued' || applyPhase === 'running') {
      toast.info('Blend in progress', {
        description: 'Wait for the current blend to finish before chatting.',
        duration: 4000,
      });
      return;
    }

    const priorForApi = messages.slice(1).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }]);
    setLoading(true);

    void (async () => {
      try {
        const res = await postCopilotChat({
          schema_version: blendInput.schema_version,
          correlation_id: hybridPayload.correlation_id,
          message: trimmed,
          messages: priorForApi,
          blend_input: blendInput,
          blend_result: hybridPayload,
        });
        setMessages((prev) => [...prev, { role: 'assistant', content: res.answer }]);
      } catch (err) {
        let detail = 'Could not reach the assistant. Try again.';
        if (isApiError(err)) {
          if (err.status === 503) {
            const body = err.details as { detail?: string } | undefined;
            detail = typeof body?.detail === 'string' ? body.detail : 'The LLM is unavailable.';
          } else if (err.bodyText) {
            detail = err.bodyText;
          }
        }
        toast.error('Chat failed', {
          description: detail,
          duration: 7000,
        });
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `Sorry — ${detail}`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    })();
  };

  const canChat =
    hybridPayload != null &&
    blendInput != null &&
    applyPhase !== 'submitting' &&
    applyPhase !== 'queued' &&
    applyPhase !== 'running';

  return (
    <DashboardCard className="flex flex-col">
      <DashboardHeader title="Ask Copilot" description="Live follow-ups using your blend context" />
      <div className="card-scroll max-h-80 flex-1 overflow-y-auto px-5 py-4">
        <div className="flex flex-col gap-3">
          {messages.map((msg, idx) => (
            <FlexMessage key={idx} role={msg.role} content={msg.content} />
          ))}
          {loading ? (
            <p className="px-1 text-xs text-slate-500">Thinking…</p>
          ) : null}
        </div>
      </div>
      <div className="border-t border-white/6 px-5 py-4">
        <div className="flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={canChat ? 'Ask a question…' : 'Apply a blend to enable chat'}
            disabled={loading}
            className="h-8 w-full rounded-lg border border-white/8 bg-slate-800 px-3 text-sm text-white transition-colors placeholder:text-slate-500 hover:border-white/12 focus-visible:border-emerald-400 focus-visible:shadow-[0_0_0_1px_#34d399] focus-visible:outline-none disabled:opacity-50"
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={loading || !canChat}
            className="inline-flex h-8 shrink-0 items-center justify-center gap-2 rounded-lg border border-[rgba(16,185,129,0.22)] font-semibold whitespace-nowrap text-emerald-400 transition-colors select-none hover:bg-[rgba(16,185,129,0.12)] disabled:pointer-events-none disabled:opacity-50"
          >
            {loading ? (
              <>
                <svg
                  className="size-3.5 animate-spin"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                >
                  <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                  <path d="M22 12a10 10 0 0 1-10 10" strokeLinecap="round" />
                </svg>
                Send
              </>
            ) : (
              'Send'
            )}
          </button>
        </div>
      </div>
    </DashboardCard>
  );
};

const FlexMessage = ({ role, content }: { role: 'user' | 'assistant'; content: string }) => (
  <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'}`}>
    <div
      className={`max-w-[85%] rounded-lg border px-3 py-2 text-sm whitespace-pre-line ${
        role === 'user'
          ? 'bg-[rgba(16,185,129,0.18)] text-green-100'
          : 'bg-white/6 text-slate-200'
      }`}
      style={{ borderColor: role === 'user' ? 'rgba(16, 185, 129, 0.28)' : undefined }}
    >
      <p className="whitespace-pre-line">{content}</p>
    </div>
  </div>
);

export default AskCopilotChat;
