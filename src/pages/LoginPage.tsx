import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTeamId } from '../context/TeamIdContext';
import { Button } from '@/components/ui/button';
import { DashboardCard } from '@/components/ui/primitives';

const LoginPage = () => {
  const [input, setInput] = useState('');
  const { setTeamId } = useTeamId();
  const navigate = useNavigate();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;
    setTeamId(trimmed);
    navigate('/', { replace: true });
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-lg items-center justify-center py-12">
      <DashboardCard className="w-full p-6 md:p-8">
        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          <div className="flex flex-col gap-2 text-center">
            <h1 className="text-2xl font-bold leading-[1.33]">FPL Copilot</h1>
            <p className="text-sm text-slate-400">
              Enter your FPL Team ID to get started
            </p>
          </div>

          <div>
            <input
              type="text"
              inputMode="numeric"
              placeholder="e.g. 123456"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="h-12 w-full rounded-lg border border-white/8 bg-slate-800 px-4 text-base text-white hover:border-white/12 placeholder:text-slate-500 focus-visible:border-emerald-400 focus-visible:shadow-[0_0_0_1px_#34d399] focus-visible:outline-none"
            />
          </div>

          <Button
            type="submit"
            size="lg"
            className="bg-emerald-500 hover:bg-emerald-400"
          >
            Continue
          </Button>
        </form>
      </DashboardCard>
    </div>
  );
};

export default LoginPage;
