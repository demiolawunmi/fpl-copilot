import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { BrowserRouter } from 'react-router-dom'
import { TeamIdProvider } from './context/TeamIdContext'
import { initCdnSeasonSegment } from './utils/cdnSeason'

// Resolve the PL asset CDN season segment (premierleague25/26/…) before the
// first render so every photo/badge URL is built against the live segment.
// Never blocks on failure – falls back to the default segment.
void initCdnSeasonSegment().finally(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <BrowserRouter>
        <TeamIdProvider>
          <App />
        </TeamIdProvider>
      </BrowserRouter>
    </StrictMode>,
  )
})
