import { useEffect, useState } from 'react'
import { api, type Access, type EngineInfo, type Machine } from './api'
import { ROUTES, type Route } from './App'
import { ModelLogo } from './brandLogo'

const ICONS: Record<Route, string> = {
  home: 'M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z',
  models: 'M4 7.5 12 3l8 4.5v9L12 21l-8-4.5zM4 7.5 12 12l8-4.5M12 12v9',
  monitor: 'M3 17l5-6 4 4 5-8 4 5M3 21h18',
  chat: 'M4 5h16v11H9l-5 4z',
  bench: 'M4 20h16M7 20v-6M12 20V8M17 20v-9M5 9l5-4 4 3 5-4',
  connect: 'M10 14a4 4 0 0 0 5.6 0l3-3a4 4 0 0 0-5.6-5.6L11.5 7M14 10a4 4 0 0 0-5.6 0l-3 3a4 4 0 0 0 5.6 5.6l1.5-1.6',
  fleet: 'M3 5h8v6H3zM13 5h8v6h-8zM3 15h8v4H3zM13 15h8v4h-8z',
  teach: 'M12 3l10 5-10 5L2 8zM6 10.5v5c0 1.5 3 3 6 3s6-1.5 6-3v-5M21 8v6',
  settings: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.8 1.2V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-2.9-1.2l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 3 14.9H3a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 4.6 9.1l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 10 5V5a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 2.9 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A1.7 1.7 0 0 0 21 9.1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1.9z',
}
const LABELS: Record<Route, string> = { home: 'Home', chat: 'Chat', teach: 'Teach', models: 'Models', connect: 'Connect', bench: 'Bench', monitor: 'Monitor', fleet: 'Fleet', settings: 'Settings' }

const isMac = /Mac|iPhone|iPad/.test(navigator.platform)

export default function Sidebar({ route, engine, machine, onPalette }: { route: Route; engine: EngineInfo | null; machine: Machine | null; onPalette: () => void }) {
  const [access, setAccess] = useState<Access | null>(null)
  useEffect(() => {
    const load = () => api.access().then(setAccess).catch(() => {})
    load()
    const t = setInterval(load, 60000)
    return () => clearInterval(t)
  }, [engine?.engine_url])
  // who can reach it, in a few words; the addresses and the token are in Settings, under Access
  const reach = !access ? null : access.engine.answers_without_key ? { cls: 'bad', text: 'Open to anyone on the network, no key' }
    : access.posture === 'warn' ? { cls: 'warn', text: !access.studio.local_only ? 'Reachable from your other devices' : 'The AI is reachable on your network' }
    : { cls: '', text: 'This machine only' }
  const st = engine?.status ?? 'unreachable'
  const dot = st === 'running' ? '' : st === 'starting' || st === 'unauthorized' ? 'warn' : st === 'failed' || st === 'error' ? 'bad' : 'off'
  const name = engine?.model_name ?? engine?.model
  const n = engine?.connections ?? 0
  const logo = <ModelLogo names={[engine?.model_name, engine?.model]} size="xs" />
  const line = st === 'running' ? <>{logo}<b>{name}</b> is ready</> : st === 'starting' ? <>{logo}Starting <b>{name ?? '…'}</b></>
    : st === 'unauthorized' ? <>The AI refuses the key</> : st === 'failed' ? <>The AI stopped with a problem</> : engine?.managed === false ? <>No AI at that address</> : <>Nothing running</>
  return (
    <aside>
      <div className="brand">
        <img src="/logo.png" alt="" />
        <div><span className="wordmark">TensorFold</span><small>Studio</small></div>
      </div>
      <nav className="nav" aria-label="Sections">
        {ROUTES.map((r, i) => (
          <a key={r} href={`#${r}`} aria-current={route === r ? 'page' : undefined}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICONS[r]} /></svg>{LABELS[r]}<kbd>{i + 1}</kbd>
          </a>
        ))}
      </nav>
      <button type="button" className="palette-hint" onClick={onPalette}>Commands<kbd>{isMac ? '⌘' : 'Ctrl'} K</kbd></button>
      <div className="status">
        <div className="row"><span className={`dot ${dot}`} /><span>{line}</span></div>
        {st === 'running' && <div className="row"><span style={{ width: 8 }} /><span className="small">{n ? `Answering ${n} ${n === 1 ? 'conversation' : 'conversations'}` : 'Idle'}{engine?.decode_tps ? `, ${Math.round(engine.decode_tps)} tokens a second` : ''}</span></div>}
        {reach && <div className="row"><span className={`dot ${reach.cls}`} title="Who can reach the Studio and the engine" /><span className="small">{reach.text}</span></div>}
        <div className="row faint"><span style={{ width: 8 }} /><span>TensorFold {machine?.tensorfold_version ?? 'not installed'}, Studio {machine?.studio_version ?? ''}</span></div>
      </div>
    </aside>
  )
}
