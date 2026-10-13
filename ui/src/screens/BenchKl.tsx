import { api, num, type BenchRun } from '../api'

// null: the same engine with drafts off; 'unchecked': an address chosen but not checked yet (the run waits)
export type KlReference = { url: string; key?: string; model: string } | null | 'unchecked'

const nats = (x: number | null | undefined) => (x == null ? '—' : x === 0 ? '0' : x < 0.0001 ? x.toExponential(1) : x.toFixed(4))
const pc = (x: number | null | undefined, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)

type Found = Awaited<ReturnType<typeof api.benchIdentify>>
// the form's whole state, kept by the Bench screen so it survives a change of mode and the run reads the same choice
export type KlState = { kind: 'drafts' | 'address'; url: string; key: string; found: Found | null; model: string }
export const KL_START: KlState = { kind: 'drafts', url: 'http://127.0.0.1:8000', key: '', found: null, model: '' }

export function klReference(s: KlState): KlReference {
  if (s.kind === 'drafts') return null
  if (!s.found || s.found.error || !s.model) return 'unchecked'
  return { url: s.found.url, key: s.key || undefined, model: s.model }
}

/** The KL run's options: what the candidate is compared with, and whether it answers under load. */
export function KlOptions({ load, setLoad, state, setState }: { load: number; setLoad: (n: number) => void; state: KlState; setState: (f: (s: KlState) => KlState) => void }) {
  const { kind, url, key, found, model } = state
  const pickKind = (k: 'drafts' | 'address') => setState((s) => ({ ...s, kind: k, found: null }))
  const edit = (patch: Partial<KlState>) => setState((s) => ({ ...s, ...patch, found: null }))
  const check = async () => {
    const f = await api.benchIdentify(url, key || undefined).catch((e) => ({ url, engine: null, version: null, models: [], error: String(e) }))
    // an answer for an address or key changed since, or after drafts-off was chosen, is dropped
    setState((s) => (s.kind === 'address' && s.url === url && s.key === key ? { ...s, found: f, model: f.models[0] ?? '' } : s))
  }
  return (
    <div className="aim">
      <span className="small"><b>Compared with</b></span>
      <label className="small"><input type="radio" name="klref" checked={kind === 'drafts'} onChange={() => pickKind('drafts')} /> The same engine with drafts off: exact decoding says the distributions are the same</label>
      <label className="small"><input type="radio" name="klref" checked={kind === 'address'} onChange={() => pickKind('address')} /> A reference engine at an address, such as the unquantized model on vLLM</label>
      {kind === 'address' && !found && <p className="small muted" style={{ margin: 0 }}>Check the reference before running: until then the run waits.</p>}
      {kind === 'address' && <div className="aim-row">
        <input className="input" value={url} onChange={(e) => edit({ url: e.target.value })} aria-label="Reference engine address" spellCheck={false} />
        <input className="input" type="password" value={key} onChange={(e) => edit({ key: e.target.value })} aria-label="Reference engine API key" placeholder="API key, if it wants one" autoComplete="off" />
        <button type="button" className="btn sm" onClick={check}>Check</button>
        {found && (found.error ? <span className="small bad">{found.error}</span>
          : <span className="small">{found.engine}{found.version ? ` ${found.version}` : ''}: <select className="sel" value={model} aria-label="Reference model"
              onChange={(e) => { const m = e.target.value; setState((s) => ({ ...s, model: m })) }}>{found.models.map((m) => <option key={m}>{m}</option>)}</select></span>)}
      </div>}
      <label className="small">The candidate answers <select className="sel" value={load} onChange={(e) => setLoad(Number(e.target.value))} aria-label="Load while the candidate answers">
        <option value={0}>alone</option>{[3, 7, 15].map((n) => <option key={n} value={n}>with {n} other requests going</option>)}</select></label>
      <p className="small muted" style={{ margin: 0 }}>Both engines need log-probabilities on chat completions (TensorFold 1.0.5 on CUDA, vLLM, llama.cpp). Each position is compared where both wrote the same text, up to the first token where they part.</p>
    </div>
  )
}

/** A KL run's figures, its largest divergences and each prompt's agreement. */
export function KlReport({ run }: { run: BenchRun }) {
  const k = run.kl
  if (!k || !k.positions) return null
  const ref = run.kl_reference
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>Against {ref?.kind === 'address' ? <><b>{ref.model}</b> at {ref.url}</> : 'the same engine with drafts off'}{run.kl_load && run.kl_load_verified === false ? `; ${run.kl_load} background requests were asked, but the load is not established (see the notes), so this run does not show the candidate under load` : run.kl_load ? `, the candidate answering with ${run.kl_load} other requests going${run.kl_load_done?.full_share != null ? ` (all of them in flight ${Math.round(run.kl_load_done.full_share * 100)}% of the time; ${run.kl_load_done.ok} background replies)` : ''}` : ''}.</p>
      <dl className="figures big">
        <div><dt>Mean KL</dt><dd className={k.mean_kl === 0 ? 'good' : undefined}>{nats(k.mean_kl)} <small>nats</small></dd></div>
        <div><dt>KL p99</dt><dd>{nats(k.p99_kl)}</dd></div>
        <div><dt>Largest</dt><dd>{nats(k.max_kl)}</dd></div>
        <div><dt>Same top token</dt><dd>{pc(k.same_top)}</dd></div>
        <div><dt>Identical replies</dt><dd>{pc(k.identical, 0)}</dd></div>
        <div><dt>Positions</dt><dd>{num(k.positions)}</dd></div>
      </dl>
      {k.worst.some((w) => w.kl > 0) && <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Largest divergences</th><th>Token</th><th>Reference</th><th>Candidate</th><th>KL</th></tr></thead>
        <tbody>{k.worst.filter((w) => w.kl > 0).map((w, i) => <tr key={i}><td>{w.prompt}</td><td>{w.at}</td><td><code>{JSON.stringify(w.ref)}</code></td><td><code>{JSON.stringify(w.cand)}</code></td><td>{nats(w.kl)}</td></tr>)}</tbody>
      </table></div>}
      <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Prompt</th><th>Compared</th><th>Parted at</th><th>Same top</th></tr></thead>
        <tbody>{run.results.filter((r) => r.kl).map((r) => <tr key={`${r.suite}/${r.prompt}`}><td>{r.suite}/{r.prompt}</td><td>{r.kl!.positions} of {r.m?.tokens ?? '—'}</td>
          <td>{r.kl!.differs_at == null ? (r.kl!.identical ? 'identical' : 'lengths differ') : `token ${r.kl!.differs_at}`}</td><td>{r.kl!.positions ? pc(r.kl!.agree / r.kl!.positions) : '—'}</td></tr>)}</tbody>
      </table></div>
      <p className="small muted">A lower bound of the full KL, from the top {k.top} log-probabilities of each side: what their lists establish, nothing guessed about the rest. Positions after the first different token are not compared: the two replies no longer read the same text.</p>
    </>
  )
}
