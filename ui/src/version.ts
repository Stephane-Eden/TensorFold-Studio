// TensorFold release numbers, compared as numbers ("1.0.9" comes before "1.0.11")

/** Whether release `v` comes before `want` ("1.0.9" before [1, 0, 11]); false when there is no version to compare. */
export function before(v: string | undefined | null, want: number[]): boolean {
  const got = (v ?? '').match(/\d+/g)?.slice(0, 3).map(Number) ?? []
  if (!got.length) return false
  for (let i = 0; i < want.length; i++) if ((got[i] ?? 0) !== want[i]) return (got[i] ?? 0) < want[i]
  return false
}
