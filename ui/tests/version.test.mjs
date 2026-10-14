import assert from 'node:assert/strict'
import { test } from 'node:test'
import { before } from '../src/version.ts'

test('releases compare as numbers, not as text', () => {
  assert.equal(before('1.0.9', [1, 0, 11]), true)          // as text, "1.0.9" would come after "1.0.11"
  assert.equal(before('1.0.11', [1, 0, 11]), false)
  assert.equal(before('1.0.12', [1, 0, 11]), false)
  assert.equal(before('1.1.0', [1, 0, 11]), false)
  assert.equal(before('0.6.6 (Python)', [1, 0, 11]), true)
  assert.equal(before(null, [1, 0, 11]), false)             // nothing to compare: no warning
  assert.equal(before('', [1, 0, 11]), false)
})
