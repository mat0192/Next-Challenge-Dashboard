import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const source = readFileSync(new URL("../portfolio_storage.js", import.meta.url), "utf8");
const { default: render } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const key = "nextchallenge.solo.portfolio.v1";

test("read, save, rerun, full reload and conflicting tab", () => {
    const values = new Map();
    let writes = 0;
    globalThis.window = { localStorage: {
        getItem: (k) => values.get(k) ?? null,
        setItem: (k, v) => { writes++; values.set(k, v); },
    }};
    let result;
    const status = {};
    const parent = { querySelector: () => status };
    const call = (data, element = parent) => render({ data: { key, ...data }, parentElement: element,
        setStateValue: (_, value) => { result = value; } });
    call({ action: "read", id: "boot" });
    assert.equal(result.status, "read");
    assert.equal(result.raw, null);
    assert.equal(writes, 0); // Reading must never write defaults.
    const raw = JSON.stringify({ version: 1, cash: 10654.39, holdings: [{ quantity: 480 }] });
    const save = { action: "write", id: "save", expected: null, raw };
    call(save);
    assert.equal(result.status, "saved");
    call(save);
    assert.equal(writes, 1); // Acknowledgement rerun is idempotent.
    call({ action: "read", id: "reload" }, { querySelector: () => ({}) });
    assert.equal(result.raw, raw); // A new component/session restores the same bytes.
    const other = '{"cash": 20000}';
    values.set(key, other);
    call({ action: "write", id: "stale", expected: raw, raw: '{"cash": 1}' });
    assert.equal(result.status, "conflict");
    assert.equal(values.get(key), other);
});

test("storage denial/quota failure reports an error, never success", () => {
    for (const storage of [
        { getItem: () => { throw new Error("SecurityError"); } },
        { getItem: () => null, setItem: () => { throw new Error("QuotaExceededError"); } },
    ]) {
        globalThis.window = { localStorage: storage };
        let result;
        render({ data: { key, action: "write", id: "error", expected: null, raw: "{}" },
            parentElement: { querySelector: () => ({}) }, setStateValue: (_, value) => { result = value; } });
        assert.equal(result.status, "error");
    }
});
