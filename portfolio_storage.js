// Only portfolio data is stored, never market prices. No user text becomes HTML.
export default function ({ data, parentElement, setStateValue }) {
    const status = parentElement.querySelector("[role=status]");
    if (!data) return;
    if (data.action === "idle") {
        status.textContent = data.message;
        return;
    }
    // Acknowledging a command triggers a Streamlit rerun: do not execute it twice.
    if (parentElement.lastCommand === data.id) return;
    parentElement.lastCommand = data.id;
    const reply = (result) => setStateValue("result", { id: data.id, ...result });
    try {
        const storage = window.localStorage;
        const current = storage.getItem(data.key);
        if (data.action === "read") {
            status.textContent = "Lecture de ta sauvegarde…";
            reply({ status: "read", raw: current });
        } else if (data.action === "write") {
            // Refuse to overwrite changes made by another tab since our last read.
            if (current !== data.expected && current !== data.raw) {
                status.textContent = "Une autre version existe dans ce navigateur.";
                reply({ status: "conflict", raw: current });
                return;
            }
            storage.setItem(data.key, data.raw);
            if (storage.getItem(data.key) !== data.raw) throw new Error("Verification failed");
            status.textContent = "✓ Enregistré dans ce navigateur";
            reply({ status: "saved", raw: data.raw });
        }
    } catch (_) {
        status.textContent = "Sauvegarde automatique indisponible : exporte ton JSON.";
        reply({ status: "error" });
    }
}
