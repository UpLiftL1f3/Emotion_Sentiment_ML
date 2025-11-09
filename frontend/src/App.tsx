// import { useState } from "react";
// import "./App.css";

// function App() {
//     // const [count, setCount] = useState(0)
//     const [statement, setStatement] = useState("");

//     const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
//         console.log(event.target.value);
//         setStatement(event.target.value);
//     };

//     const handleSubmit = () => {
//         console.log("Submitted:", statement);
//     };

//     return (
//         <>
//             <div className="container">
//                 <input
//                     className="input"
//                     type="text"
//                     value={statement}
//                     onChange={handleChange}
//                 />
//                 {statement.trim().length > 0 && (
//                     <button className="submit-button" onClick={handleSubmit}>
//                         Submit
//                     </button>
//                 )}
//             </div>
//         </>
//     );
// }

// export default App;

// src/App.tsx
import { useLayoutEffect, useRef, useState } from "react";
import { usePredict } from "./features/predict/usePredict";
import { getInitialTheme, toggleTheme, type Theme } from "./theme";

export default function App() {
    const [text, setText] = useState("");
    const [theme, setThemeState] = useState<Theme>(getInitialTheme());
    const predict = usePredict();
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);
    const MAX_ROWS = 8;

    function adjustTextareaHeight(el: HTMLTextAreaElement, maxRows: number) {
        // Reset height to measure true scroll height
        el.style.height = "auto";
        const computed = window.getComputedStyle(el);
        let lineHeight = parseFloat(computed.lineHeight);
        if (Number.isNaN(lineHeight)) {
            const fontSize = parseFloat(computed.fontSize) || 16;
            lineHeight = fontSize * 1.4;
        }
        const paddingTop = parseFloat(computed.paddingTop) || 0;
        const paddingBottom = parseFloat(computed.paddingBottom) || 0;
        const borderTop = parseFloat(computed.borderTopWidth) || 0;
        const borderBottom = parseFloat(computed.borderBottomWidth) || 0;
        const maxHeight =
            lineHeight * maxRows +
            paddingTop +
            paddingBottom +
            borderTop +
            borderBottom;
        const newHeight = Math.min(el.scrollHeight, Math.ceil(maxHeight));
        el.style.height = `${newHeight}px`;
        el.style.overflowY = el.scrollHeight > newHeight ? "auto" : "hidden";
    }

    useLayoutEffect(() => {
        if (textareaRef.current) {
            adjustTextareaHeight(textareaRef.current, MAX_ROWS);
        }
    }, [text]);

    function onSubmit(e: React.FormEvent) {
        e.preventDefault();
        predict.mutate(text.trim());
    }

    return (
        <div
            style={{
                width: "100vw",
                height: "100vh",
                // minHeight: "100dvh",
                display: "grid",
                placeItems: "center",
                background: "var(--bg)",
                color: "var(--text)",
            }}
        >
            <div
                style={{
                    position: "fixed",
                    top: 12,
                    right: 12,
                    display: "flex",
                    gap: 8,
                }}
            >
                <button
                    type="button"
                    onClick={() => setThemeState(toggleTheme(theme))}
                    aria-label="Toggle theme"
                    title="Toggle theme"
                    style={{
                        padding: "8px 12px",
                        borderRadius: 10,
                    }}
                >
                    {theme === "dark" ? "Light mode" : "Dark mode"}
                </button>
            </div>
            <form
                onSubmit={onSubmit}
                style={{
                    display: "flex",
                    width: "min(90vw, 560px)",
                    justifyContent: "center",
                    flexDirection: "column",
                }}
            >
                <textarea
                    ref={textareaRef}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type text for analysis…"
                    style={{
                        width: "100%",
                        padding: "14px 16px",
                        borderRadius: 12,
                        border: "1px solid var(--border)",
                        background: "var(--input-bg)",
                        color: "var(--text)",
                        resize: "none",
                    }}
                    rows={1}
                />
                <button
                    type="submit"
                    disabled={predict.isPending || !text.trim()}
                    style={{
                        marginTop: 12,
                        padding: "10px 16px",
                        borderRadius: 10,
                        alignSelf: "center",
                        background: "var(--button-bg)",
                        color: "var(--button-text)",
                    }}
                >
                    {predict.isPending ? "Analyzing…" : "Analyze"}
                </button>

                {predict.isError && (
                    <p style={{ marginTop: 12, color: "#ff7a7a" }}>
                        {(predict.error as Error).message}
                    </p>
                )}

                {predict.isSuccess && (
                    <pre
                        style={{
                            marginTop: 12,
                            background: "var(--card-bg)",
                            padding: 12,
                            borderRadius: 12,
                            overflow: "auto",
                        }}
                    >
                        {JSON.stringify(predict.data, null, 2)}
                    </pre>
                )}
            </form>
        </div>
    );
}
