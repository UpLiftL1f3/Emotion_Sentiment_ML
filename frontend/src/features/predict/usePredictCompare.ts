// src/features/predict/usePredictCompare.ts
import { useMutation } from "@tanstack/react-query";
import { apiPost } from "../../lib/api";

export type PredictResponse = {
    winner: string;
    models: Record<
        string,
        {
            sentiment: { label: string; probs: number[] };
            emotion: { label: string; probs: number[] };
        }
    >;
};

export function usePredictCompare() {
    return useMutation({
        mutationKey: ["predict-compare"],
        mutationFn: (text: string) =>
            apiPost<PredictResponse>("/api/predict_compare", { text }),
        // optional: centralize error toast, analytics, etc.
        onError: (err) => {
            console.error("predict_compare failed:", err);
        },
    });
}
