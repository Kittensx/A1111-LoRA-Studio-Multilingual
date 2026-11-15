/* Pure curve helpers for the native LoRA Studio.
 *
 * Rendering stays in loraStudioView.js. Keeping interpolation, chart ranges
 * and point insertion here makes the visual editor testable without Gradio,
 * the DOM, WebUI, or torch.
 */
(function (root) {
    "use strict";

    const EPSILON = 1e-12;

    function finite(value, label) {
        const number = Number(value);
        if (!Number.isFinite(number)) throw new Error(`${label} must be finite`);
        return number;
    }

    function normalizePoints(points) {
        if (!Array.isArray(points) || !points.length) throw new Error("Curve needs at least one point");
        const deduplicated = new Map();
        points.forEach((point) => {
            if (!point || typeof point !== "object") throw new Error("Curve point must be an object");
            const position = finite(point.position, "Curve position");
            const value = finite(point.value, "Curve value");
            // This matches the prompt parser: the last spelling of a repeated
            // position wins, and the visual editor shows the normalized form.
            deduplicated.set(position, value);
        });
        return Array.from(deduplicated, ([position, value]) => ({position, value}))
            .sort((left, right) => left.position - right.position);
    }

    function pointsForWeight(weight) {
        if (!weight || typeof weight !== "object") throw new Error("Weight is missing");
        if (weight.kind === "scalar") {
            const value = finite(weight.value, "Weight");
            return [{position: 0, value}, {position: 1, value}];
        }
        if (weight.kind !== "curve") throw new Error("Weight must be scalar or curve");
        return normalizePoints(weight.points);
    }

    function isNormalizedTimeline(points) {
        return normalizePoints(points).every((point) => point.position >= 0 && point.position <= 1);
    }

    function positionDomain(points) {
        const normalized = normalizePoints(points);
        if (isNormalizedTimeline(normalized)) return {min: 0, max: 1, normalized: true};
        const min = normalized[0].position;
        const max = normalized[normalized.length - 1].position;
        if (Math.abs(max - min) > EPSILON) return {min, max, normalized: false};
        const padding = Math.max(Math.abs(min) * 0.1, 1);
        return {min: min - padding, max: max + padding, normalized: false};
    }

    function valueAt(points, position) {
        const normalized = normalizePoints(points);
        const target = finite(position, "Curve position");
        if (target <= normalized[0].position) return normalized[0].value;
        const last = normalized[normalized.length - 1];
        if (target >= last.position) return last.value;
        for (let index = 1; index < normalized.length; index += 1) {
            const right = normalized[index];
            const left = normalized[index - 1];
            if (target <= right.position) {
                const progress = (target - left.position) / (right.position - left.position);
                return left.value + (right.value - left.value) * progress;
            }
        }
        return last.value;
    }

    function valueDomain(points) {
        const values = normalizePoints(points).map((point) => point.value);
        // Include the neutral baseline in the chart when useful; it makes a
        // negative or muted multiplier understandable at a glance.
        values.push(0);
        let min = Math.min(...values);
        let max = Math.max(...values);
        if (Math.abs(max - min) <= EPSILON) {
            const padding = Math.max(Math.abs(max) * 0.15, 0.25);
            min -= padding;
            max += padding;
        } else {
            const padding = (max - min) * 0.1;
            min -= padding;
            max += padding;
        }
        return {min, max};
    }

    function chart(weight, sampleCount) {
        const points = pointsForWeight(weight);
        const x = positionDomain(points);
        const y = valueDomain(points);
        const count = Math.max(2, Math.min(256, Math.floor(Number(sampleCount) || 48)));
        const samples = [];
        for (let index = 0; index < count; index += 1) {
            const position = x.min + ((x.max - x.min) * index) / (count - 1);
            samples.push({position, value: valueAt(points, position)});
        }
        return {points, x, y, samples};
    }

    function insertNormalizedMidpoint(points) {
        const normalized = normalizePoints(points);
        if (!isNormalizedTimeline(normalized)) {
            throw new Error("Add points through the raw field when curve positions are absolute sampler steps");
        }
        const stops = [0, ...normalized.map((point) => point.position), 1];
        let left = stops[0];
        let right = stops[1];
        for (let index = 1; index < stops.length - 1; index += 1) {
            if (stops[index + 1] - stops[index] > right - left) {
                left = stops[index];
                right = stops[index + 1];
            }
        }
        if (right - left <= EPSILON) throw new Error("There is no free normalized position for another point");
        const position = (left + right) / 2;
        return normalizePoints([...normalized, {position, value: valueAt(normalized, position)}]);
    }

    function insertMidpoint(points) {
        const sorted = normalizePoints(points);
        if (isNormalizedTimeline(sorted)) return insertNormalizedMidpoint(sorted);
        if (sorted.length === 1) return normalizePoints([...sorted, {position: sorted[0].position + 1, value: sorted[0].value}]);
        let left = sorted[0].position;
        let right = sorted[1].position;
        for (let i = 1; i < sorted.length - 1; i += 1) {
            if (sorted[i + 1].position - sorted[i].position > right - left) {
                left = sorted[i].position;
                right = sorted[i + 1].position;
            }
        }
        const position = (left + right) / 2;
        if (!(position > left && position < right)) throw new Error("No representable position between these points");
        return normalizePoints([...sorted, {position, value: valueAt(sorted, position)}]);
    }

    const api = {
        normalizePoints,
        pointsForWeight,
        isNormalizedTimeline,
        positionDomain,
        valueAt,
        chart,
        insertNormalizedMidpoint,
        insertMidpoint,
    };
    root.loraStudioCurves = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof window === "undefined" ? globalThis : window));
