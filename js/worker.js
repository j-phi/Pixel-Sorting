// worker.js — Background thread for intensive pixel-sorting image processing
// All sort algorithms, mask generators, and effects ported from main.py

// ─── Colour helpers ──────────────────────────────────────────────────────────

function getHue(r, g, b) {
    let r_f = r / 255.0, g_f = g / 255.0, b_f = b / 255.0;
    let mx = Math.max(r_f, g_f, b_f), mn = Math.min(r_f, g_f, b_f);
    let df = mx - mn;
    let h = 0;
    if (mx === mn) h = 0;
    else if (mx === r_f) h = (60 * ((g_f - b_f) / df) + 360) % 360;
    else if (mx === g_f) h = (60 * ((b_f - r_f) / df) + 120) % 360;
    else if (mx === b_f) h = (60 * ((r_f - g_f) / df) + 240) % 360;
    return h / 360.0;
}

function getLuminance(r, g, b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function getIntensity(r, g, b) {
    return (r + g + b) / 3.0;
}

// ─── Sort-key computation ────────────────────────────────────────────────────

function computeSortKeys(imgData, width, height, sortMode) {
    const keys = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
        const idx = i * 4;
        const r = imgData[idx], g = imgData[idx + 1], b = imgData[idx + 2];
        if (sortMode === 0) keys[i] = getHue(r, g, b);
        else if (sortMode === 1) keys[i] = getLuminance(r, g, b) / 255.0;
        else keys[i] = getIntensity(r, g, b) / 255.0;
    }
    return keys;
}

// ─── Mask generators ─────────────────────────────────────────────────────────

function fastSobelMask(imgData, width, height, threshold) {
    const mask = new Uint8Array(width * height);
    const gray = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
        const idx = i * 4;
        gray[i] = 0.2989 * imgData[idx] + 0.5870 * imgData[idx + 1] + 0.1140 * imgData[idx + 2];
        mask[i] = 1;
    }

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const tl = gray[(y - 1) * width + (x - 1)];
            const tc = gray[(y - 1) * width + x];
            const tr = gray[(y - 1) * width + (x + 1)];
            const l  = gray[y * width + (x - 1)];
            const r  = gray[y * width + (x + 1)];
            const bl = gray[(y + 1) * width + (x - 1)];
            const bc = gray[(y + 1) * width + x];
            const br = gray[(y + 1) * width + (x + 1)];

            const gx = -tl + tr - 2 * l + 2 * r - bl + br;
            const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
            const mag = Math.sqrt(gx * gx + gy * gy);

            mask[y * width + x] = Math.min(255, mag) < threshold ? 1 : 0;
        }
    }
    return mask;
}

function fastIntensityMask(imgData, width, height, threshold, invert) {
    const mask = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        const idx = i * 4;
        const lum = 0.2126 * imgData[idx] + 0.7152 * imgData[idx + 1] + 0.0722 * imgData[idx + 2];
        if (invert) mask[i] = lum < threshold ? 1 : 0;
        else mask[i] = lum > threshold ? 1 : 0;
    }
    return mask;
}

// ─── Linear pixel sort (horizontal / vertical) ──────────────────────────────

function pixelSort1D(imgData, width, height, keys, mask, direction, minLen, reverse) {
    const isHorizontal = (direction === 'horizontal');
    const limitOuter = isHorizontal ? height : width;
    const limitInner = isHorizontal ? width : height;
    const outData = new Uint8ClampedArray(imgData);

    for (let i = 0; i < limitOuter; i++) {
        let start = 0;
        while (start < limitInner) {
            const mIdx = isHorizontal ? (i * width + start) : (start * width + i);
            if (!mask[mIdx]) { start++; continue; }

            let end = start;
            while (end < limitInner) {
                const eIdx = isHorizontal ? (i * width + end) : (end * width + i);
                if (!mask[eIdx]) break;
                end++;
            }

            const length = end - start;
            if (length >= minLen) {
                const seg = new Array(length);
                for (let k = 0; k < length; k++) {
                    const pIdx = isHorizontal ? (i * width + start + k) : ((start + k) * width + i);
                    const p4 = pIdx * 4;
                    seg[k] = { key: keys[pIdx], r: outData[p4], g: outData[p4 + 1], b: outData[p4 + 2], a: outData[p4 + 3] };
                }

                seg.sort((a, b) => a.key - b.key);
                if (reverse) seg.reverse();

                for (let k = 0; k < length; k++) {
                    const pIdx = isHorizontal ? (i * width + start + k) : ((start + k) * width + i);
                    const p4 = pIdx * 4;
                    outData[p4]     = seg[k].r;
                    outData[p4 + 1] = seg[k].g;
                    outData[p4 + 2] = seg[k].b;
                    outData[p4 + 3] = seg[k].a;
                }
            }
            start = end;
        }
    }
    return outData;
}

// ─── Diagonal sort ───────────────────────────────────────────────────────────

function pixelSortDiagonal(imgData, width, height, keys, mask, reverse, angleDeg) {
    const isRight = angleDeg > 0;
    const outData = new Uint8ClampedArray(imgData);
    const diagonals = {};

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const d = isRight ? (x - y) : (x + y);
            if (!diagonals[d]) diagonals[d] = [];
            diagonals[d].push({ x, y });
        }
    }

    for (const d in diagonals) {
        const line = diagonals[d];
        let start = 0;
        while (start < line.length) {
            const p1 = line[start];
            if (!mask[p1.y * width + p1.x]) { start++; continue; }

            let end = start;
            while (end < line.length) {
                const p2 = line[end];
                if (!mask[p2.y * width + p2.x]) break;
                end++;
            }

            const segLen = end - start;
            if (segLen >= 2) {
                const seg = new Array(segLen);
                for (let k = 0; k < segLen; k++) {
                    const p = line[start + k];
                    const pIdx = p.y * width + p.x;
                    const p4 = pIdx * 4;
                    seg[k] = { key: keys[pIdx], r: outData[p4], g: outData[p4 + 1], b: outData[p4 + 2], a: outData[p4 + 3] };
                }
                seg.sort((a, b) => a.key - b.key);
                if (reverse) seg.reverse();

                for (let k = 0; k < segLen; k++) {
                    const p = line[start + k];
                    const pIdx = p.y * width + p.x;
                    const p4 = pIdx * 4;
                    outData[p4]     = seg[k].r;
                    outData[p4 + 1] = seg[k].g;
                    outData[p4 + 2] = seg[k].b;
                    outData[p4 + 3] = seg[k].a;
                }
            }
            start = end;
        }
    }
    return outData;
}

// ─── Polar sort (circle / burst) ─────────────────────────────────────────────

function pixelSortPolar(imgData, width, height, keys, mask, mode) {
    const outData = new Uint8ClampedArray(imgData);
    const cx = width / 2, cy = height / 2;
    const groups = {};

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dx = x - cx, dy = y - cy;
            const r = Math.sqrt(dx * dx + dy * dy);
            const a = Math.atan2(dy, dx);

            let gId;
            if (mode === 'circle') gId = Math.round(r);
            else gId = Math.round((a + Math.PI) / (Math.PI * 2) * 360);

            if (!groups[gId]) groups[gId] = [];
            groups[gId].push({ x, y, r, a });
        }
    }

    for (const g in groups) {
        const line = groups[g];
        if (mode === 'circle') line.sort((m, n) => m.a - n.a);
        else line.sort((m, n) => m.r - n.r);

        let start = 0;
        while (start < line.length) {
            const p1 = line[start];
            if (!mask[p1.y * width + p1.x]) { start++; continue; }

            let end = start;
            while (end < line.length) {
                const p2 = line[end];
                if (!mask[p2.y * width + p2.x]) break;
                end++;
            }

            const segLen = end - start;
            if (segLen >= 2) {
                const seg = new Array(segLen);
                for (let k = 0; k < segLen; k++) {
                    const p = line[start + k];
                    const pIdx = p.y * width + p.x;
                    const p4 = pIdx * 4;
                    seg[k] = { key: keys[pIdx], r: outData[p4], g: outData[p4 + 1], b: outData[p4 + 2], a: outData[p4 + 3] };
                }
                seg.sort((A, B) => A.key - B.key);

                for (let k = 0; k < segLen; k++) {
                    const p = line[start + k];
                    const pIdx = p.y * width + p.x;
                    const p4 = pIdx * 4;
                    outData[p4]     = seg[k].r;
                    outData[p4 + 1] = seg[k].g;
                    outData[p4 + 2] = seg[k].b;
                    outData[p4 + 3] = seg[k].a;
                }
            }
            start = end;
        }
    }
    return outData;
}

// ─── Effects ─────────────────────────────────────────────────────────────────

function applyGaussianMild(imgData, width, height) {
    const out = new Uint8ClampedArray(imgData);
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const kSum = 16;

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let r = 0, g = 0, b = 0, ki = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = ((y + ky) * width + (x + kx)) * 4;
                    const w = kernel[ki++];
                    r += imgData[idx] * w;
                    g += imgData[idx + 1] * w;
                    b += imgData[idx + 2] * w;
                }
            }
            const outIdx = (y * width + x) * 4;
            out[outIdx]     = r / kSum;
            out[outIdx + 1] = g / kSum;
            out[outIdx + 2] = b / kSum;
        }
    }
    return out;
}

function applyNoiseMild(imgData) {
    const out = new Uint8ClampedArray(imgData);
    for (let i = 0; i < out.length; i += 4) {
        const noise = (Math.random() - 0.5) * 30;
        out[i]     = Math.min(255, Math.max(0, out[i] + noise));
        out[i + 1] = Math.min(255, Math.max(0, out[i + 1] + noise));
        out[i + 2] = Math.min(255, Math.max(0, out[i + 2] + noise));
    }
    return out;
}

function applyChromAberration(imgData, width, height) {
    const out = new Uint8ClampedArray(imgData);
    const shift = 2;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const xL = (x - shift + width) % width;
            const xR = (x + shift) % width;
            const idxL = (y * width + xL) * 4;
            const idxR = (y * width + xR) * 4;
            out[idx]     = imgData[idxL];       // R shifted left
            out[idx + 1] = imgData[idx + 1];    // G unchanged
            out[idx + 2] = imgData[idxR + 2];   // B shifted right
        }
    }
    return out;
}

// ─── Message handler ─────────────────────────────────────────────────────────

self.onmessage = function (e) {
    const { action, width, height, params } = e.data;
    const imgData = new Uint8ClampedArray(e.data.imgData);

    if (action === 'linear' || action === 'diagonal' || action === 'polar') {
        const { maskMode, threshold, sortModeIdx } = params;

        let mask;
        if (maskMode === 'sobel')      mask = fastSobelMask(imgData, width, height, threshold);
        else if (maskMode === 'bright') mask = fastIntensityMask(imgData, width, height, threshold, false);
        else if (maskMode === 'dark')   mask = fastIntensityMask(imgData, width, height, threshold, true);

        const keys = computeSortKeys(imgData, width, height, sortModeIdx);

        let resultData;
        if (action === 'linear') {
            resultData = pixelSort1D(imgData, width, height, keys, mask, params.direction, 2, params.reverse);
        } else if (action === 'diagonal') {
            resultData = pixelSortDiagonal(imgData, width, height, keys, mask, false, params.angle);
        } else if (action === 'polar') {
            resultData = pixelSortPolar(imgData, width, height, keys, mask, params.mode);
        }

        self.postMessage({ resultData });
    } else if (action === 'blur') {
        self.postMessage({ resultData: applyGaussianMild(imgData, width, height) });
    } else if (action === 'noise') {
        self.postMessage({ resultData: applyNoiseMild(imgData) });
    } else if (action === 'rgb') {
        self.postMessage({ resultData: applyChromAberration(imgData, width, height) });
    }
};
