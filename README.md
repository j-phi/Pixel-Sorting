# Pixel Sort Studio

> Originally created by [u/No_Commercial_7458](https://www.reddit.com/r/generative/comments/1rq4vko/pixelsortstudio_opensource_python_application/)

A browser-based pixel sorting tool for creating glitch art. Upload an image, choose a sorting direction, and watch pixels rearrange into striking visual patterns. Everything runs locally in your browser — no servers, no uploads, no data leaves your device.

**[→ Try it live](https://j-phi.github.io/Pixel-Sorting/)**

---

## How It Works

Pixel sorting rearranges pixels along a line (row, column, diagonal, or radial path) by a colour property — hue, luminance, or intensity. A **mask** controls which regions get sorted: the Sobel mask detects edges so sorting flows *between* them, while bright/dark masks target specific tonal ranges. The **threshold** slider controls sensitivity.

## Controls

| Group | Controls |
|---|---|
| **Sort direction** | Up · Down · Left · Right · Diag / · Diag \ · Circle · Burst |
| **Sort by** | Hue · Luminance · Intensity |
| **Mask** | Sobel (edges) · Bright · Dark |
| **Effects** | Blur · Noise · RGB Shift |
| **Workflow** | Undo · Reset · Record macro · Play macro · Auto · Save PNG |

## Keyboard Shortcuts

Press **?** at any time to see this in-app. Hold any sort key to continuously apply.

| Key | Action |
|---|---|
| `E` | Sort Up |
| `S` | Sort Left |
| `D` | Sort Down |
| `F` | Sort Right |
| `S`+`D` | Diagonal / |
| `D`+`F` | Diagonal \ |
| `C` | Circle sort |
| `B` | Burst sort |
| `⌘/Ctrl`+`Z` | Undo |
| `Esc` | Stop Auto mode / Close help |
| `?` | Toggle help modal |

## Quick Start

1. Open the app and **drag-and-drop** an image (or click **Upload**)
2. Set the **threshold** — lower values = more sorting, higher = less
3. Click a **sort direction** or press a keyboard shortcut (ESDF)
4. Stack multiple operations, adjust the mask or sort mode, experiment
5. Turn on **Auto** to continuously re-apply the last action
6. **Save** the result as a PNG

## Deploy Your Own

```bash
# Local preview
npx http-server . -p 8080

# GitHub Pages
# Push to GitHub → Settings → Pages → main branch / root
```

## Gallery

<img width="675" height="900" alt="Pixel sorted portrait" src="https://github.com/user-attachments/assets/8149184f-40d3-4f7b-ab65-3e4432652271" />
<img width="1200" height="900" alt="Pixel sorted landscape" src="https://github.com/user-attachments/assets/52342671-7342-401d-bc35-d4d717c328ea" />
<img width="675" height="900" alt="Pixel sorted artwork" src="https://github.com/user-attachments/assets/783716b4-33aa-4304-b7c8-449cee4fe325" />
<img width="1600" height="1200" alt="Pixel sorted scene" src="https://github.com/user-attachments/assets/e31630ee-671a-4dc3-b9f4-40a455f9110b" />

## Architecture

```
index.html        → UI layout, controls
styles.css        → Dark theme, glassmorphism, animations
js/app.js         → UI controller, history, macros, file I/O
js/worker.js      → Web Worker: masks, sort algorithms, effects
```

All image processing runs in a **Web Worker** thread so the UI stays responsive. The worker receives raw pixel data as an `ArrayBuffer`, processes it, and returns the sorted result.

## See Also

- [TASKS.md](TASKS.md) — Feature roadmap and planned work
