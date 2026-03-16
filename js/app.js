/* app.js — Main UI controller for Pixel Sort Studio */

document.addEventListener('DOMContentLoaded', () => {
    // ─── State ───────────────────────────────────────────────────────────
    let images = [];          // File[]
    let currentIndex = -1;
    let history = [];         // ImageData[]
    let originalImageData = null;
    let currentImageData = null;

    let threshold = 60;
    let sortModeIdx = 0;      // 0=Hue, 1=Luma, 2=Intensity
    let maskMode = 'sobel';

    let isRecording = false;
    let recordedSequence = [];

    // ─── DOM ─────────────────────────────────────────────────────────────
    const canvas = document.getElementById('main-canvas');
    const ctx = canvas.getContext('2d');

    const dropOverlay    = document.getElementById('drop-overlay');
    const loadingInd     = document.getElementById('loading-indicator');
    const imageInfo      = document.getElementById('image-info');
    const filenameEl     = document.getElementById('filename-display');
    const indexEl        = document.getElementById('index-display');
    const statusText     = document.getElementById('status-text');
    const historyCount   = document.getElementById('history-count');
    const uploadPlaceholder = document.getElementById('upload-placeholder');

    const fileInput      = document.getElementById('file-input');
    const uploadBtn      = document.getElementById('upload-btn');
    const sliderThresh   = document.getElementById('slider-threshold');
    const thresholdVal   = document.getElementById('threshold-val');

    // ─── Worker ──────────────────────────────────────────────────────────
    const worker = new Worker('js/worker.js');
    let isWorkerBusy = false;

    worker.onmessage = (e) => {
        const resultData = e.data.resultData;
        const newImg = new ImageData(new Uint8ClampedArray(resultData), canvas.width, canvas.height);
        currentImageData = newImg;
        ctx.putImageData(currentImageData, 0, 0);

        isWorkerBusy = false;
        loadingInd.classList.add('hidden');
        setStatus('Ready');

        if (playingMacro && macroIndex < recordedSequence.length) {
            executeMacroStep();
        } else if (playingMacro) {
            playingMacro = false;
        }
    };

    function dispatchWorkerTask(action, params = {}) {
        if (!currentImageData || isWorkerBusy) return;

        history.push(new ImageData(new Uint8ClampedArray(currentImageData.data), canvas.width, canvas.height));
        updateHistoryCount();

        isWorkerBusy = true;
        loadingInd.classList.remove('hidden');
        setStatus('Processing…');

        worker.postMessage({
            action,
            imgData: currentImageData.data.buffer,
            width: canvas.width,
            height: canvas.height,
            params
        });
    }

    // ─── Image loading ───────────────────────────────────────────────────
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            images = Array.from(e.target.files);
            currentIndex = 0;
            await loadImage(images[currentIndex]);
        }
    });

    // Drag-and-drop
    window.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropOverlay.classList.add('visible');
    });
    window.addEventListener('dragleave', (e) => {
        if (e.relatedTarget === null) dropOverlay.classList.remove('visible');
    });
    window.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropOverlay.classList.remove('visible');
        if (e.dataTransfer.files.length > 0) {
            images = Array.from(e.dataTransfer.files);
            currentIndex = 0;
            await loadImage(images[currentIndex]);
        }
    });

    async function loadImage(file) {
        if (!file) return;
        setStatus('Loading image…');
        loadingInd.classList.remove('hidden');

        try {
            let blob = file;
            if (file.name.toLowerCase().endsWith('.heic') || file.name.toLowerCase().endsWith('.heif')) {
                if (typeof heic2any !== 'undefined') {
                    blob = await heic2any({ blob: file, toType: 'image/jpeg' });
                }
            }

            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                currentImageData = new ImageData(new Uint8ClampedArray(originalImageData.data), canvas.width, canvas.height);

                history = [];
                updateHistoryCount();

                filenameEl.textContent = file.name;
                indexEl.textContent = `${currentIndex + 1} / ${images.length}`;
                imageInfo.classList.remove('hidden');
                uploadPlaceholder.classList.add('hidden');

                loadingInd.classList.add('hidden');
                setStatus('Ready');
                URL.revokeObjectURL(url);
            };
            img.src = url;
        } catch (err) {
            console.error('Error loading image', err);
            setStatus('Error loading image.');
            loadingInd.classList.add('hidden');
        }
    }

    // ─── Settings ────────────────────────────────────────────────────────
    sliderThresh.addEventListener('input', (e) => {
        threshold = parseInt(e.target.value);
        thresholdVal.textContent = threshold;
        if (isRecording) recordAction({ type: 'param', name: 'threshold', val: threshold });
    });

    document.getElementById('sort-modes').addEventListener('click', (e) => {
        if (e.target.classList.contains('radio-btn')) {
            document.querySelectorAll('#sort-modes .radio-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            sortModeIdx = parseInt(e.target.dataset.val);
            if (isRecording) recordAction({ type: 'param', name: 'sortModeIdx', val: sortModeIdx });
        }
    });

    document.getElementById('mask-modes').addEventListener('click', (e) => {
        if (e.target.classList.contains('radio-btn')) {
            document.querySelectorAll('#mask-modes .radio-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            maskMode = e.target.dataset.val;
            if (isRecording) recordAction({ type: 'param', name: 'maskMode', val: maskMode });
        }
    });

    // ─── Actions ─────────────────────────────────────────────────────────
    function executeAction(type, action, params = {}) {
        if (isRecording && type !== 'param') recordAction({ type, action, params });
        if (type === 'linear')   dispatchWorkerTask('linear',   { ...params, maskMode, threshold, sortModeIdx });
        else if (type === 'diagonal') dispatchWorkerTask('diagonal', { ...params, maskMode, threshold, sortModeIdx });
        else if (type === 'polar')    dispatchWorkerTask('polar',    { ...params, maskMode, threshold, sortModeIdx });
        else if (type === 'effect')   dispatchWorkerTask(action);
    }

    // Linear
    document.getElementById('btn-up').addEventListener('click',    () => executeAction('linear', 'sort', { direction: 'vertical',   reverse: false }));
    document.getElementById('btn-down').addEventListener('click',  () => executeAction('linear', 'sort', { direction: 'vertical',   reverse: true }));
    document.getElementById('btn-left').addEventListener('click',  () => executeAction('linear', 'sort', { direction: 'horizontal', reverse: true }));
    document.getElementById('btn-right').addEventListener('click', () => executeAction('linear', 'sort', { direction: 'horizontal', reverse: false }));

    // Diagonal
    document.getElementById('btn-diag-up').addEventListener('click',   () => executeAction('diagonal', 'sort', { angle: 45 }));
    document.getElementById('btn-diag-down').addEventListener('click', () => executeAction('diagonal', 'sort', { angle: -45 }));

    // Polar
    document.getElementById('btn-circle').addEventListener('click', () => executeAction('polar', 'sort', { mode: 'circle' }));
    document.getElementById('btn-burst').addEventListener('click',  () => executeAction('polar', 'sort', { mode: 'burst' }));

    // Effects
    document.getElementById('btn-blur').addEventListener('click',  () => executeAction('effect', 'blur'));
    document.getElementById('btn-noise').addEventListener('click', () => executeAction('effect', 'noise'));
    document.getElementById('btn-rgb').addEventListener('click',   () => executeAction('effect', 'rgb'));

    // ─── History & Navigation ────────────────────────────────────────────
    document.getElementById('btn-undo').addEventListener('click', () => {
        if (history.length > 0 && !isWorkerBusy) {
            currentImageData = history.pop();
            ctx.putImageData(currentImageData, 0, 0);
            updateHistoryCount();
        }
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        if (originalImageData && !isWorkerBusy) {
            history.push(new ImageData(new Uint8ClampedArray(currentImageData.data), canvas.width, canvas.height));
            currentImageData = new ImageData(new Uint8ClampedArray(originalImageData.data), canvas.width, canvas.height);
            ctx.putImageData(currentImageData, 0, 0);
            updateHistoryCount();
        }
    });

    document.getElementById('btn-prev').addEventListener('click', async () => {
        if (images.length > 0 && !isWorkerBusy) {
            currentIndex = (currentIndex - 1 + images.length) % images.length;
            await loadImage(images[currentIndex]);
        }
    });

    document.getElementById('btn-next').addEventListener('click', async () => {
        if (images.length > 0 && !isWorkerBusy) {
            currentIndex = (currentIndex + 1) % images.length;
            await loadImage(images[currentIndex]);
        }
    });

    // ─── Macro ───────────────────────────────────────────────────────────
    const btnRecord = document.getElementById('btn-record');
    const btnPlay   = document.getElementById('btn-play');
    let playingMacro = false;
    let macroIndex = 0;

    btnRecord.addEventListener('click', () => {
        isRecording = !isRecording;
        if (isRecording) {
            recordedSequence = [];
            btnRecord.textContent = '■ Stop';
            btnRecord.classList.add('recording');
        } else {
            btnRecord.textContent = 'Record';
            btnRecord.classList.remove('recording');
        }
    });

    function recordAction(obj) { recordedSequence.push(obj); }

    btnPlay.addEventListener('click', () => {
        if (!isRecording && recordedSequence.length > 0 && currentImageData && !isWorkerBusy) {
            playingMacro = true;
            macroIndex = 0;
            executeMacroStep();
        }
    });

    function executeMacroStep() {
        if (macroIndex >= recordedSequence.length) { playingMacro = false; return; }
        const step = recordedSequence[macroIndex++];

        if (step.type === 'param') {
            if (step.name === 'threshold') {
                threshold = step.val;
                sliderThresh.value = threshold;
                thresholdVal.textContent = threshold;
            } else if (step.name === 'sortModeIdx') {
                sortModeIdx = step.val;
                document.querySelectorAll('#sort-modes .radio-btn').forEach(b => b.classList.remove('active'));
                document.querySelector(`#sort-modes .radio-btn[data-val="${sortModeIdx}"]`).classList.add('active');
            } else if (step.name === 'maskMode') {
                maskMode = step.val;
                document.querySelectorAll('#mask-modes .radio-btn').forEach(b => b.classList.remove('active'));
                document.querySelector(`#mask-modes .radio-btn[data-val="${maskMode}"]`).classList.add('active');
            }
            executeMacroStep();
        } else {
            const wasRec = isRecording;
            isRecording = false;
            executeAction(step.type, step.action, step.params);
            isRecording = wasRec;
        }
    }

    // ─── Export ──────────────────────────────────────────────────────────
    document.getElementById('btn-save').addEventListener('click', () => {
        if (!currentImageData || isWorkerBusy) return;
        const link = document.createElement('a');
        const d = new Date();
        link.download = `sort_${String(d.getHours()).padStart(2,'0')}${String(d.getMinutes()).padStart(2,'0')}${String(d.getSeconds()).padStart(2,'0')}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    });

    // ─── Keyboard shortcuts ─────────────────────────────────────────────
    window.addEventListener('keydown', (e) => {
        if (e.key === 'z' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            document.getElementById('btn-undo').click();
        }
    });

    // ─── Helpers ─────────────────────────────────────────────────────────
    function setStatus(text) { statusText.textContent = text; }
    function updateHistoryCount() { historyCount.textContent = `Undo: ${history.length}`; }
});
