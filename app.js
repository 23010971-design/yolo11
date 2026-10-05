// ============================================================
// YOLO11 REALTIME OBJECT DETECTION
// iPhone Camera + ONNX Runtime Web
// Tô màu vật thể + tên + confidence
// ============================================================

// =========================
// CONFIG
// =========================

const MODEL_PATH = "./best.onnx";

const INPUT_SIZE = 320;

// Chỉ hiển thị vật có confidence >= 60%
const CONF_THRESHOLD = 0.60;

const IOU_THRESHOLD = 0.50;

// Thời gian giữa 2 lần inference
const INFERENCE_INTERVAL = 100;


// =========================
// CLASS NAMES
// =========================

const CLASS_NAMES = [
    "Camera",
    "Emergency_light",
    "Light",
    "Switch",
    "Volume"
];


// =========================
// CLASS COLORS
// =========================

const CLASS_COLORS = [
    [0, 255, 0],       // Camera - xanh lá
    [255, 165, 0],     // Emergency_light - cam
    [255, 255, 0],     // Light - vàng
    [0, 200, 255],     // Switch - xanh dương
    [200, 80, 255]     // Volume - tím
];


// =========================
// DOM
// =========================

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");

const processCanvas = document.getElementById("processCanvas");

const statusElement = document.getElementById("status");

const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");

const objectCountElement =
    document.getElementById("objectCount");

const detectionListElement =
    document.getElementById("detectionList");


// =========================
// VARIABLES
// =========================

let session = null;

let stream = null;

let isRunning = false;

let isInferencing = false;

let lastInferenceTime = 0;

let animationId = null;


// =========================
// STATUS
// =========================

function setStatus(text) {

    if (statusElement) {
        statusElement.textContent = text;
    }

    console.log(text);
}


// =========================
// LOAD MODEL
// =========================

async function loadModel() {

    try {

        setStatus("Loading YOLO11 model...");

        if (typeof ort === "undefined") {

            throw new Error(
                "ONNX Runtime Web chưa được load."
            );

        }

        // Ưu tiên WebGL
        try {

            session = await ort.InferenceSession.create(
                MODEL_PATH,
                {
                    executionProviders: ["webgl"],
                    graphOptimizationLevel: "all"
                }
            );

            console.log(
                "YOLO model loaded with WebGL"
            );

        } catch (webglError) {

            console.warn(
                "WebGL không chạy được. Chuyển sang WASM.",
                webglError
            );

            session = await ort.InferenceSession.create(
                MODEL_PATH,
                {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "all"
                }
            );

            console.log(
                "YOLO model loaded with WASM"
            );
        }

        console.log(
            "Input names:",
            session.inputNames
        );

        console.log(
            "Output names:",
            session.outputNames
        );

        setStatus("YOLO11 model loaded");

        return true;

    } catch (error) {

        console.error(
            "Model loading error:",
            error
        );

        setStatus(
            "Lỗi load model: " + error.message
        );

        return false;
    }
}


// =========================
// CAMERA
// =========================

async function startCamera() {

    try {

        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            throw new Error(
                "Trình duyệt không hỗ trợ camera."
            );
        }

        // Nếu camera cũ đang chạy
        if (stream) {

            stream.getTracks().forEach(
                track => track.stop()
            );

            stream = null;
        }

        setStatus("Opening iPhone camera...");

        stream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    facingMode: {
                        ideal: "environment"
                    },

                    width: {
                        ideal: 640
                    },

                    height: {
                        ideal: 480
                    },

                    frameRate: {
                        ideal: 30,
                        max: 30
                    }
                },

                audio: false
            });


        video.srcObject = stream;

        await video.play();


        // Resize canvas theo camera
        resizeCanvas();


        isRunning = true;

        setStatus("Camera running");


        // Bắt đầu realtime detection
        requestAnimationFrame(
            realtimeLoop
        );


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );

        setStatus(
            "Không mở được camera: " +
            error.message
        );
    }
}


// =========================
// STOP CAMERA
// =========================

function stopCamera() {

    isRunning = false;

    isInferencing = false;


    if (animationId) {

        cancelAnimationFrame(
            animationId
        );

        animationId = null;
    }


    if (stream) {

        stream.getTracks().forEach(
            track => track.stop()
        );

        stream = null;
    }


    video.srcObject = null;


    const ctx =
        overlay.getContext("2d");

    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    if (objectCountElement) {

        objectCountElement.textContent =
            "0";
    }


    if (detectionListElement) {

        detectionListElement.innerHTML =
            "";
    }


    setStatus("Camera stopped");
}


// =========================
// RESIZE CANVAS
// =========================

function resizeCanvas() {

    if (!video.videoWidth ||
        !video.videoHeight) {

        return;
    }


    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;
}


// =========================
// PREPROCESS IMAGE
// =========================

function preprocess() {

    const ctx =
        processCanvas.getContext(
            "2d",
            {
                willReadFrequently: true
            }
        );


    processCanvas.width =
        INPUT_SIZE;

    processCanvas.height =
        INPUT_SIZE;


    // Xóa canvas
    ctx.clearRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    if (!videoWidth ||
        !videoHeight) {

        return null;
    }


    // =========================
    // LETTERBOX
    // =========================

    const scale =
        Math.min(
            INPUT_SIZE / videoWidth,
            INPUT_SIZE / videoHeight
        );


    const newWidth =
        Math.round(
            videoWidth * scale
        );


    const newHeight =
        Math.round(
            videoHeight * scale
        );


    const padX =
        (INPUT_SIZE - newWidth) / 2;


    const padY =
        (INPUT_SIZE - newHeight) / 2;


    // Nền đen
    ctx.fillStyle = "black";

    ctx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    // Vẽ camera
    ctx.drawImage(
        video,
        0,
        0,
        videoWidth,
        videoHeight,
        padX,
        padY,
        newWidth,
        newHeight
    );


    // =========================
    // GET PIXELS
    // =========================

    const imageData =
        ctx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const pixels =
        imageData.data;


    // YOLO input:
    // [1, 3, H, W]

    const area =
        INPUT_SIZE * INPUT_SIZE;


    const input =
        new Float32Array(
            3 * area
        );


    for (
        let i = 0;
        i < area;
        i++
    ) {

        const pixelIndex =
            i * 4;


        const r =
            pixels[pixelIndex] / 255.0;

        const g =
            pixels[pixelIndex + 1] / 255.0;

        const b =
            pixels[pixelIndex + 2] / 255.0;


        input[i] =
            r;

        input[area + i] =
            g;

        input[2 * area + i] =
            b;
    }


    return {

        tensor: new ort.Tensor(
            "float32",
            input,
            [
                1,
                3,
                INPUT_SIZE,
                INPUT_SIZE
            ]
        ),

        scale: scale,

        padX: padX,

        padY: padY
    };
}


// =========================
// RUN INFERENCE
// =========================

async function runInference() {

    if (!session) {

        return [];
    }


    const result =
        preprocess();


    if (!result) {

        return [];
    }


    const inputName =
        session.inputNames[0];


    const feeds = {};

    feeds[inputName] =
        result.tensor;


    const outputs =
        await session.run(
            feeds
        );


    const outputName =
        session.outputNames[0];


    const output =
        outputs[outputName];


    return postprocess(
        output,
        result.scale,
        result.padX,
        result.padY
    );
}


// =========================
// POSTPROCESS
// YOLO DETECTION
// OUTPUT:
// [1, 9, 8400]
// =========================

function postprocess(
    output,
    scale,
    padX,
    padY
) {

    const data =
        output.data;


    const dims =
        output.dims;


    console.log(
        "Output shape:",
        dims
    );


    // YOLO11 detection:
    //
    // [1, 9, 8400]
    //
    // 4 bbox:
    // cx cy w h
    //
    // 5 class:
    // Camera
    // Emergency_light
    // Light
    // Switch
    // Volume


    const numClasses =
        CLASS_NAMES.length;


    const numBoxes =
        8400;


    const detections = [];


    for (
        let i = 0;
        i < numBoxes;
        i++
    ) {

        const cx =
            data[i];


        const cy =
            data[numBoxes + i];


        const w =
            data[2 * numBoxes + i];


        const h =
            data[3 * numBoxes + i];


        // =========================
        // FIND BEST CLASS
        // =========================

        let bestClass =
            -1;

        let bestScore =
            0;


        for (
            let c = 0;
            c < numClasses;
            c++
        ) {

            const score =
                data[
                    (4 + c) *
                    numBoxes +
                    i
                ];


            if (
                score >
                bestScore
            ) {

                bestScore =
                    score;

                bestClass =
                    c;
            }
        }


        // Confidence threshold
        if (
            bestScore <
            CONF_THRESHOLD
        ) {

            continue;
        }


        // =========================
        // XYWH -> XYXY
        // =========================

        let x1 =
            cx - w / 2;

        let y1 =
            cy - h / 2;

        let x2 =
            cx + w / 2;

        let y2 =
            cy + h / 2;


        // =========================
        // REMOVE LETTERBOX
        // =========================

        x1 =
            (x1 - padX) /
            scale;

        y1 =
            (y1 - padY) /
            scale;

        x2 =
            (x2 - padX) /
            scale;

        y2 =
            (y2 - padY) /
            scale;


        // =========================
        // CLAMP
        // =========================

        x1 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x1
                )
            );


        y1 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y1
                )
            );


        x2 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x2
                )
            );


        y2 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y2
                )
            );


        detections.push({

            x1: x1,

            y1: y1,

            x2: x2,

            y2: y2,

            classId: bestClass,

            confidence: bestScore
        });
    }


    // =========================
    // NMS
    // =========================

    return nms(
        detections,
        IOU_THRESHOLD
    );
}


// =========================
// IOU
// =========================

function calculateIoU(
    a,
    b
) {

    const x1 =
        Math.max(
            a.x1,
            b.x1
        );

    const y1 =
        Math.max(
            a.y1,
            b.y1
        );

    const x2 =
        Math.min(
            a.x2,
            b.x2
        );

    const y2 =
        Math.min(
            a.y2,
            b.y2
        );


    const width =
        Math.max(
            0,
            x2 - x1
        );


    const height =
        Math.max(
            0,
            y2 - y1
        );


    const intersection =
        width * height;


    const areaA =
        Math.max(
            0,
            a.x2 - a.x1
        ) *
        Math.max(
            0,
            a.y2 - a.y1
        );


    const areaB =
        Math.max(
            0,
            b.x2 - b.x1
        ) *
        Math.max(
            0,
            b.y2 - b.y1
        );


    const union =
        areaA +
        areaB -
        intersection;


    if (union <= 0) {

        return 0;
    }


    return intersection /
        union;
}


// =========================
// NMS
// =========================

function nms(
    detections,
    iouThreshold
) {

    const sorted =
        [...detections].sort(
            (a, b) =>
                b.confidence -
                a.confidence
        );


    const result = [];


    while (
        sorted.length > 0
    ) {

        const best =
            sorted.shift();


        result.push(best);


        for (
            let i =
                sorted.length - 1;
            i >= 0;
            i--
        ) {

            const current =
                sorted[i];


            // Chỉ NMS cùng class
            if (
                current.classId !==
                best.classId
            ) {

                continue;
            }


            const iou =
                calculateIoU(
                    best,
                    current
                );


            if (
                iou >
                iouThreshold
            ) {

                sorted.splice(
                    i,
                    1
                );
            }
        }
    }


    return result;
}


// ============================================================
// DRAW DETECTIONS
// TÔ MÀU + TÊN + CONFIDENCE
// ============================================================

function drawDetections(
    detections
) {

    const ctx =
        overlay.getContext(
            "2d"
        );


    // Xóa kết quả cũ
    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    // =========================
    // VẼ TỪNG VẬT
    // =========================

    for (
        const det of detections
    ) {

        const {
            x1,
            y1,
            x2,
            y2,
            classId,
            confidence
        } = det;


        const color =
            CLASS_COLORS[
                classId
            ] ||
            [0, 255, 0];


        const r =
            color[0];

        const g =
            color[1];

        const b =
            color[2];


        const width =
            x2 - x1;

        const height =
            y2 - y1;


        // =========================
        // 1. TÔ MÀU VẬT
        // =========================

        ctx.fillStyle =
            `rgba(${r}, ${g}, ${b}, 0.35)`;


        ctx.fillRect(
            x1,
            y1,
            width,
            height
        );


        // =========================
        // 2. VIỀN
        // =========================

        ctx.strokeStyle =
            `rgb(${r}, ${g}, ${b})`;


        ctx.lineWidth = 3;


        ctx.strokeRect(
            x1,
            y1,
            width,
            height
        );


        // =========================
        // 3. TÊN VẬT
        // =========================

        const className =
            CLASS_NAMES[
                classId
            ] ||
            `Class ${classId}`;


        const confidenceText =
            `${(
                confidence * 100
            ).toFixed(1)}%`;


        const label =
            `${className} ${confidenceText}`;


        // =========================
        // 4. LABEL FONT
        // =========================

        ctx.font =
            "bold 16px Arial";


        const textWidth =
            ctx.measureText(
                label
            ).width;


        const labelHeight =
            25;


        const labelWidth =
            textWidth + 14;


        // Label nằm trên vật
        let labelX =
            x1;


        let labelY =
            y1 -
            labelHeight;


        // Nếu sát mép trên
        // đưa label xuống
        if (
            labelY < 0
        ) {

            labelY =
                y1;
        }


        // =========================
        // 5. NỀN LABEL
        // =========================

        ctx.fillStyle =
            `rgba(${r}, ${g}, ${b}, 0.90)`;


        ctx.fillRect(
            labelX,
            labelY,
            labelWidth,
            labelHeight
        );


        // =========================
        // 6. CHỮ TRẮNG
        // =========================

        ctx.fillStyle =
            "#ffffff";


        ctx.textBaseline =
            "middle";


        ctx.fillText(
            label,
            labelX + 7,
            labelY +
            labelHeight / 2
        );


        ctx.textBaseline =
            "alphabetic";
    }


    // =========================
    // UPDATE OBJECT COUNT
    // =========================

    if (
        objectCountElement
    ) {

        objectCountElement.textContent =
            detections.length;
    }


    // =========================
    // UPDATE DETECTION LIST
    // =========================

    updateDetectionList(
        detections
    );
}


// =========================
// DETECTION LIST
// =========================

function updateDetectionList(
    detections
) {

    if (
        !detectionListElement
    ) {

        return;
    }


    detectionListElement.innerHTML =
        "";


    if (
        detections.length === 0
    ) {

        detectionListElement.innerHTML =
            "<div>Không phát hiện vật thể</div>";

        return;
    }


    detections.forEach(
        det => {

            const name =
                CLASS_NAMES[
                    det.classId
                ] ||
                `Class ${det.classId}`;


            const percentage =
                (
                    det.confidence *
                    100
                ).toFixed(1);


            const color =
                CLASS_COLORS[
                    det.classId
                ] ||
                [0, 255, 0];


            const item =
                document.createElement(
                    "div"
                );


            item.style.padding =
                "6px 8px";


            item.style.marginBottom =
                "5px";


            item.style.borderRadius =
                "6px";


            item.style.background =
                `rgba(
                    ${color[0]},
                    ${color[1]},
                    ${color[2]},
                    0.15
                )`;


            item.innerHTML =
                `<strong>${name}</strong>
                 - ${percentage}%`;


            detectionListElement.appendChild(
                item
            );
        }
    );
}


// ============================================================
// REALTIME LOOP
// ============================================================

async function realtimeLoop(
    timestamp
) {

    if (!isRunning) {

        return;
    }


    animationId =
        requestAnimationFrame(
            realtimeLoop
        );


    // =========================
    // CAMERA RESIZE
    // =========================

    if (
        overlay.width !==
            video.videoWidth ||
        overlay.height !==
            video.videoHeight
    ) {

        resizeCanvas();
    }


    // =========================
    // LIMIT INFERENCE FPS
    // =========================

    if (
        timestamp -
        lastInferenceTime <
        INFERENCE_INTERVAL
    ) {

        return;
    }


    // =========================
    // KHÔNG CHO INFERENCE
    // CHỒNG LÊN NHAU
    // =========================

    if (
        isInferencing
    ) {

        return;
    }


    isInferencing =
        true;


    lastInferenceTime =
        timestamp;


    try {

        const detections =
            await runInference();


        // Vẽ kết quả
        drawDetections(
            detections
        );


        // Status
        if (
            detections.length > 0
        ) {

            setStatus(
                `Detected ${detections.length} object(s)`
            );

        } else {

            setStatus(
                "Scanning..."
            );
        }


    } catch (error) {

        console.error(
            "Inference error:",
            error
        );


        setStatus(
            "Inference error"
        );


    } finally {

        isInferencing =
            false;
    }
}


// ============================================================
// BUTTON EVENTS
// ============================================================

if (startButton) {

    startButton.addEventListener(
        "click",
        async () => {

            if (!session) {

                const loaded =
                    await loadModel();


                if (!loaded) {

                    return;
                }
            }


            await startCamera();
        }
    );
}


if (stopButton) {

    stopButton.addEventListener(
        "click",
        () => {

            stopCamera();
        }
    );
}


// ============================================================
// VIDEO EVENT
// ============================================================

video.addEventListener(
    "loadedmetadata",
    () => {

        resizeCanvas();
    }
);


// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
    "resize",
    () => {

        resizeCanvas();
    }
);


// ============================================================
// INITIAL STATUS
// ============================================================

setStatus(
    "Ready - load model"
);