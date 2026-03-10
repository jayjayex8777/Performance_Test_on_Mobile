package com.example.infer

import android.app.KeyguardManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.os.BatteryManager
import android.os.Environment
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import com.example.infer.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val coroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val timeSteps = 20
    private val random = Random(System.currentTimeMillis())
    private var wakeLock: PowerManager.WakeLock? = null

    private fun acquireWakeLock() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "infer:measurement")
        wakeLock?.acquire(30 * 60 * 1000L) // 30분 타임아웃 safety net
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    @Suppress("DEPRECATION")
    private fun wakeUpScreen() {
        // 1) API 27+ 공식 방법: setTurnScreenOn
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setTurnScreenOn(true)
            setShowWhenLocked(true)
        }

        // 2) Window flags (API 26 이하 호환 + 보조)
        window.addFlags(
            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        )

        // 3) KeyguardManager로 잠금화면 해제 요청 (API 26+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val km = getSystemService(KEYGUARD_SERVICE) as KeyguardManager
            km.requestDismissKeyguard(this, null)
        }

        // 4) Deprecated wake lock (추가 fallback)
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val screenLock = pm.newWakeLock(
            PowerManager.FULL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
            "infer:screen_on"
        )
        screenLock.acquire(5000) // 5초 후 자동 해제
    }

    @Suppress("DEPRECATION")
    private fun clearScreenFlags() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setTurnScreenOn(false)
            setShowWhenLocked(false)
        }
        window.clearFlags(
            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.main) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        binding.measureButton.setOnClickListener {
            startMeasure()
        }
        binding.accuracyButton.setOnClickListener {
            startAccuracy()
        }
        binding.rebuttalMeasureButton.setOnClickListener {
            startRebuttalMeasure()
        }
        binding.rebuttalAccuracyButton.setOnClickListener {
            startRebuttalAccuracy()
        }
        binding.ampMeasureButton.setOnClickListener {
            startAmpMeasure()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        releaseWakeLock()
        coroutineScope.cancel()
    }

    private fun startMeasure() {
        setButtonsEnabled(false)
        binding.statusText.text = "Latency + Battery 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        // 측정 전 화면 밝기를 최소로 (OLED에서 전력 소모 최소화)
        val savedBrightness = window.attributes.screenBrightness
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                runInference()
            }
            // 측정 완료 후 밝기 복원
            window.attributes = window.attributes.apply { screenBrightness = savedBrightness }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        binding.measureButton.isEnabled = enabled
        binding.accuracyButton.isEnabled = enabled
        binding.rebuttalMeasureButton.isEnabled = enabled
        binding.rebuttalAccuracyButton.isEnabled = enabled
        binding.ampMeasureButton.isEnabled = enabled
    }

    private data class InferenceResult(val message: String, val success: Boolean)
    private data class PreprocessedCsv(val name: String, val tensor: Tensor, val preprocessMs: Double)

    private fun runInference(): InferenceResult {
        val assetManager = assets
        val allCsvFiles = assetManager.list("data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("CSV 파일을 불러오지 못했습니다.", false)
        if (allCsvFiles.isEmpty()) {
            return InferenceResult("assets/data 폴더에 CSV가 없습니다.", false)
        }
        val csvFiles = allCsvFiles.take(allCsvFiles.size / 4)

        val sgModels = listOf(
            "snn_smallest_sensor.ptl",
            "snn_small_sensor.ptl",
            "snn_medium_sensor.ptl",
            "snn_large_sensor.ptl",
            "snn_largest_sensor.ptl",
        )
        val steModels = listOf(
            "student_kd_smallest_sensor.ptl",
            "student_kd_small_sensor.ptl",
            "student_kd_medium_sensor.ptl",
            "student_kd_large_sensor.ptl",
            "student_kd_largest_sensor.ptl",
        )
        val cnnModels = listOf(
            "cnn_smallest_sensor.ptl",
            "cnn_small_sensor.ptl",
            "cnn_medium_sensor.ptl",
            "cnn_large_sensor.ptl",
            "cnn_largest_sensor.ptl",
        )
        val qcnnModels = listOf(
            "qcnn_smallest_sensor.ptl",
            "qcnn_small_sensor.ptl",
            "qcnn_medium_sensor.ptl",
            "qcnn_large_sensor.ptl",
            "qcnn_largest_sensor.ptl",
        )

        val snnT3 = listOf("snn_smallest_T3.ptl","snn_small_T3.ptl","snn_medium_T3.ptl","snn_large_T3.ptl","snn_largest_T3.ptl")
        val snnT5 = listOf("snn_smallest_T5.ptl","snn_small_T5.ptl","snn_medium_T5.ptl","snn_large_T5.ptl","snn_largest_T5.ptl")
        val snnT10 = listOf("snn_smallest_T10.ptl","snn_small_T10.ptl","snn_medium_T10.ptl","snn_large_T10.ptl","snn_largest_T10.ptl")
        val snnT15 = listOf("snn_smallest_T15.ptl","snn_small_T15.ptl","snn_medium_T15.ptl","snn_large_T15.ptl","snn_largest_T15.ptl")
        val steT3 = listOf("student_kd_smallest_T3.ptl","student_kd_small_T3.ptl","student_kd_medium_T3.ptl","student_kd_large_T3.ptl","student_kd_largest_T3.ptl")
        val steT5 = listOf("student_kd_smallest_T5.ptl","student_kd_small_T5.ptl","student_kd_medium_T5.ptl","student_kd_large_T5.ptl","student_kd_largest_T5.ptl")
        val steT10 = listOf("student_kd_smallest_T10.ptl","student_kd_small_T10.ptl","student_kd_medium_T10.ptl","student_kd_large_T10.ptl","student_kd_largest_T10.ptl")
        val steT15 = listOf("student_kd_smallest_T15.ptl","student_kd_small_T15.ptl","student_kd_medium_T15.ptl","student_kd_large_T15.ptl","student_kd_largest_T15.ptl")

        val sparseT3fr05 = listOf("sparse_smallest_T3_fr05.ptl","sparse_small_T3_fr05.ptl","sparse_medium_T3_fr05.ptl","sparse_large_T3_fr05.ptl","sparse_largest_T3_fr05.ptl")
        val sparseT3fr10 = listOf("sparse_smallest_T3_fr10.ptl","sparse_small_T3_fr10.ptl","sparse_medium_T3_fr10.ptl","sparse_large_T3_fr10.ptl","sparse_largest_T3_fr10.ptl")
        val sparseT3fr20 = listOf("sparse_smallest_T3_fr20.ptl","sparse_small_T3_fr20.ptl","sparse_medium_T3_fr20.ptl","sparse_large_T3_fr20.ptl","sparse_largest_T3_fr20.ptl")
        val sparseT3fr30 = listOf("sparse_smallest_T3_fr30.ptl","sparse_small_T3_fr30.ptl","sparse_medium_T3_fr30.ptl","sparse_large_T3_fr30.ptl","sparse_largest_T3_fr30.ptl")
        val sparseT5fr05 = listOf("sparse_smallest_T5_fr05.ptl","sparse_small_T5_fr05.ptl","sparse_medium_T5_fr05.ptl","sparse_large_T5_fr05.ptl","sparse_largest_T5_fr05.ptl")
        val sparseT5fr10 = listOf("sparse_smallest_T5_fr10.ptl","sparse_small_T5_fr10.ptl","sparse_medium_T5_fr10.ptl","sparse_large_T5_fr10.ptl","sparse_largest_T5_fr10.ptl")
        val sparseT5fr20 = listOf("sparse_smallest_T5_fr20.ptl","sparse_small_T5_fr20.ptl","sparse_medium_T5_fr20.ptl","sparse_large_T5_fr20.ptl","sparse_largest_T5_fr20.ptl")
        val sparseT5fr30 = listOf("sparse_smallest_T5_fr30.ptl","sparse_small_T5_fr30.ptl","sparse_medium_T5_fr30.ptl","sparse_large_T5_fr30.ptl","sparse_largest_T5_fr30.ptl")

        val qsparseT3fr05 = listOf(
            "qsparse_smallest_T3_fr05.ptl",
            "qsparse_small_T3_fr05.ptl",
            "qsparse_medium_T3_fr05.ptl",
            "qsparse_large_T3_fr05.ptl",
            "qsparse_largest_T3_fr05.ptl"
        )

        val allModels = listOf(
            "QCNN" to qcnnModels,
            "QSPARSE_T3_FR05" to qsparseT3fr05,
        )
        val scaleFactors = listOf(1)

        val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val currentSupported =
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) != Int.MIN_VALUE

        val builder = StringBuilder()
        builder.append("CSV 파일 수: ${csvFiles.size}\n시간 축(T): $timeSteps\n")
        builder.append("스케일 팩터: ${scaleFactors.joinToString(",")}\n")
        builder.append("전류 측정 지원: ${if (currentSupported) "지원" else "미지원"}\n")

        return try {
            builder.append("파일별 on-the-fly 처리 (메모리 절약 모드)\n\n")

            val outDir = getOutputDir()
            val latencyFile = nextResultFile(outDir, "latency_result")
            val batteryFile = nextResultFile(outDir, "battery_result")

            latencyFile.printWriter().use { latPw ->
                batteryFile.printWriter().use { batPw ->
                    latPw.println("scale,group,model,total_forward_ms,avg_forward_ms,count")
                    batPw.println("scale,group,model,avg_current_uA,est_mAh,current_samples")

                    for (scale in scaleFactors) {
                        builder.append("--- Scale ${scale}x ---\n")

                        for ((groupName, models) in allModels) {
                            for ((idx, modelFile) in models.withIndex()) {
                                val modulePath = assetFilePath(modelFile)
                                val module = LiteModuleLoader.load(modulePath)
                                var totalForwardNs = 0L
                                var forwardCount = 0

                                val groupT = when {
                                    groupName.contains("T3") -> 3
                                    groupName.contains("T5") -> 5
                                    groupName.contains("T10") -> 10
                                    groupName.contains("T15") -> 15
                                    else -> timeSteps
                                }

                                // [Phase 1] 텐서를 메모리에 미리 전부 로딩 (I/O를 측정 구간에서 완전 분리)
                                val preloadedTensors = csvFiles.map { csv ->
                                    val tensor = loadCsvAsTensor("data/$csv", groupT)
                                    if (scale == 1) tensor else scaleTensorH(tensor, scale)
                                }

                                // Warm-up: 미리 로딩된 텐서로 5회 추론 (JIT/캐시 안정화, 측정 제외)
                                for (i in 0 until 5) {
                                    module.forward(IValue.from(preloadedTensors[i])).toTensor()
                                }

                                // 배터리 샘플링은 warm-up 후 시작
                                var stopSampling = false
                                var sampler: Job? = null
                                val currentSamples = mutableListOf<Int>()
                                var chargeCoulomb = 0.0
                                var lastNs = System.nanoTime()
                                if (currentSupported) {
                                    sampler = coroutineScope.launch(Dispatchers.IO) {
                                        while (!stopSampling) {
                                            val nowNs = System.nanoTime()
                                            val dt = (nowNs - lastNs) / 1_000_000_000.0
                                            lastNs = nowNs
                                            val raw = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                                            if (raw != Int.MIN_VALUE) {
                                                val uA = kotlin.math.abs(raw)
                                                currentSamples.add(uA)
                                                chargeCoulomb += (uA / 1_000_000.0) * dt
                                            }
                                            delay(50)
                                        }
                                    }
                                }

                                // [Phase 2] 순수 추론만 100회 반복 (I/O 완전 제거, 배터리 샘플 극대화)
                                val repeatCount = 100
                                for (rep in 0 until repeatCount) {
                                    for (tensor in preloadedTensors) {
                                        val start = System.nanoTime()
                                        module.forward(IValue.from(tensor)).toTensor()
                                        totalForwardNs += System.nanoTime() - start
                                        forwardCount += 1
                                    }
                                }
                                // GC는 모델 단위로 1회만 (측정 루프 밖)
                                System.gc()
                                stopSampling = true
                                sampler?.let { kotlinx.coroutines.runBlocking { it.join() } }

                                val totalMs = totalForwardNs / 1_000_000.0
                                val avgMs = if (forwardCount > 0) totalMs / forwardCount else 0.0
                                latPw.println("${scale}x,$groupName,$modelFile,${formatMs(totalMs)},${formatMs(avgMs)},$forwardCount")

                                val avgCurrent = if (currentSamples.isNotEmpty()) currentSamples.average() else 0.0
                                val estmAh = chargeCoulomb * 1000.0 / 3600.0
                                batPw.println(
                                    "${scale}x,$groupName,$modelFile," +
                                        "${formatValue(avgCurrent)}," +
                                        "${formatValue(estmAh)}," +
                                        currentSamples.size
                                )

                                builder.append("[${scale}x][$groupName] $modelFile: avg ${formatMs(avgMs)} ms, ${formatValue(avgCurrent)} uA\n")
                                Thread.sleep(3_000)
                            }
                        }
                    }
                }
            }

            builder.append("\n완료.\nLatency CSV: ${latencyFile.absolutePath}\nBattery CSV: ${batteryFile.absolutePath}")
            InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            InferenceResult("인퍼런스 중 오류 발생: ${t.localizedMessage}", false)
        }
    }

    /**
     * 텐서의 H축(dim=2)을 scale배 복제: (1,12,H,T) -> (1,12,H*scale,T)
     */
    private fun scaleTensorH(original: Tensor, scale: Int): Tensor {
        if (scale <= 1) return original
        val data = original.dataAsFloatArray
        val shape = original.shape()   // [1, 12, H, T]
        val C = shape[1].toInt()       // 12
        val H = shape[2].toInt()
        val T = shape[3].toInt()
        val newH = H * scale
        val newData = FloatArray(C * newH * T)
        for (c in 0 until C) {
            for (s in 0 until scale) {
                System.arraycopy(
                    data, c * H * T,
                    newData, c * newH * T + s * H * T,
                    H * T
                )
            }
        }
        return Tensor.fromBlob(newData, longArrayOf(1, C.toLong(), newH.toLong(), T.toLong()))
    }

    private fun preprocessCsvFiles(csvFiles: List<String>): Pair<List<PreprocessedCsv>, Double> {
        val result = mutableListOf<PreprocessedCsv>()
        var totalMs = 0.0
        for (csv in csvFiles) {
            val start = System.nanoTime()
            val tensor = loadCsvAsTensor("data/$csv")
            val elapsedMs = (System.nanoTime() - start) / 1_000_000.0
            totalMs += elapsedMs
            result.add(PreprocessedCsv(csv, tensor, elapsedMs))
        }
        return Pair(result, totalMs)
    }

    private fun loadCsvAsTensor(path: String, T: Int = timeSteps): Tensor {
        assets.open(path).bufferedReader().use { reader ->
            val rows = mutableListOf<FloatArray>()
            reader.lineSequence().drop(1).forEach { line ->
                val tokens = line.split(',')
                if (tokens.size >= 7) {
                    val values = FloatArray(6)
                    for (i in 0 until 6) {
                        values[i] = tokens[i + 1].toFloatOrNull() ?: 0f
                    }
                    rows.add(values)
                }
            }

            val length = rows.size
            if (length == 0) {
                throw IllegalArgumentException("CSV가 비어 있습니다: $path")
            }

            val signals = Array(6) { FloatArray(length) }
            for (i in rows.indices) {
                for (c in 0 until 6) {
                    signals[c][i] = rows[i][c]
                }
            }

            val buffer = FloatArray(12 * length * T)

            for (c in 0 until 6) {
                val channelOffset = c * 2
                val pos = FloatArray(length) { idx -> max(signals[c][idx], 0f) }
                val neg = FloatArray(length) { idx -> max(-signals[c][idx], 0f) }

                val posProb = toProbabilities(pos)
                val negProb = toProbabilities(neg)

                fillChannel(channelOffset, posProb, length, buffer, T)
                fillChannel(channelOffset + 1, negProb, length, buffer, T)
            }

            val shape = longArrayOf(1, 12, length.toLong(), T.toLong())
            return Tensor.fromBlob(buffer, shape)
        }
    }

    private fun toProbabilities(signal: FloatArray): FloatArray {
        val mean = signal.average().toFloat()
        var variance = 0f
        for (v in signal) {
            val diff = v - mean
            variance += diff * diff
        }
        val std = sqrt(variance / signal.size.toFloat() + 1e-8f)
        val probs = FloatArray(signal.size)
        for (i in signal.indices) {
            val z = (signal[i] - mean) / (std + 1e-8f)
            probs[i] = 1f / (1f + exp(-z))
        }
        return probs
    }

    private fun fillChannel(channel: Int, probs: FloatArray, length: Int, buffer: FloatArray, T: Int = timeSteps) {
        val offset = channel * length * T
        for (row in 0 until length) {
            val rowOffset = offset + row * T
            val prob = probs[row].coerceIn(0f, 1f)
            for (t in 0 until T) {
                buffer[rowOffset + t] = if (random.nextFloat() < prob) 1f else 0f
            }
        }
    }

    private fun assetFilePath(assetName: String): String {
        val outFile = File(filesDir, assetName)
        // Always overwrite to ensure latest asset version is used
        assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
                output.flush()
            }
        }
        return outFile.absolutePath
    }

    private fun formatMs(value: Double): String = String.format("%.3f", value)
    private fun formatValue(value: Double): String = String.format("%.6f", value)

    private fun nextResultFile(dir: File, prefix: String): File {
        var idx = 0
        while (true) {
            val f = File(dir, "${prefix}_${idx}.csv")
            if (!f.exists()) return f
            idx++
        }
    }

    private fun getOutputDir(): File {
        val downloads = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
        return downloads ?: filesDir
    }

    private fun startAccuracy() {
        setButtonsEnabled(false)
        binding.statusText.text = "Accuracy 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                runAccuracy()
            }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    private fun runAccuracy(): InferenceResult {
        val assetManager = assets
        val csvFiles = assetManager.list("infer_data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("infer_data에서 CSV를 불러올 수 없습니다.", false)
        if (csvFiles.isEmpty()) {
            return InferenceResult("infer_data 폴더에 CSV가 없습니다.", false)
        }

        val classNames = arrayOf("swipe_up", "swipe_down", "flick_up", "flick_down")

        val qcnnModelsAcc = listOf(
            "qcnn_smallest_sensor.ptl",
            "qcnn_small_sensor.ptl",
            "qcnn_medium_sensor.ptl",
            "qcnn_large_sensor.ptl",
            "qcnn_largest_sensor.ptl",
        )
        val qsparseModelsAcc = listOf(
            "qsparse_smallest_T3_fr05.ptl",
            "qsparse_small_T3_fr05.ptl",
            "qsparse_medium_T3_fr05.ptl",
            "qsparse_large_T3_fr05.ptl",
            "qsparse_largest_T3_fr05.ptl",
        )

        val allAccModels = listOf(
            "QCNN" to (qcnnModelsAcc to timeSteps),
            "QSPARSE_T3_FR05" to (qsparseModelsAcc to 3),
        )

        val builder = StringBuilder()
        builder.append("=== QCNN vs QSparse 정확도 비교 ===\n")
        builder.append("CSV 파일 수: ${csvFiles.size}\n\n")

        try {
            for ((groupName, modelInfo) in allAccModels) {
                val (modelFiles, tValue) = modelInfo
                builder.append("── $groupName (T=$tValue) ──\n")

                for (modelFile in modelFiles) {
                    val modelPath = assetFilePath(modelFile)
                    val module = try {
                        LiteModuleLoader.load(modelPath)
                    } catch (t: Throwable) {
                        builder.append("  $modelFile: 로드 실패 (${t.localizedMessage})\n")
                        continue
                    }

                    val totalPerClass = IntArray(classNames.size)
                    val correctPerClass = IntArray(classNames.size)
                    var total = 0
                    var correct = 0

                    for (csv in csvFiles) {
                        val label = inferLabel(csv) ?: continue
                        val tensor = loadCsvAsTensor("infer_data/$csv", tValue)
                        val output = module.forward(IValue.from(tensor)).toTensor()
                        val scores = output.dataAsFloatArray
                        var maxIdx = 0
                        var maxVal = scores[0]
                        for (i in 1 until scores.size) {
                            if (scores[i] > maxVal) {
                                maxVal = scores[i]
                                maxIdx = i
                            }
                        }
                        total += 1
                        totalPerClass[label] += 1
                        if (maxIdx == label) {
                            correct += 1
                            correctPerClass[label] += 1
                        }
                    }

                    if (total > 0) {
                        val acc = correct.toDouble() / total.toDouble() * 100.0
                        builder.append("  $modelFile: ${formatValue(acc)} %  ($correct/$total)\n")
                        for (i in classNames.indices) {
                            val tot = totalPerClass[i]
                            val cor = correctPerClass[i]
                            if (tot == 0) continue
                            val accCls = cor.toDouble() / tot.toDouble() * 100.0
                            builder.append("    ${classNames[i]}: $cor/$tot (${formatValue(accCls)} %)\n")
                        }
                    } else {
                        builder.append("  $modelFile: 유효 샘플 없음\n")
                    }

                    module.destroy()
                    System.gc()
                }
                builder.append("\n")
            }
        } catch (t: Throwable) {
            return InferenceResult("추론 중 오류: ${t.localizedMessage}", false)
        }

        return InferenceResult(builder.toString(), true)
    }

    private fun inferLabel(name: String): Int? {
        val lower = name.lowercase()
        return when {
            lower.contains("swipe_up") -> 0
            lower.contains("swipe_down") -> 1
            lower.contains("flick_up") -> 2
            lower.contains("flick_down") -> 3
            else -> null
        }
    }

    // ===================== Rebuttal =====================

    private val rebuttalModels = listOf(
        "TSNN" to "triggered_snn.ptl",
        "TCNN" to "tiny_cnn.ptl",
        "CCNN" to "comp_cnn.ptl",
    )

    private fun startRebuttalMeasure() {
        setButtonsEnabled(false)
        binding.statusText.text = "Rebuttal Latency + Battery 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()
        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                runRebuttalInference()
            }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    private fun startRebuttalAccuracy() {
        setButtonsEnabled(false)
        binding.statusText.text = "Rebuttal Accuracy 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()
        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                runRebuttalAccuracy()
            }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    /**
     * Raw sensor CSV -> (N, 6) FloatArray for triggered_snn (forward_stream)
     */
    private data class RebuttalRawCsv(
        val name: String,
        val streamTensor: Tensor,
        val rawSignals: Array<FloatArray>,
        val rawRows: List<FloatArray>,
        val length: Int
    )

    /** z-score 정규화 + CNN 텐서 생성 (측정 시간에 포함) */
    private fun normalizeToCnnTensor(rawSignals: Array<FloatArray>, length: Int): Tensor {
        val cnnBuf = FloatArray(6 * length)
        for (c in 0 until 6) {
            val mean = rawSignals[c].average().toFloat()
            var variance = 0f
            for (v in rawSignals[c]) {
                val diff = v - mean
                variance += diff * diff
            }
            val std = sqrt(variance / length.toFloat() + 1e-8f)
            for (i in 0 until length) {
                cnnBuf[c * length + i] = (rawSignals[c][i] - mean) / (std + 1e-8f)
            }
        }
        return Tensor.fromBlob(cnnBuf, longArrayOf(1, 6, length.toLong()))
    }

    // --- TriggerLIF for SNN streaming detect ---
    private class KotlinTriggerLIF(
        val tau: Double = 8.0,
        val thOn: Double = 0.1,
        val thOff: Double = 0.05,
        val refractory: Int = 50
    ) {
        var v = 0.0
        var refCnt = 0
        var armed = true

        fun reset() { v = 0.0; refCnt = 0; armed = true }

        fun step(e: Double): Boolean {
            if (refCnt > 0) {
                refCnt--
                v *= (1.0 - 1.0 / tau)
                return false
            }
            val a = 1.0 - 1.0 / tau
            v = v * a + e * (1.0 - a)
            if (armed && v >= thOn) {
                armed = false
                refCnt = refractory
                return true
            }
            if (v <= thOff) armed = true
            return false
        }
    }

    /** raw rows (L, 6) → SNN input (1, 12, L, T): delta → pos/neg → repeat T */
    private fun buildSnnInput(rows: List<FloatArray>, T: Int = 20): Tensor {
        val L = rows.size
        val buf = FloatArray(12 * L * T)
        for (h in 0 until L) {
            for (rawCh in 0 until 6) {
                val delta = if (h == 0) 0f else rows[h][rawCh] - rows[h - 1][rawCh]
                val pos = maxOf(delta, 0f)
                val neg = maxOf(-delta, 0f)
                val posIdx = rawCh * L * T + h * T
                val negIdx = (rawCh + 6) * L * T + h * T
                for (t in 0 until T) {
                    buf[posIdx + t] = pos
                    buf[negIdx + t] = neg
                }
            }
        }
        return Tensor.fromBlob(buf, longArrayOf(1, 12, L.toLong(), T.toLong()))
    }

    private fun loadCsvRaw(path: String): Triple<Tensor, Array<FloatArray>, Int> {
        assets.open(path).bufferedReader().use { reader ->
            val rows = mutableListOf<FloatArray>()
            reader.lineSequence().drop(1).forEach { line ->
                val tokens = line.split(',')
                if (tokens.size >= 7) {
                    val values = FloatArray(6)
                    for (i in 0 until 6) {
                        values[i] = tokens[i + 1].toFloatOrNull() ?: 0f
                    }
                    rows.add(values)
                }
            }
            val length = rows.size
            if (length == 0) {
                throw IllegalArgumentException("CSV가 비어 있습니다: $path")
            }

            // triggered_snn: (N, 6)
            val streamBuf = FloatArray(length * 6)
            for (i in rows.indices) {
                for (c in 0 until 6) {
                    streamBuf[i * 6 + c] = rows[i][c]
                }
            }
            val streamTensor = Tensor.fromBlob(streamBuf, longArrayOf(length.toLong(), 6))

            // raw signals for CNN (normalization deferred to measurement)
            val signals = Array(6) { FloatArray(length) }
            for (i in rows.indices) {
                for (c in 0 until 6) {
                    signals[c][i] = rows[i][c]
                }
            }

            return Triple(streamTensor, signals, length)
        }
    }

    private fun runRebuttalInference(): InferenceResult {
        val csvFiles = assets.list("rebuttal_data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("rebuttal_data에서 CSV를 불러올 수 없습니다.", false)
        if (csvFiles.isEmpty()) {
            return InferenceResult("rebuttal_data 폴더에 CSV가 없습니다.", false)
        }

        val windowSize = 20
        val snnT = 20
        val preLen = 20
        val postLen = 40
        val gyroWeight = 1.0

        val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val currentSupported =
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) != Int.MIN_VALUE

        val builder = StringBuilder()
        builder.append("=== Rebuttal Measure (Streaming) ===\n")
        builder.append("CSV 파일 수: ${csvFiles.size}\n")
        builder.append("CNN window: $windowSize, SNN T: $snnT\n")
        builder.append("전류 측정 지원: ${if (currentSupported) "지원" else "미지원"}\n")

        return try {
            // Preprocess: CSV 파싱
            val preprocessed = mutableListOf<RebuttalRawCsv>()
            var totalPreprocessMs = 0.0
            for (csv in csvFiles) {
                val start = System.nanoTime()
                val (streamT, rawSignals, len) = loadCsvRaw("rebuttal_data/$csv")
                val rawRows = (0 until len).map { i -> FloatArray(6) { c -> rawSignals[c][i] } }
                val elapsedMs = (System.nanoTime() - start) / 1_000_000.0
                totalPreprocessMs += elapsedMs
                preprocessed.add(RebuttalRawCsv(csv, streamT, rawSignals, rawRows, len))
            }
            val avgPreprocessMs = if (preprocessed.isNotEmpty()) totalPreprocessMs / preprocessed.size else 0.0
            builder.append("전처리(공통, 1회만 실행)\n")
            builder.append("- 총 전처리 시간: ${formatMs(totalPreprocessMs)} ms\n")
            builder.append("- 평균 전처리 시간(파일당): ${formatMs(avgPreprocessMs)} ms\n\n")

            val outDir = getOutputDir()
            val latencyFile = nextResultFile(outDir, "rebuttal_latency")
            val batteryFile = nextResultFile(outDir, "rebuttal_battery")
            val perFileFile = nextResultFile(outDir, "rebuttal_latency_perfile")

            latencyFile.printWriter().use { latPw ->
                batteryFile.printWriter().use { batPw ->
                    perFileFile.printWriter().use { pfPw ->
                        latPw.println("group,model,total_ms,detect_ms,infer_ms,num_triggers,num_windows,count")
                        batPw.println("group,model,avg_current_uA,est_mAh,current_samples")
                        pfPw.println("group,model,file,forward_ms,detect_ms,infer_ms,num_triggers,num_windows")

                        // ========== TSNN: Kotlin TriggerLIF detect + classifier_snn.ptl infer ==========
                        val snnGroupName = "TSNN"
                        val snnModelFile = "classifier_snn.ptl"
                        try {
                            val clsModulePath = assetFilePath(snnModelFile)
                            val clsModule = LiteModuleLoader.load(clsModulePath)
                            val triggerLif = KotlinTriggerLIF()

                            var totalDetectNs = 0L
                            var totalInferNs = 0L
                            var totalTriggers = 0
                            val fileCount = preprocessed.size

                            var stopSampling = false
                            var sampler: Job? = null
                            val currentSamples = mutableListOf<Int>()
                            var chargeCoulomb = 0.0
                            var lastNs = System.nanoTime()
                            if (currentSupported) {
                                sampler = coroutineScope.launch(Dispatchers.IO) {
                                    while (!stopSampling) {
                                        val nowNs = System.nanoTime()
                                        val dt = (nowNs - lastNs) / 1_000_000_000.0
                                        lastNs = nowNs
                                        val raw = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                                        if (raw != Int.MIN_VALUE) {
                                            val uA = kotlin.math.abs(raw)
                                            currentSamples.add(uA)
                                            chargeCoulomb += (uA / 1_000_000.0) * dt
                                        }
                                        delay(50)
                                    }
                                }
                            }

                            for (item in preprocessed) {
                                val rows = item.rawRows

                                // Phase 1: Detect (Kotlin TriggerLIF)
                                val detectStart = System.nanoTime()
                                triggerLif.reset()
                                val triggerIndices = mutableListOf<Int>()
                                for (t in 1 until rows.size) {
                                    var accelSum = 0.0
                                    var gyroSum = 0.0
                                    for (c in 0 until 3) {
                                        accelSum += kotlin.math.abs((rows[t][c] - rows[t - 1][c]).toDouble())
                                    }
                                    for (c in 3 until 6) {
                                        gyroSum += kotlin.math.abs((rows[t][c] - rows[t - 1][c]).toDouble())
                                    }
                                    val e = accelSum + gyroWeight * gyroSum
                                    if (triggerLif.step(e)) {
                                        triggerIndices.add(t)
                                    }
                                }
                                val detectNs = System.nanoTime() - detectStart
                                totalDetectNs += detectNs

                                // Phase 2: Infer (classifier_snn.ptl)
                                val inferStart = System.nanoTime()
                                for (trigIdx in triggerIndices) {
                                    val segStart = maxOf(trigIdx - preLen, 0)
                                    val segEnd = minOf(trigIdx + postLen, rows.size - 1)
                                    val segment = rows.subList(segStart, segEnd + 1)
                                    val snnInput = buildSnnInput(segment, snnT)
                                    val validH = Tensor.fromBlob(intArrayOf(segment.size), longArrayOf(1))
                                    clsModule.forward(IValue.from(snnInput), IValue.from(validH)).toTensor()
                                }
                                val inferNs = System.nanoTime() - inferStart
                                totalInferNs += inferNs
                                totalTriggers += triggerIndices.size

                                val fileDetectMs = detectNs / 1_000_000.0
                                val fileInferMs = inferNs / 1_000_000.0
                                pfPw.println("$snnGroupName,$snnModelFile,${item.name},${formatMs(fileDetectMs + fileInferMs)},${formatMs(fileDetectMs)},${formatMs(fileInferMs)},${triggerIndices.size},0")
                            }

                            stopSampling = true
                            sampler?.let { kotlinx.coroutines.runBlocking { it.join() } }

                            val totalDetectMs = totalDetectNs / 1_000_000.0
                            val totalInferMs = totalInferNs / 1_000_000.0
                            val totalMs = totalDetectMs + totalInferMs

                            latPw.println("$snnGroupName,$snnModelFile,${formatMs(totalMs)},${formatMs(totalDetectMs)},${formatMs(totalInferMs)},$totalTriggers,0,$fileCount")

                            val avgCurrent = if (currentSamples.isNotEmpty()) currentSamples.average() else 0.0
                            val estmAh = chargeCoulomb * 1000.0 / 3600.0
                            batPw.println("$snnGroupName,$snnModelFile,${formatValue(avgCurrent)},${formatValue(estmAh)},${currentSamples.size}")

                            builder.append("[$snnGroupName] $snnModelFile: total ${formatMs(totalMs)} ms (detect ${formatMs(totalDetectMs)} + infer ${formatMs(totalInferMs)}), triggers $totalTriggers, ${formatValue(avgCurrent)} uA\n")
                        } catch (modelErr: Throwable) {
                            builder.append("[$snnGroupName] $snnModelFile: ERROR - ${modelErr.localizedMessage}\n")
                            latPw.println("$snnGroupName,$snnModelFile,ERROR,ERROR,ERROR,0,0,0")
                            batPw.println("$snnGroupName,$snnModelFile,ERROR,ERROR,0")
                        }

                        // 모델 사이 10초 쉬기
                        Thread.sleep(3_000)

                        // ========== CNN models: sliding window polling ==========
                        val cnnModels = listOf("TCNN" to "tiny_cnn.ptl", "CCNN" to "comp_cnn.ptl")
                        for ((cnnIdx, cnnEntry) in cnnModels.withIndex()) {
                            val (groupName, modelFile) = cnnEntry
                            try {
                                val modulePath = assetFilePath(modelFile)
                                val module = LiteModuleLoader.load(modulePath)

                                var totalForwardNs = 0L
                                var totalWindows = 0L
                                val fileCount = preprocessed.size

                                var stopSampling = false
                                var sampler: Job? = null
                                val currentSamples = mutableListOf<Int>()
                                var chargeCoulomb = 0.0
                                var lastNs = System.nanoTime()
                                if (currentSupported) {
                                    sampler = coroutineScope.launch(Dispatchers.IO) {
                                        while (!stopSampling) {
                                            val nowNs = System.nanoTime()
                                            val dt = (nowNs - lastNs) / 1_000_000_000.0
                                            lastNs = nowNs
                                            val raw = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                                            if (raw != Int.MIN_VALUE) {
                                                val uA = kotlin.math.abs(raw)
                                                currentSamples.add(uA)
                                                chargeCoulomb += (uA / 1_000_000.0) * dt
                                            }
                                            delay(50)
                                        }
                                    }
                                }

                                for (item in preprocessed) {
                                    val numWindows = item.length - windowSize + 1
                                    if (numWindows <= 0) continue
                                    val fileStart = System.nanoTime()
                                    for (i in 0 until numWindows) {
                                        val winBuf = FloatArray(6 * windowSize)
                                        for (c in 0 until 6) {
                                            for (j in 0 until windowSize) {
                                                winBuf[c * windowSize + j] = item.rawSignals[c][i + j]
                                            }
                                        }
                                        val winTensor = Tensor.fromBlob(winBuf, longArrayOf(1, 6, windowSize.toLong()))
                                        module.forward(IValue.from(winTensor)).toTensor()
                                    }
                                    val fileElapsedNs = System.nanoTime() - fileStart
                                    totalForwardNs += fileElapsedNs
                                    totalWindows += numWindows
                                    pfPw.println("$groupName,$modelFile,${item.name},${formatMs(fileElapsedNs / 1_000_000.0)},0,0,0,$numWindows")
                                }

                                stopSampling = true
                                sampler?.let { kotlinx.coroutines.runBlocking { it.join() } }

                                val totalMs = totalForwardNs / 1_000_000.0

                                latPw.println("$groupName,$modelFile,${formatMs(totalMs)},0,0,0,$totalWindows,$fileCount")

                                val avgCurrent = if (currentSamples.isNotEmpty()) currentSamples.average() else 0.0
                                val estmAh = chargeCoulomb * 1000.0 / 3600.0
                                batPw.println("$groupName,$modelFile,${formatValue(avgCurrent)},${formatValue(estmAh)},${currentSamples.size}")

                                builder.append("[$groupName] $modelFile: total ${formatMs(totalMs)} ms, $totalWindows windows, ${formatValue(avgCurrent)} uA\n")
                            } catch (modelErr: Throwable) {
                                builder.append("[$groupName] $modelFile: ERROR - ${modelErr.localizedMessage}\n")
                                latPw.println("$groupName,$modelFile,ERROR,ERROR,ERROR,0,0,0")
                                batPw.println("$groupName,$modelFile,ERROR,ERROR,0")
                            }
                            // 모델 사이 10초 쉬기 (마지막 모델 제외)
                            if (cnnIdx < cnnModels.lastIndex) {
                                Thread.sleep(3_000)
                            }
                        }
                    }
                }
            }

            builder.append("\n완료.\nLatency CSV: ${latencyFile.absolutePath}")
            builder.append("\nBattery CSV: ${batteryFile.absolutePath}")
            builder.append("\nPer-file CSV: ${perFileFile.absolutePath}")
            InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            InferenceResult("Rebuttal 인퍼런스 중 오류 발생: ${t.localizedMessage}", false)
        }
    }

    private fun runRebuttalAccuracy(): InferenceResult {
        val csvFiles = assets.list("rebuttal_data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("rebuttal_data에서 CSV를 불러올 수 없습니다.", false)
        if (csvFiles.isEmpty()) {
            return InferenceResult("rebuttal_data 폴더에 CSV가 없습니다.", false)
        }

        val classNames = arrayOf("swipe_up", "swipe_down", "flick_up", "flick_down")
        val builder = StringBuilder()
        builder.append("=== Rebuttal Accuracy ===\n")
        builder.append("CSV 파일 수: ${csvFiles.size}\n\n")

        val outDir = getOutputDir()
        val outFile = nextResultFile(outDir, "rebuttal_accuracy")

        try {
            outFile.printWriter().use { pw ->
                pw.println("group,model,total,correct,accuracy,swipe_up_acc,swipe_down_acc,flick_up_acc,flick_down_acc")

                for ((groupName, modelFile) in rebuttalModels) {
                    val isTriggeredSnn = modelFile == "triggered_snn.ptl"

                    try {
                        val modulePath = assetFilePath(modelFile)
                        val module = LiteModuleLoader.load(modulePath)

                        val totalPerClass = IntArray(classNames.size)
                        val correctPerClass = IntArray(classNames.size)
                        var total = 0
                        var correct = 0

                        for (csv in csvFiles) {
                            val label = inferLabel(csv) ?: continue
                            val (streamT, rawSignals, len) = loadCsvRaw("rebuttal_data/$csv")

                            val scores: FloatArray
                            if (isTriggeredSnn) {
                                val result = module.runMethod("forward_stream", IValue.from(streamT))
                                val tuple = result.toTuple()
                                val logits = tuple[2].toTensor()
                                scores = logits.dataAsFloatArray
                            } else {
                                val cnnT = normalizeToCnnTensor(rawSignals, len)
                                val output = module.forward(IValue.from(cnnT)).toTensor()
                                scores = output.dataAsFloatArray
                            }

                            var maxIdx = 0
                            var maxVal = scores[0]
                            for (i in 1 until scores.size) {
                                if (scores[i] > maxVal) {
                                    maxVal = scores[i]
                                    maxIdx = i
                                }
                            }
                            total += 1
                            totalPerClass[label] += 1
                            if (maxIdx == label) {
                                correct += 1
                                correctPerClass[label] += 1
                            }
                        }

                        val acc = if (total > 0) correct.toDouble() / total.toDouble() else 0.0
                        val classAccs = classNames.indices.map { i ->
                            if (totalPerClass[i] > 0) correctPerClass[i].toDouble() / totalPerClass[i].toDouble() else 0.0
                        }

                        pw.println(
                            "$groupName,$modelFile,$total,$correct," +
                                "${formatValue(acc)}," +
                                classAccs.joinToString(",") { formatValue(it) }
                        )

                        builder.append("[$groupName] $modelFile\n")
                        builder.append("  총 ${total}건, 정답 ${correct}건, Accuracy: ${formatValue(acc * 100)}%\n")
                        for (i in classNames.indices) {
                            val tot = totalPerClass[i]
                            val cor = correctPerClass[i]
                            if (tot == 0) continue
                            val accCls = cor.toDouble() / tot.toDouble() * 100.0
                            builder.append("  - ${classNames[i]}: $cor / $tot (${formatValue(accCls)}%)\n")
                        }
                    } catch (modelErr: Throwable) {
                        builder.append("[$groupName] $modelFile: ERROR - ${modelErr.localizedMessage}\n")
                        pw.println("$groupName,$modelFile,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR")
                    }
                    builder.append("\n")
                }
            }

            builder.append("결과 CSV: ${outFile.absolutePath}")
            return InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            return InferenceResult("Rebuttal 정확도 측정 중 오류: ${t.localizedMessage}", false)
        }
    }

    // ===================== Amplified Measure =====================

    private fun startAmpMeasure() {
        setButtonsEnabled(false)
        binding.statusText.text = "Amplified Measure 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        // 측정 전 화면 밝기를 최소로 (OLED에서 전력 소모 최소화)
        val savedBrightness = window.attributes.screenBrightness
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                runAmpInference()
            }
            // 측정 완료 후 밝기 복원
            window.attributes = window.attributes.apply { screenBrightness = savedBrightness }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    private fun repeatTensorRows(original: Tensor, repeatFactor: Int): Tensor {
        if (repeatFactor <= 1) return original
        val data = original.dataAsFloatArray
        val shape = original.shape()   // [1, 12, H, T]
        val C = shape[1].toInt()
        val H = shape[2].toInt()
        val T = shape[3].toInt()
        val newH = H * repeatFactor
        val newData = FloatArray(C * newH * T)
        for (c in 0 until C) {
            for (r in 0 until repeatFactor) {
                System.arraycopy(
                    data, c * H * T,
                    newData, c * newH * T + r * H * T,
                    H * T
                )
            }
        }
        return Tensor.fromBlob(newData, longArrayOf(1, C.toLong(), newH.toLong(), T.toLong()))
    }

    private fun runAmpInference(): InferenceResult {
        val assetManager = assets
        val allCsvFiles = assetManager.list("data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("CSV 파일을 불러오지 못했습니다.", false)
        if (allCsvFiles.isEmpty()) {
            return InferenceResult("assets/data 폴더에 CSV가 없습니다.", false)
        }
        val csvFiles = allCsvFiles.take(allCsvFiles.size / 4)

        val qcnnModels = listOf("qcnn_smallest_sensor.ptl","qcnn_small_sensor.ptl","qcnn_medium_sensor.ptl","qcnn_large_sensor.ptl","qcnn_largest_sensor.ptl")
        val qsparseT3fr05 = listOf("qsparse_smallest_T3_fr05.ptl","qsparse_small_T3_fr05.ptl","qsparse_medium_T3_fr05.ptl","qsparse_large_T3_fr05.ptl","qsparse_largest_T3_fr05.ptl")

        val allModels = listOf(
            "QCNN" to qcnnModels,
            "QSPARSE_T3_FR05" to qsparseT3fr05,
        )
        val ampFactors = listOf(1, 2)
        val numThreads = 8
        val N = 50

        val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val currentSupported =
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) != Int.MIN_VALUE

        val builder = StringBuilder()
        builder.append("=== Amplified Measure (Multi-thread Differential) ===\n")
        builder.append("CSV 파일 수: ${csvFiles.size}\n")
        builder.append("증폭 배율: ${ampFactors.joinToString(",")}\n")
        builder.append("병렬 스레드: $numThreads, N=$N\n\n")

        return try {
            val outDir = getOutputDir()
            val latencyFile = nextResultFile(outDir, "amp_latency")
            val batteryFile = nextResultFile(outDir, "amp_battery")

            latencyFile.printWriter().use { latPw ->
                batteryFile.printWriter().use { batPw ->
                    latPw.println("amp,group,model,total_forward_ms,avg_forward_ms,count,threads")
                    batPw.println("amp,group,model,method,energy_short_uAs,energy_long_uAs,diff_energy_uAs,energy_per_inference_uAs,avg_current_short_uA,avg_current_long_uA,time_short_s,time_long_s,samples_short,samples_long,threads")

                    for (amp in ampFactors) {
                        builder.append("--- ${amp}x (${numThreads} threads) ---\n")

                        for ((groupName, models) in allModels) {
                            val groupT = when {
                                groupName.contains("T3") -> 3
                                groupName.contains("T5") -> 5
                                groupName.contains("T10") -> 10
                                groupName.contains("T15") -> 15
                                else -> timeSteps
                            }

                            for (modelFile in models) {
                                val modulePath = assetFilePath(modelFile)

                                // 텐서 프리로딩 (1회, 스레드간 파티션으로 분할하여 공유)
                                val preloadedTensors = csvFiles.map { csv ->
                                    val tensor = loadCsvAsTensor("data/$csv", groupT)
                                    if (amp == 1) tensor else repeatTensorRows(tensor, amp)
                                }

                                // 8개 Module 인스턴스 로딩 (스레드당 1개)
                                val modules = (0 until numThreads).map { LiteModuleLoader.load(modulePath) }

                                // 스레드별 텐서 파티션 (각 스레드가 다른 텐서만 접근 → race condition 방지)
                                val chunkSize = (preloadedTensors.size + numThreads - 1) / numThreads
                                val partitions = (0 until numThreads).map { t ->
                                    val start = t * chunkSize
                                    val end = minOf(start + chunkSize, preloadedTensors.size)
                                    if (start < preloadedTensors.size) preloadedTensors.subList(start, end) else emptyList()
                                }

                                val executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads)

                                // Warm-up: 각 Module을 병렬로 워밍업
                                val warmupLatch = java.util.concurrent.CountDownLatch(numThreads)
                                for (t in 0 until numThreads) {
                                    executor.submit {
                                        val tensors = partitions[t]
                                        for (i in 0 until minOf(5, tensors.size)) {
                                            modules[t].forward(IValue.from(tensors[i])).toTensor()
                                        }
                                        warmupLatch.countDown()
                                    }
                                }
                                warmupLatch.await()

                                // --- Phase 1: N회 × 8스레드 병렬 inference ---
                                Thread.sleep(3_000)
                                var stopShort = false
                                val shortSamples = mutableListOf<Int>()
                                var shortSampler: Job? = null
                                val shortStartNs = System.nanoTime()
                                if (currentSupported) {
                                    shortSampler = coroutineScope.launch(Dispatchers.IO) {
                                        while (!stopShort) {
                                            val raw = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                                            if (raw != Int.MIN_VALUE) shortSamples.add(kotlin.math.abs(raw))
                                            delay(50)
                                        }
                                    }
                                }
                                val latch1 = java.util.concurrent.CountDownLatch(numThreads)
                                for (t in 0 until numThreads) {
                                    executor.submit {
                                        val mod = modules[t]
                                        val tensors = partitions[t]
                                        for (rep in 0 until N) {
                                            for (tensor in tensors) {
                                                mod.forward(IValue.from(tensor)).toTensor()
                                            }
                                        }
                                        latch1.countDown()
                                    }
                                }
                                latch1.await()
                                stopShort = true
                                shortSampler?.let { kotlinx.coroutines.runBlocking { it.join() } }
                                val shortElapsedS = (System.nanoTime() - shortStartNs) / 1_000_000_000.0
                                val avgCurrentShort = if (shortSamples.isNotEmpty()) shortSamples.average() else 0.0
                                val energyShort = avgCurrentShort * shortElapsedS

                                // --- Phase 2: 2N회 × 8스레드 병렬 inference (latency 측정 포함) ---
                                Thread.sleep(3_000)
                                var stopLong = false
                                val longSamples = mutableListOf<Int>()
                                var longSampler: Job? = null
                                val longStartNs = System.nanoTime()
                                if (currentSupported) {
                                    longSampler = coroutineScope.launch(Dispatchers.IO) {
                                        while (!stopLong) {
                                            val raw = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                                            if (raw != Int.MIN_VALUE) longSamples.add(kotlin.math.abs(raw))
                                            delay(50)
                                        }
                                    }
                                }
                                val latencyPerThread = Array(numThreads) { LongArray(2) } // [totalNs, count]
                                val latch2 = java.util.concurrent.CountDownLatch(numThreads)
                                for (t in 0 until numThreads) {
                                    executor.submit {
                                        val mod = modules[t]
                                        val tensors = partitions[t]
                                        var threadNs = 0L
                                        var threadCount = 0L
                                        for (rep in 0 until 2 * N) {
                                            for (tensor in tensors) {
                                                val s = System.nanoTime()
                                                mod.forward(IValue.from(tensor)).toTensor()
                                                threadNs += System.nanoTime() - s
                                                threadCount++
                                            }
                                        }
                                        latencyPerThread[t][0] = threadNs
                                        latencyPerThread[t][1] = threadCount
                                        latch2.countDown()
                                    }
                                }
                                latch2.await()
                                stopLong = true
                                longSampler?.let { kotlinx.coroutines.runBlocking { it.join() } }
                                val longElapsedS = (System.nanoTime() - longStartNs) / 1_000_000_000.0
                                val avgCurrentLong = if (longSamples.isNotEmpty()) longSamples.average() else 0.0
                                val energyLong = avgCurrentLong * longElapsedS

                                // Cleanup
                                executor.shutdown()
                                for (mod in modules) mod.destroy()
                                System.gc()

                                // --- Differential 결과 계산 ---
                                val totalForwardNs = latencyPerThread.sumOf { it[0] }
                                val forwardCount = latencyPerThread.sumOf { it[1] }.toInt()
                                val diffEnergy = max(0.0, energyLong - energyShort)
                                val totalInferencesInDiff = preloadedTensors.size * N
                                val energyPerInference = if (totalInferencesInDiff > 0) diffEnergy / totalInferencesInDiff else 0.0

                                val totalMs = totalForwardNs / 1_000_000.0
                                val avgMs = if (forwardCount > 0) totalMs / forwardCount else 0.0
                                latPw.println("${amp}x,$groupName,$modelFile,${formatMs(totalMs)},${formatMs(avgMs)},$forwardCount,$numThreads")
                                batPw.println("${amp}x,$groupName,$modelFile,mt_differential,${formatValue(energyShort)},${formatValue(energyLong)},${formatValue(diffEnergy)},${formatValue(energyPerInference)},${formatValue(avgCurrentShort)},${formatValue(avgCurrentLong)},${formatValue(shortElapsedS)},${formatValue(longElapsedS)},${shortSamples.size},${longSamples.size},$numThreads")
                                builder.append("[${amp}x][$groupName] $modelFile: avg ${formatMs(avgMs)} ms, diff ${formatValue(diffEnergy)} uA·s, per_infer ${formatValue(energyPerInference)} uA·s\n")

                                Thread.sleep(3_000)
                            }
                        }
                    }
                }
            }

            builder.append("\n완료.\nLatency CSV: ${latencyFile.absolutePath}\nBattery CSV: ${batteryFile.absolutePath}")
            InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            InferenceResult("Amplified 인퍼런스 중 오류 발생: ${t.localizedMessage}", false)
        }
    }
}
