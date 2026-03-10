package com.example.infer

import android.app.KeyguardManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.BatteryManager
import android.os.Environment
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
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
        wakeLock?.acquire(30 * 60 * 1000L)
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    @Suppress("DEPRECATION")
    private fun wakeUpScreen() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setTurnScreenOn(true)
            setShowWhenLocked(true)
        }
        window.addFlags(
            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val km = getSystemService(KEYGUARD_SERVICE) as KeyguardManager
            km.requestDismissKeyguard(this, null)
        }
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val screenLock = pm.newWakeLock(
            PowerManager.FULL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
            "infer:screen_on"
        )
        screenLock.acquire(5000)
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

        binding.basicButton.setOnClickListener {
            startBasicMeasure()
        }
        binding.improvedButton.setOnClickListener {
            startImprovedMeasure()
        }
        binding.basicAccuracyButton.setOnClickListener {
            startBasicAccuracy()
        }
        binding.improvedAccuracyButton.setOnClickListener {
            startImprovedAccuracy()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        releaseWakeLock()
        coroutineScope.cancel()
    }

    // ===================== Basic Models (CNN vs SNNs) =====================

    private fun startBasicMeasure() {
        setButtonsEnabled(false)
        binding.statusText.text = "Basic Models 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        val savedBrightness = window.attributes.screenBrightness
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                val cnnModels = listOf(
                    "cnn_smallest_sensor.ptl",
                    "cnn_small_sensor.ptl",
                    "cnn_medium_sensor.ptl",
                    "cnn_large_sensor.ptl",
                    "cnn_largest_sensor.ptl",
                )
                val snnT3 = listOf(
                    "snn_smallest_T3.ptl",
                    "snn_small_T3.ptl",
                    "snn_medium_T3.ptl",
                    "snn_large_T3.ptl",
                    "snn_largest_T3.ptl",
                )
                val studentKdT3 = listOf(
                    "student_kd_smallest_T3.ptl",
                    "student_kd_small_T3.ptl",
                    "student_kd_medium_T3.ptl",
                    "student_kd_large_T3.ptl",
                    "student_kd_largest_T3.ptl",
                )
                val allModels = listOf(
                    "CNN" to cnnModels,
                    "SNN_T3" to snnT3,
                    "STUDENT_KD_T3" to studentKdT3,
                )
                runDifferentialMeasurement(allModels, "basic")
            }
            window.attributes = window.attributes.apply { screenBrightness = savedBrightness }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    // ===================== Improved Models (QCNN vs QSparse) =====================

    private fun startImprovedMeasure() {
        setButtonsEnabled(false)
        binding.statusText.text = "Improved Models 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        val savedBrightness = window.attributes.screenBrightness
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                val qcnnModels = listOf(
                    "qcnn_smallest_sensor.ptl",
                    "qcnn_small_sensor.ptl",
                    "qcnn_medium_sensor.ptl",
                    "qcnn_large_sensor.ptl",
                    "qcnn_largest_sensor.ptl",
                )
                val qsparseT3fr05 = listOf(
                    "qsparse_smallest_T3_fr05.ptl",
                    "qsparse_small_T3_fr05.ptl",
                    "qsparse_medium_T3_fr05.ptl",
                    "qsparse_large_T3_fr05.ptl",
                    "qsparse_largest_T3_fr05.ptl",
                )
                val allModels = listOf(
                    "QCNN" to qcnnModels,
                    "QSPARSE_T3_FR05" to qsparseT3fr05,
                )
                runDifferentialMeasurement(allModels, "improved")
            }
            window.attributes = window.attributes.apply { screenBrightness = savedBrightness }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    // ===================== Shared Differential Measurement =====================

    private data class InferenceResult(val message: String, val success: Boolean)

    private fun runDifferentialMeasurement(
        allModels: List<Pair<String, List<String>>>,
        csvPrefix: String
    ): InferenceResult {
        val assetManager = assets
        val allCsvFiles = assetManager.list("data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("CSV 파일을 불러오지 못했습니다.", false)
        if (allCsvFiles.isEmpty()) {
            return InferenceResult("assets/data 폴더에 CSV가 없습니다.", false)
        }
        val csvFiles = allCsvFiles.take(allCsvFiles.size / 4)

        val numThreads = 8
        val N = 50

        val batteryManager = getSystemService(BATTERY_SERVICE) as BatteryManager
        val currentSupported =
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) != Int.MIN_VALUE

        val builder = StringBuilder()
        builder.append("=== ${csvPrefix.uppercase()} Measure (Multi-thread Differential) ===\n")
        builder.append("CSV 파일 수: ${csvFiles.size}\n")
        builder.append("병렬 스레드: $numThreads, N=$N\n\n")

        return try {
            val outDir = getOutputDir()
            val latencyFile = nextResultFile(outDir, "${csvPrefix}_latency")
            val batteryFile = nextResultFile(outDir, "${csvPrefix}_battery")

            latencyFile.printWriter().use { latPw ->
                batteryFile.printWriter().use { batPw ->
                    latPw.println("group,model,model_size_kb,total_forward_ms,avg_forward_ms,count,threads")
                    batPw.println("group,model,model_size_kb,method,energy_short_uAs,energy_long_uAs,diff_energy_uAs,energy_per_inference_uAs,avg_current_short_uA,avg_current_long_uA,time_short_s,time_long_s,samples_short,samples_long,threads")

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
                            val modelSizeKb = File(modulePath).length() / 1024.0

                            // 텐서 프리로딩
                            val preloadedTensors = csvFiles.map { csv ->
                                loadCsvAsTensor("data/$csv", groupT)
                            }

                            // 8개 Module 인스턴스 로딩
                            val modules = (0 until numThreads).map { LiteModuleLoader.load(modulePath) }

                            // 스레드별 텐서 파티션
                            val chunkSize = (preloadedTensors.size + numThreads - 1) / numThreads
                            val partitions = (0 until numThreads).map { t ->
                                val start = t * chunkSize
                                val end = minOf(start + chunkSize, preloadedTensors.size)
                                if (start < preloadedTensors.size) preloadedTensors.subList(start, end) else emptyList()
                            }

                            val executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads)

                            // Warm-up
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

                            // --- Phase 1: N회 × 8스레드 ---
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

                            // --- Phase 2: 2N회 × 8스레드 (latency 측정 포함) ---
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
                            val latencyPerThread = Array(numThreads) { LongArray(2) }
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
                            val sizeStr = String.format("%.1f", modelSizeKb)
                            latPw.println("$groupName,$modelFile,$sizeStr,${formatMs(totalMs)},${formatMs(avgMs)},$forwardCount,$numThreads")
                            batPw.println("$groupName,$modelFile,$sizeStr,mt_differential,${formatValue(energyShort)},${formatValue(energyLong)},${formatValue(diffEnergy)},${formatValue(energyPerInference)},${formatValue(avgCurrentShort)},${formatValue(avgCurrentLong)},${formatValue(shortElapsedS)},${formatValue(longElapsedS)},${shortSamples.size},${longSamples.size},$numThreads")
                            builder.append("[$groupName] $modelFile ($sizeStr KB): avg ${formatMs(avgMs)} ms, diff ${formatValue(diffEnergy)} uA·s, per_infer ${formatValue(energyPerInference)} uA·s\n")

                            Thread.sleep(3_000)
                        }
                    }
                }
            }

            builder.append("\n완료.\nLatency CSV: ${latencyFile.absolutePath}\nBattery CSV: ${batteryFile.absolutePath}")
            InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            InferenceResult("측정 중 오류 발생: ${t.localizedMessage}", false)
        }
    }

    // ===================== Accuracy Comparison =====================

    private fun startBasicAccuracy() {
        setButtonsEnabled(false)
        binding.statusText.text = "Basic Models Accuracy 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                val cnnModels = listOf(
                    "cnn_smallest_sensor.ptl",
                    "cnn_small_sensor.ptl",
                    "cnn_medium_sensor.ptl",
                    "cnn_large_sensor.ptl",
                    "cnn_largest_sensor.ptl",
                )
                val snnT3 = listOf(
                    "snn_smallest_T3.ptl",
                    "snn_small_T3.ptl",
                    "snn_medium_T3.ptl",
                    "snn_large_T3.ptl",
                    "snn_largest_T3.ptl",
                )
                val studentKdT3 = listOf(
                    "student_kd_smallest_T3.ptl",
                    "student_kd_small_T3.ptl",
                    "student_kd_medium_T3.ptl",
                    "student_kd_large_T3.ptl",
                    "student_kd_largest_T3.ptl",
                )
                val allModels = listOf(
                    "CNN" to cnnModels,
                    "SNN_T3" to snnT3,
                    "STUDENT_KD_T3" to studentKdT3,
                )
                runAccuracyComparison(allModels, "basic_accuracy")
            }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    private fun startImprovedAccuracy() {
        setButtonsEnabled(false)
        binding.statusText.text = "Improved Models Accuracy 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        coroutineScope.launch {
            val result = withContext(Dispatchers.IO) {
                val qcnnModels = listOf(
                    "qcnn_smallest_sensor.ptl",
                    "qcnn_small_sensor.ptl",
                    "qcnn_medium_sensor.ptl",
                    "qcnn_large_sensor.ptl",
                    "qcnn_largest_sensor.ptl",
                )
                val qsparseT3fr05 = listOf(
                    "qsparse_smallest_T3_fr05.ptl",
                    "qsparse_small_T3_fr05.ptl",
                    "qsparse_medium_T3_fr05.ptl",
                    "qsparse_large_T3_fr05.ptl",
                    "qsparse_largest_T3_fr05.ptl",
                )
                val allModels = listOf(
                    "QCNN" to qcnnModels,
                    "QSPARSE_T3_FR05" to qsparseT3fr05,
                )
                runAccuracyComparison(allModels, "improved_accuracy")
            }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료" else "오류 발생"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            releaseWakeLock()
        }
    }

    private fun runAccuracyComparison(
        allModels: List<Pair<String, List<String>>>,
        csvPrefix: String
    ): InferenceResult {
        val csvFiles = assets.list("data")?.filter { it.endsWith(".csv") }?.sorted()
            ?: return InferenceResult("data 폴더에서 CSV를 불러올 수 없습니다.", false)
        if (csvFiles.isEmpty()) {
            return InferenceResult("data 폴더에 CSV가 없습니다.", false)
        }

        val classNames = arrayOf("swipe_up", "swipe_down", "flick_up", "flick_down")

        val builder = StringBuilder()
        builder.append("=== ${csvPrefix.uppercase().replace("_", " ")} ===\n")
        builder.append("CSV 파일 수: ${csvFiles.size} (${csvFiles.size / 4} per gesture)\n\n")

        return try {
            val outDir = getOutputDir()
            val outFile = nextResultFile(outDir, csvPrefix)

            outFile.printWriter().use { pw ->
                pw.println("group,model,model_size_kb,total,correct,accuracy,swipe_up_acc,swipe_down_acc,flick_up_acc,flick_down_acc")

                for ((groupName, models) in allModels) {
                    val groupT = when {
                        groupName.contains("T3") -> 3
                        groupName.contains("T5") -> 5
                        groupName.contains("T10") -> 10
                        groupName.contains("T15") -> 15
                        else -> timeSteps
                    }

                    builder.append("── $groupName (T=$groupT) ──\n")

                    for (modelFile in models) {
                        val modelPath = assetFilePath(modelFile)
                        val modelSizeKb = File(modelPath).length() / 1024.0
                        val sizeStr = String.format("%.1f", modelSizeKb)

                        val module = try {
                            LiteModuleLoader.load(modelPath)
                        } catch (t: Throwable) {
                            builder.append("  $modelFile ($sizeStr KB): 로드 실패 (${t.localizedMessage})\n")
                            continue
                        }

                        val totalPerClass = IntArray(classNames.size)
                        val correctPerClass = IntArray(classNames.size)
                        var total = 0
                        var correct = 0

                        for (csv in csvFiles) {
                            val label = inferLabel(csv) ?: continue
                            val tensor = loadCsvAsTensor("data/$csv", groupT)
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
                            total++
                            totalPerClass[label]++
                            if (maxIdx == label) {
                                correct++
                                correctPerClass[label]++
                            }
                        }

                        val acc = if (total > 0) correct.toDouble() / total.toDouble() * 100.0 else 0.0
                        val classAccs = classNames.indices.map { i ->
                            if (totalPerClass[i] > 0) correctPerClass[i].toDouble() / totalPerClass[i].toDouble() * 100.0 else 0.0
                        }

                        pw.println(
                            "$groupName,$modelFile,$sizeStr,$total,$correct," +
                                "${formatValue(acc)}," +
                                classAccs.joinToString(",") { formatValue(it) }
                        )

                        builder.append("  $modelFile ($sizeStr KB): ${String.format("%.2f", acc)}% ($correct/$total)\n")
                        for (i in classNames.indices) {
                            if (totalPerClass[i] == 0) continue
                            val classAcc = classAccs[i]
                            builder.append("    ${classNames[i]}: ${correctPerClass[i]}/${totalPerClass[i]} (${String.format("%.2f", classAcc)}%)\n")
                        }

                        module.destroy()
                        System.gc()
                    }
                    builder.append("\n")
                }
            }

            builder.append("결과 CSV: ${outFile.absolutePath}")
            InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            InferenceResult("Accuracy 측정 중 오류: ${t.localizedMessage}", false)
        }
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

    // ===================== Utilities =====================

    private fun setButtonsEnabled(enabled: Boolean) {
        binding.basicButton.isEnabled = enabled
        binding.improvedButton.isEnabled = enabled
        binding.basicAccuracyButton.isEnabled = enabled
        binding.improvedAccuracyButton.isEnabled = enabled
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
}
