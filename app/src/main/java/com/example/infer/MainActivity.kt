package com.example.infer

import android.app.AlertDialog
import android.app.KeyguardManager
import android.content.Intent
import android.widget.EditText
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.BatteryManager
import android.os.Environment
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.documentfile.provider.DocumentFile
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
    private var originalMinFreqs = mutableMapOf<Int, String>()
    private var customQcnnPaths = listOf<String>()
    private var customSnnPaths = listOf<String>()
    private var customQcnnFolderName = ""
    private var customSnnFolderName = ""

    private val qcnnFolderPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        uri?.let { handleFolderSelected(it, isQcnn = true) }
    }
    private val snnFolderPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        uri?.let { handleFolderSelected(it, isQcnn = false) }
    }

    // CPU 주파수 고정: 코어별 scaling_min_freq = scaling_max_freq
    // 사전 조건: setenforce 0 + chmod -R 777 /sys/devices/system/cpu/cpu*/cpufreq/
    private fun lockCpuFrequency() {
        // 사전 조건: setenforce 0 + chmod -R 777 /sys/devices/system/cpu/cpu*/cpufreq/
        for (cpu in 0..7) {
            try {
                val base = "/sys/devices/system/cpu/cpu$cpu/cpufreq"
                // 현재 scaling_min_freq 저장 (원복용)
                val currentMin = File("$base/scaling_min_freq").readText().trim()
                if (currentMin.isNotEmpty()) {
                    originalMinFreqs[cpu] = currentMin
                }
                // scaling_max_freq 읽기
                val maxFreq = File("$base/scaling_max_freq").readText().trim()
                if (maxFreq.isNotEmpty()) {
                    // FileWriter로 scaling_min_freq에 max값 직접 쓰기
                    val file = File("$base/scaling_min_freq")
                    val writer = java.io.FileWriter(file)
                    writer.write(maxFreq, 0, maxFreq.length)
                    writer.flush()
                    writer.close()
                    // 검증
                    val verify = File("$base/scaling_min_freq").readText().trim()
                    android.util.Log.d("CPULock", "cpu$cpu min_freq: $currentMin → $verify (max: $maxFreq)")
                }
            } catch (e: Exception) {
                android.util.Log.e("CPULock", "cpu$cpu lock failed", e)
            }
        }
    }

    // CPU 주파수 원복: 저장해둔 원래 scaling_min_freq 복원
    private fun unlockCpuFrequency() {
        try {
            for ((cpu, minFreq) in originalMinFreqs) {
                val file = File("/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_min_freq")
                val writer = java.io.FileWriter(file)
                writer.write(minFreq, 0, minFreq.length)
                writer.flush()
                writer.close()
            }
            originalMinFreqs.clear()
        } catch (_: Exception) { }
    }

    private fun readCpuFrequencies(): String {
        val sb = StringBuilder()
        for (cpu in 0..7) {
            val freqPath = "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq"
            val mhz = try {
                // 직접 파일 읽기 시도 (root 불필요)
                val khz = File(freqPath).readText().trim()
                if (khz.isNotEmpty()) "${khz.toLong() / 1000}MHz" else "N/A"
            } catch (_: Exception) {
                try {
                    // fallback: su 사용
                    val proc = Runtime.getRuntime().exec(arrayOf("su", "-c", "cat $freqPath"))
                    val khz = proc.inputStream.bufferedReader().readText().trim()
                    proc.waitFor()
                    if (khz.isNotEmpty()) "${khz.toLong() / 1000}MHz" else "N/A"
                } catch (_: Exception) { "N/A" }
            }
            if (sb.isNotEmpty()) sb.append(", ")
            sb.append("cpu$cpu: $mhz")
        }
        return sb.toString()
    }

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

        binding.customQcnnButton.setOnClickListener {
            qcnnFolderPicker.launch(null)
        }
        binding.customSnnButton.setOnClickListener {
            snnFolderPicker.launch(null)
        }
        binding.customBmButton.setOnClickListener {
            startCustomBmMeasure()
        }
        binding.customAccuracyButton.setOnClickListener {
            startCustomAccuracy()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unlockCpuFrequency()
        releaseWakeLock()
        coroutineScope.cancel()
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
                        val sortedModels = models.sortedBy { sizeOrder(it) }

                        for (modelFile in sortedModels) {
                            val modelDisplayName = displayName(modelFile)
                            // 파일명에서 개별 T값 추출 (cv_models 등 파일마다 T가 다른 경우 대응)
                            val modelT = Regex("_T(\\d+)").find(modelFile)?.groupValues?.get(1)?.toIntOrNull() ?: groupT
                            val modulePath = if (modelFile.startsWith("/")) modelFile else assetFilePath(modelFile)
                            val modelSizeKb = File(modulePath).length() / 1024.0

                            // 텐서 프리로딩
                            val preloadedTensors = csvFiles.map { csv ->
                                loadCsvAsTensor("data/$csv", modelT)
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

                            // CPU 클럭 로그 (측정 전) — 필요 시 주석 해제
                            // val beforeFreq = readCpuFrequencies()
                            // builder.append("  [Before] $beforeFreq\n")

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
                            latPw.println("$groupName,$modelDisplayName,$sizeStr,${formatMs(totalMs)},${formatMs(avgMs)},$forwardCount,$numThreads")
                            batPw.println("$groupName,$modelDisplayName,$sizeStr,mt_differential,${formatValue(energyShort)},${formatValue(energyLong)},${formatValue(diffEnergy)},${formatValue(energyPerInference)},${formatValue(avgCurrentShort)},${formatValue(avgCurrentLong)},${formatValue(shortElapsedS)},${formatValue(longElapsedS)},${shortSamples.size},${longSamples.size},$numThreads")
                            builder.append("[$groupName] $modelDisplayName ($sizeStr KB): avg ${formatMs(avgMs)} ms, diff ${formatValue(diffEnergy)} uA·s, per_infer ${formatValue(energyPerInference)} uA·s\n")

                            // CPU 클럭 로그 (측정 후) — 필요 시 주석 해제
                            // val afterFreq = readCpuFrequencies()
                            // builder.append("  [After]  $afterFreq\n")
                            // builder.append("\n")

                            Thread.sleep(3_000)
                        }
                        builder.append("\n")
                    }
                }
            }

            builder.append("완료.\nLatency CSV: ${latencyFile.absolutePath}\nBattery CSV: ${batteryFile.absolutePath}")
            InferenceResult(builder.toString(), true)
        } catch (t: Throwable) {
            InferenceResult("측정 중 오류 발생: ${t.localizedMessage}", false)
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
                    val sortedModels = models.sortedBy { sizeOrder(it) }

                    builder.append("── $groupName (T=$groupT) ──\n")

                    for (modelFile in sortedModels) {
                        val modelDisplayName = displayName(modelFile)
                        // 파일명에서 개별 T값 추출 (cv_models 등 파일마다 T가 다른 경우 대응)
                        val modelT = Regex("_T(\\d+)").find(modelFile)?.groupValues?.get(1)?.toIntOrNull() ?: groupT
                        val modelPath = if (modelFile.startsWith("/")) modelFile else assetFilePath(modelFile)
                        val modelSizeKb = File(modelPath).length() / 1024.0
                        val sizeStr = String.format("%.1f", modelSizeKb)

                        val module = try {
                            LiteModuleLoader.load(modelPath)
                        } catch (t: Throwable) {
                            builder.append("  $modelDisplayName ($sizeStr KB): 로드 실패 (${t.localizedMessage})\n")
                            continue
                        }

                        val totalPerClass = IntArray(classNames.size)
                        val correctPerClass = IntArray(classNames.size)
                        var total = 0
                        var correct = 0

                        for (csv in csvFiles) {
                            val label = inferLabel(csv) ?: continue
                            val tensor = loadCsvAsTensor("data/$csv", modelT)
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
                            "$groupName,$modelDisplayName,$sizeStr,$total,$correct," +
                                "${formatValue(acc)}," +
                                classAccs.joinToString(",") { formatValue(it) }
                        )

                        builder.append("  $modelDisplayName ($sizeStr KB): ${String.format("%.2f", acc)}% ($correct/$total)\n")
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

    // ===================== Custom Folder Selection =====================

    private fun handleFolderSelected(uri: Uri, isQcnn: Boolean) {
        contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        val docTree = DocumentFile.fromTreeUri(this, uri) ?: return
        val folderName = docTree.name ?: "unknown"
        val ptlFiles = docTree.listFiles().filter { it.name?.endsWith(".ptl") == true }

        if (ptlFiles.isEmpty()) {
            Toast.makeText(this, "선택한 폴더에 .ptl 파일이 없습니다.", Toast.LENGTH_SHORT).show()
            return
        }

        val cacheSubDir = File(cacheDir, if (isQcnn) "custom_qcnn" else "custom_snn")
        cacheSubDir.deleteRecursively()
        cacheSubDir.mkdirs()

        val paths = ptlFiles.mapNotNull { docFile ->
            val name = docFile.name ?: return@mapNotNull null
            val outFile = File(cacheSubDir, name)
            try {
                contentResolver.openInputStream(docFile.uri)?.use { input ->
                    FileOutputStream(outFile).use { output ->
                        input.copyTo(output)
                        output.flush()
                    }
                }
                outFile.absolutePath
            } catch (_: Exception) { null }
        }.sorted()

        if (isQcnn) {
            customQcnnPaths = paths
            customQcnnFolderName = folderName
            binding.customQcnnButton.text = "QCNN ($folderName)"
        } else {
            customSnnPaths = paths
            customSnnFolderName = folderName
            binding.customSnnButton.text = "SNN ($folderName)"
        }
        Toast.makeText(this, "$folderName: ${paths.size}개 PTL 로드됨", Toast.LENGTH_SHORT).show()
    }

    // ===================== Custom BM Measurement =====================

    private fun startCustomBmMeasure() {
        if (customQcnnPaths.isEmpty() || customSnnPaths.isEmpty()) {
            Toast.makeText(this, "먼저 QCNN과 SNN 폴더를 선택하세요.", Toast.LENGTH_SHORT).show()
            return
        }
        showFileNameDialog("custom_bm") { csvPrefix ->
            launchBmMeasurement(csvPrefix)
        }
    }

    private fun launchBmMeasurement(csvPrefix: String) {
        setButtonsEnabled(false)
        binding.statusText.text = "Custom BM 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        val savedBrightness = window.attributes.screenBrightness
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }

        coroutineScope.launch {
            val startNs = System.nanoTime()
            withContext(Dispatchers.IO) { lockCpuFrequency() }
            for (i in 30 downTo 1) {
                binding.statusText.text = "Custom BM 안정화 대기 ${i}초..."
                delay(1000)
            }
            binding.statusText.text = "Custom BM 측정 중..."
            val result = withContext(Dispatchers.IO) {
                val allModels = listOf(
                    "QCNN" to customQcnnPaths,
                    "SNN" to customSnnPaths,
                )
                runDifferentialMeasurement(allModels, csvPrefix)
            }
            val elapsed = formatElapsed(System.nanoTime() - startNs)
            window.attributes = window.attributes.apply { screenBrightness = savedBrightness }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료 (소요시간: $elapsed)" else "오류 발생 ($elapsed)"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            unlockCpuFrequency()
            releaseWakeLock()
        }
    }

    // ===================== Custom Accuracy =====================

    private fun startCustomAccuracy() {
        if (customQcnnPaths.isEmpty() || customSnnPaths.isEmpty()) {
            Toast.makeText(this, "먼저 QCNN과 SNN 폴더를 선택하세요.", Toast.LENGTH_SHORT).show()
            return
        }
        showFileNameDialog("custom_accuracy") { csvPrefix ->
            launchAccuracyMeasurement(csvPrefix)
        }
    }

    private fun launchAccuracyMeasurement(csvPrefix: String) {
        setButtonsEnabled(false)
        binding.statusText.text = "Custom Accuracy 측정 중..."
        binding.resultText.text = ""
        acquireWakeLock()

        val savedBrightness = window.attributes.screenBrightness
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }

        coroutineScope.launch {
            val startNs = System.nanoTime()
            withContext(Dispatchers.IO) { lockCpuFrequency() }
            binding.statusText.text = "Custom Accuracy 측정 중..."
            val result = withContext(Dispatchers.IO) {
                val allModels = listOf(
                    "QCNN" to customQcnnPaths,
                    "SNN" to customSnnPaths,
                )
                runAccuracyComparison(allModels, csvPrefix)
            }
            val elapsed = formatElapsed(System.nanoTime() - startNs)
            window.attributes = window.attributes.apply { screenBrightness = savedBrightness }
            binding.resultText.text = result.message
            binding.statusText.text = if (result.success) "완료 (소요시간: $elapsed)" else "오류 발생 ($elapsed)"
            setButtonsEnabled(true)
            wakeUpScreen()
            clearScreenFlags()
            unlockCpuFrequency()
            releaseWakeLock()
        }
    }

    private fun showFileNameDialog(defaultPrefix: String, onResult: (String) -> Unit) {
        val input = EditText(this).apply {
            setText(defaultPrefix)
            selectAll()
            setPadding(48, 32, 48, 16)
        }
        AlertDialog.Builder(this)
            .setTitle("결과 파일명 입력")
            .setMessage("파일명 prefix를 입력하세요.")
            .setView(input)
            .setPositiveButton("확인") { _, _ ->
                val text = input.text.toString().trim()
                onResult(if (text.isNotEmpty()) text else defaultPrefix)
            }
            .setNegativeButton("취소") { _, _ ->
                onResult(defaultPrefix)
            }
            .setCancelable(false)
            .show()
    }

    // ===================== Utilities =====================

    private fun setButtonsEnabled(enabled: Boolean) {
        binding.customQcnnButton.isEnabled = enabled
        binding.customSnnButton.isEnabled = enabled
        binding.customBmButton.isEnabled = enabled
        binding.customAccuracyButton.isEnabled = enabled
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
        outFile.parentFile?.mkdirs()
        assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
                output.flush()
            }
        }
        return outFile.absolutePath
    }

    private fun sizeOrder(name: String): Int {
        val lower = File(name).name.lowercase()
        return when {
            lower.contains("smallest") -> 0
            lower.contains("small") && !lower.contains("smallest") -> 1
            lower.contains("medium") -> 2
            lower.contains("largest") -> 4
            lower.contains("large") -> 3
            else -> 5
        }
    }

    private fun displayName(path: String): String = File(path).name

    private fun formatMs(value: Double): String = String.format("%.3f", value)
    private fun formatValue(value: Double): String = String.format("%.6f", value)

    private fun formatElapsed(ns: Long): String {
        val totalSec = ns / 1_000_000_000
        val h = totalSec / 3600
        val m = (totalSec % 3600) / 60
        val s = totalSec % 60
        return if (h > 0) "${h}시간 ${m}분 ${s}초" else if (m > 0) "${m}분 ${s}초" else "${s}초"
    }

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
