// 1. 기존에 있던 plugins { ... } 블록을 이 내용으로 교체합니다.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.infer" // 본인 프로젝트의 namespace로 맞추세요.
    compileSdk = 34 // 또는 35, 36 등

    defaultConfig {
        applicationId = "com.example.infer" // 본인 프로젝트의 applicationId로 맞추세요.
        minSdk = 26
        targetSdk = 34 // compileSdk와 맞추는 것을 권장합니다.
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    // viewBinding을 사용하고 있다면 이 블록을 추가하세요.
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // PyTorch 라이브러리 추가
    implementation("org.pytorch:pytorch_android_lite:2.1.0")
    implementation("org.pytorch:pytorch_android_torchvision_lite:2.1.0")

    // libs.versions.toml에 정의된 라이브러리들을 사용
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.documentfile:documentfile:1.0.1")
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
