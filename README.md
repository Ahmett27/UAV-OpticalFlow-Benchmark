# 🚁 UAV Optical Flow Benchmarker & SLAM Trajectory Mapper

This project is a comprehensive benchmarking tool that analyzes the accuracy of classical and deep learning-based optical flow algorithms using synthetic flight data (AirSim + PX4/MAVLink) obtained from Unmanned Aerial Vehicles (UAVs) equipped with a fixed downward-facing (nadir) camera.

The project goes beyond calculating standard computer vision metrics; it performs **Sensor Fusion** using IMU and GPS data from the Autopilot and generates a 2D SLAM-like flight trajectory map.

## ✨ Key Features

* **Algorithm Comparison:** Evaluates deep learning-based AI models (PTLFlow - RAFT) alongside classical computer vision algorithms (Farneback, Lucas-Kanade) simultaneously.
* **Digital Gimbal (Mathematical Stabilization):** Filters out the camera shake (Rotational Flow) caused by the UAV's braking and maneuvering using angular velocity data (`pitchspeed`, `rollspeed`) from the IMU, isolating only the translational movement.
* **Dynamic Kinematic Transformation:** Converts Global Earth Frame GPS velocity vectors (North/East) into the Camera's Body Frame (Forward/Right) dynamically using the drone's instant compass heading (`yaw`).
* **Sub-pixel Accuracy:** Calculates `EPE (End-Point Error)` and `AE (Angular Error)` metrics with sub-pixel precision through velocity and time ($\Delta t$) integrations.
* **SLAM-like Trajectory Map:** Cumulatively stitches together the instantaneous pixel shift estimates of the algorithms to draw a comparative 2D SLAM-like trajectory against the Ground Truth.

## 🛠️ Installation & Requirements

The following libraries are required to run the project:

pip install numpy pandas opencv-python matplotlib torch ptlflow
(Note: An NVIDIA CUDA-supported system is highly recommended for faster execution of deep learning models.)



📁 Dataset Hierarchy (Input Format)
For the code to work, your dataset must have the following folder structure and MAVLink-standard CSV outputs:
flight_data_folder/
├── video/
│   ├── nadir_1280x720_30fps.mp4
│   └── frame_timestamps.csv
├── telemetry/
│   └── attitude_log.csv
└── ground_truth/
    └── expected_flow.csv


    🚀 Usage
Simply initialize the OpticalFlowBenchmarker class and specify your dataset path:
from benchmarker import OpticalFlowBenchmarker

# Specify the dataset path
DATASET_KLASORU = "/path/to/your/flight_data"

# Initialize the class and run the benchmark
benchmarker = OpticalFlowBenchmarker(dataset_dir=DATASET_KLASORU, ptlflow_model_name='raft_small')
benchmarker.run_benchmark(max_frames=200, warmup_frames=50)

# warmup_frames: The number of initial frames to exclude from the analysis to allow the navigation filters (EKF) to converge and to skip startup noise/takeoff vibrations.


📊 Outputs
Upon completion, the code generates the following files in the main directory:

benchmark_results.csv: Detailed U/V pixel shifts and EPE/AE error rates for each frame.

benchmark_report.png: A comprehensive analysis chart including processing speed (FPS), systematic bias, error time series, and quiver plots.

trajectory_map.png: A 2D bird's-eye SLAM comparison of the true flight path (Ground Truth) versus the paths estimated by the optical flow algorithms.




🇹🇷 TR - Türkçe
Bu proje, sabit aşağı bakan (nadir) kameraya sahip İnsansız Hava Araçlarından (İHA) elde edilen sentetik uçuş verileri (AirSim + PX4/MAVLink) üzerinden optik akış (optical flow) algoritmalarının doğruluk paylarını analiz eden kapsamlı bir benchmark aracıdır.

Proje, sadece görüntü işleme metriklerini hesaplamakla kalmaz; aynı zamanda Otopilot'tan gelen IMU ve GPS verilerini kullanarak Sensör Füzyonu (Sensor Fusion) yapar ve drone'un 2D SLAM benzeri uçuş rotasını haritalandırır.

✨ Öne Çıkan Özellikler
Algoritma Karşılaştırması: Derin öğrenme tabanlı yapay zeka modelleri (PTLFlow - RAFT) ile klasik bilgisayarlı görü algoritmalarını (Farneback, Lucas-Kanade) aynı anda test eder.
Dijital Gimbal (Matematiksel Stabilizasyon): İHA'nın frenleme ve manevra anlarındaki sarsıntılarını (Rotational Flow), IMU'dan alınan açısal hız (pitchspeed, rollspeed) verileriyle filtreler. Sadece öteleme (Translational) hareketini izole eder.
Dinamik Kinematik Dönüşüm: Drone'un anlık pusula yönünü (yaw) kullanarak, Global Dünya Eksenindeki (Kuzey/Doğu) GPS hız vektörlerini, Kameranın Gövde Eksenine (İleri/Sağa) dinamik olarak dönüştürür.
Piksel Altı (Sub-pixel) Hassasiyet: Hız ve zaman ($\Delta t$) entegrasyonları sayesinde EPE (End-Point Error) ve AE (Angular Error) metriklerini piksel altı doğrulukla hesaplar.
SLAM Benzeri Rota Haritası: Algoritmaların anlık piksel kayma tahminlerini kümülatif olarak uç uca ekleyerek, referans rota (Ground Truth) ile tahmin edilen 2D rotayı karşılaştırmalı olarak çizer.

🛠️ Kurulum ve GereksinimlerProjeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
pip install numpy pandas opencv-python matplotlib torch ptlflow
(Not: Derin öğrenme modellerinin hızlı çalışması için NVIDIA CUDA destekli bir sistem önerilir.)

📁 Veri Seti Hiyerarşisi (Girdi Formatı)
Kodun çalışabilmesi için veri setinin aşağıdaki klasör yapısına ve MAVLink standartlarında CSV çıktılarına sahip olması beklenir:
flight_data_folder/
├── video/
│   ├── nadir_1280x720_30fps.mp4
│   └── frame_timestamps.csv
├── telemetry/
│   └── attitude_log.csv
└── ground_truth/
    └── expected_flow.csv


🚀 Kullanım
OpticalFlowBenchmarker sınıfını başlatıp veri setinizin yolunu göstermeniz yeterlidir:
from benchmarker import OpticalFlowBenchmarker

# Veri seti yolunu belirtin
DATASET_KLASORU = "/path/to/your/flight_data"

# Sınıfı başlatın ve testi çalıştırın
benchmarker = OpticalFlowBenchmarker(dataset_dir=DATASET_KLASORU, ptlflow_model_name='raft_small')
benchmarker.run_benchmark(max_frames=200, warmup_frames=50)

# warmup_frames: Navigasyon filtrelerinin (EKF) yakınsaması ve kalkış sarsıntılarının (startup noise) atlanması için analiz dışı bırakılacak başlangıç kare sayısıdır.

📊 Çıktılar (Outputs)
Çalıştırma tamamlandığında, kod ana dizine aşağıdaki dosyaları üretir:

benchmark_results.csv: Her kare için detaylı U/V piksel kaymaları ve EPE/AE hata oranları.

benchmark_report.png: İşlem hızı (FPS), sistematik önyargı (Bias), hata zaman serileri ve vektörel akış yönlerini (Quiver) içeren kapsamlı analiz grafiği.

trajectory_map.png: Gerçek (Ground Truth) uçuş rotası ile optik akış algoritmalarının tahmin ettiği rotaların kuş bakışı (2D) SLAM karşılaştırması.
