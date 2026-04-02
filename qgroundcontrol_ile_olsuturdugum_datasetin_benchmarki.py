import os
import cv2
import math
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from ptlflow import get_model
from ptlflow.utils.utils import InputPadder

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GRAY   = "\033[90m"

class OpticalFlowBenchmarker:
    """
    PTLFlow / Farneback / Lucas-Kanade karşılaştırma aracı.

    Güncellemeler:
        • PTLFlow ön eğitim ağırlıklarıyla (ckpt_path='things') başlatılıyor.
        • Padding işlemi optimize edildi.
        • GT verisindeki kümülatif artıştan (toplam kayma) fark alınarak anlık optik akış (px/frame) bulunuyor.
    """

    def __init__(self, dataset_dir: str, ptlflow_model_name: str = 'raft_small',
                 gt_rotation: int = 0):
        self.dataset_dir = dataset_dir
        self.gt_rotation = gt_rotation
        self.att_path = os.path.join(dataset_dir, "telemetry", "attitude_log.csv")
        self.df_att   = pd.read_csv(self.att_path)

        self.video_path      = self._find_video_file()
        self.timestamps_path = os.path.join(dataset_dir, "video", "frame_timestamps.csv")
        self.gt_path         = os.path.join(dataset_dir, "ground_truth", "expected_flow.csv")

        print(f"{CYAN}[Sistem]{RESET} Veri seti okunuyor…")
        self.df_times = pd.read_csv(self.timestamps_path)
        self.df_gt    = pd.read_csv(self.gt_path)

        print(f"{CYAN}[Sistem]{RESET} PTLFlow modeli ({ptlflow_model_name}) yükleniyor…")
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PTLFlow önceden eğitilmiş ağırlıkları yükleme
        self.ptl_model = get_model(ptlflow_model_name, ckpt_path='things').to(self.device)
        self.ptl_model.eval()
        print(f"{GREEN}[OK]{RESET} Model cihaz: {self.device}")

        self.results: list[dict] = []
        self._ptl_debug_count = 0

        print(f"\n{CYAN}[GT Kontrol]{RESET} expected_flow.csv ilk 3 satır:")
        print(self.df_gt.head(3).to_string())
        print(f"{CYAN}[GT Kontrol]{RESET} Kolon adları: {list(self.df_gt.columns)}\n")

    def _find_video_file(self) -> str:
        video_dir = os.path.join(self.dataset_dir, "video")
        for f in os.listdir(video_dir):
            if f.endswith((".mp4", ".avi", ".mkv")):
                return os.path.join(video_dir, f)
        raise FileNotFoundError("Video dosyası bulunamadı!")

    def _flow_mean_filtered(self, u_map: np.ndarray, v_map: np.ndarray,
                             pct: int = 70) -> tuple[float, float]:
        magnitude = np.sqrt(u_map ** 2 + v_map ** 2)
        low_thresh  = np.percentile(magnitude, 20)
        q75 = np.percentile(magnitude, 75)
        q25 = np.percentile(magnitude, 25)
        iqr = q75 - q25
        high_thresh = q75 + 3.0 * iqr

        mask = (magnitude > low_thresh) & (magnitude < high_thresh)
        if mask.sum() == 0:
            mask = magnitude > low_thresh
        if mask.sum() == 0:
            return 0.0, 0.0

        return float(np.median(u_map[mask])), float(np.median(v_map[mask]))

    def get_interpolated_ground_truth(self, timestamp: float) -> tuple[float, float]:
        # --- 1. ÖTELEME (TRANSLATIONAL) AKIŞI (Global Eksen: Kuzey/Doğu) ---
        raw_u = float(np.interp(timestamp, self.df_gt['timestamp'], self.df_gt['flow_u']))
        raw_v = float(np.interp(timestamp, self.df_gt['timestamp'], self.df_gt['flow_v']))

        # --- YENİ: DİNAMİK YAW AÇISI ---
        # Sabit 45 derece yerine, o milisaniyedeki gerçek pusula yönünü (Yaw) çekiyoruz!
        yaw_rad = float(np.interp(timestamp, self.df_att['timestamp'], self.df_att['yaw_rad']))

        # Pusula (Global) -> Kamera (Body) Ekseni Dönüşümü
        # (Drone'un anlık yönüne göre Kuzey/Doğu hızlarını İleri/Sağa hızlarına çeviririz)
        trans_v = raw_v * math.cos(yaw_rad) + raw_u * math.sin(yaw_rad)
        trans_u = raw_v * math.sin(yaw_rad) - raw_u * math.cos(yaw_rad)

        # --- 2. DÖNME (ROTATIONAL) AKIŞI (MATEMATİKSEL GİMBAL) ---
        pitchspeed = float(np.interp(timestamp, self.df_att['timestamp'], self.df_att['pitchspeed_rads']))
        rollspeed  = float(np.interp(timestamp, self.df_att['timestamp'], self.df_att['rollspeed_rads']))

        focal_length = 640.0
        # Sarsıntıyı piksel hızına çeviriyoruz
        rot_v = pitchspeed * focal_length
        rot_u = -rollspeed * focal_length

        # --- 3. TOPLAM GERÇEK OPTİK AKIŞ ---
        corrected_u = trans_u + rot_u
        corrected_v = trans_v + rot_v

        if not hasattr(self, '_gt_debug_count'):
            self._gt_debug_count = 0
        if self._gt_debug_count < 5:
            print(f"  {YELLOW}[GT DEBUG]{RESET} "
                  f"Yaw={math.degrees(yaw_rad):.1f}° | "
                  f"Oteleme=(u:{trans_u:+.1f}, v:{trans_v:+.1f}) | "
                  f"Donme=(u:{rot_u:+.1f}, v:{rot_v:+.1f})")
            self._gt_debug_count += 1

        return -corrected_u, corrected_v

    @torch.no_grad()
    def run_ptlflow(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
        def to_tensor(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        t1, t2 = to_tensor(img1), to_tensor(img2)

        images = torch.stack([t1, t2], dim=1)  # (1, 2, C, H, W)

        stride_val = getattr(self.ptl_model, 'stride', 8)
        padder = InputPadder(images.shape, stride_val)

        images_padded = padder.pad(images)
        preds = self.ptl_model({'images': images_padded})

        flow_padded = preds['flows'][0, 0]          
        flow = padder.unpad(flow_padded).cpu().numpy()

        if self._ptl_debug_count < 5:
            mag = np.sqrt(flow[0]**2 + flow[1]**2)
            print(f"  {CYAN}[PTL DEBUG]{RESET} "
                  f"u: med={np.median(flow[0]):+.2f} mean={flow[0].mean():+.2f} "
                  f"min={flow[0].min():+.2f} max={flow[0].max():+.2f}  |  "
                  f"v: med={np.median(flow[1]):+.2f} mean={flow[1].mean():+.2f}  |  "
                  f"mag: med={np.median(mag):.2f} max={mag.max():.2f}")
            self._ptl_debug_count += 1

        return self._flow_mean_filtered(flow[0], flow[1])

    def run_farneback(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow  = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
        return self._flow_mean_filtered(flow[..., 0], flow[..., 1])

    def run_lucas_kanade(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=200,
                                      qualityLevel=0.3, minDistance=7)
        if p0 is None:
            return 0.0, 0.0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None,
                                               winSize=(15, 15), maxLevel=2)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) == 0:
            return 0.0, 0.0

        mv = good_new - good_old
        return float(np.mean(mv[:, 0])), float(np.mean(mv[:, 1]))

    @staticmethod
    def metric_epe(pu, pv, gu, gv) -> float:
        return math.sqrt((pu - gu) ** 2 + (pv - gv) ** 2)

    @staticmethod
    def metric_rmse(pu, pv, gu, gv) -> float:
        return math.sqrt(((pu - gu) ** 2 + (pv - gv) ** 2) / 2)

    @staticmethod
    def metric_angle_error(pu, pv, gu, gv) -> float:
        eps = 1e-6
        dot = pu * gu + pv * gv
        mp  = math.sqrt(pu**2 + pv**2) + eps
        mg  = math.sqrt(gu**2 + gv**2) + eps
        cos = max(-1.0, min(1.0, dot / (mp * mg)))
        return math.degrees(math.acos(cos))

    @staticmethod
    def metric_relative_error(pu, pv, gu, gv) -> float:
        gt_mag = math.sqrt(gu**2 + gv**2)
        if gt_mag < 1e-6:
            return 0.0
        return (OpticalFlowBenchmarker.metric_epe(pu, pv, gu, gv) / gt_mag) * 100

    @staticmethod
    def metric_bias(pred_vals: list[float], gt_vals: list[float]) -> float:
        diffs = [p - g for p, g in zip(pred_vals, gt_vals)]
        return float(np.mean(diffs))

    def run_benchmark(self, max_frames: int = 200, warmup_frames: int = 50):
        print(f"\n{BOLD}[Test]{RESET} Başlıyor… (maksimum {max_frames} kare)\n")
        cap = cv2.VideoCapture(self.video_path)

        ret, prev_frame = cap.read()
        frame_idx = 0

        quiver_data = {
            'frames': [],
            'gt':  {'u': [], 'v': []},
            'ptl': {'u': [], 'v': []},
            'fb':  {'u': [], 'v': []},
            'lk':  {'u': [], 'v': []},
        }
        timing = {'ptl': [], 'fb': [], 'lk': []}

        while ret and frame_idx < max_frames - 1:
            ret, next_frame = cap.read()
            if not ret:
                break

            # YENİ: İlk X kareyi analizden hariç tut (Startup Noise / Sarsıntı engelleme)
            if frame_idx < warmup_frames:
                prev_frame = next_frame.copy()
                frame_idx += 1
                continue

            t1_ts   = self.df_times.iloc[frame_idx]['timestamp']
            t2_ts   = self.df_times.iloc[frame_idx + 1]['timestamp']
            delta_t = t2_ts - t1_ts

            # CSV'deki saniyedeki hızı çekiyoruz
            gt_u_sec, gt_v_sec = self.get_interpolated_ground_truth(t1_ts)

            # HIZI ZAMANLA ÇARPARAK İKİ KARE ARASI PİKSEL KAYMASINI BULUYORUZ
            gt_u = gt_u_sec * delta_t
            gt_v = gt_v_sec * delta_t

            if abs(gt_u) > 0.01 or abs(gt_v) > 0.01:

                t0 = time.perf_counter()
                ptl_u, ptl_v = self.run_ptlflow(prev_frame, next_frame)
                timing['ptl'].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                fb_u, fb_v = self.run_farneback(prev_frame, next_frame)
                timing['fb'].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                lk_u, lk_v = self.run_lucas_kanade(prev_frame, next_frame)
                timing['lk'].append(time.perf_counter() - t0)

                row = {
                    'frame': frame_idx,
                    'gt_u': gt_u, 'gt_v': gt_v,
                    'ptl_u': ptl_u, 'ptl_v': ptl_v,
                    'fb_u':  fb_u,  'fb_v':  fb_v,
                    'lk_u':  lk_u,  'lk_v':  lk_v,
                    'epe_ptl':  self.metric_epe(ptl_u, ptl_v, gt_u, gt_v),
                    'epe_fb':   self.metric_epe(fb_u,  fb_v,  gt_u, gt_v),
                    'epe_lk':   self.metric_epe(lk_u,  lk_v,  gt_u, gt_v),
                    'rmse_ptl': self.metric_rmse(ptl_u, ptl_v, gt_u, gt_v),
                    'rmse_fb':  self.metric_rmse(fb_u,  fb_v,  gt_u, gt_v),
                    'rmse_lk':  self.metric_rmse(lk_u,  lk_v,  gt_u, gt_v),
                    'ae_ptl':   self.metric_angle_error(ptl_u, ptl_v, gt_u, gt_v),
                    'ae_fb':    self.metric_angle_error(fb_u,  fb_v,  gt_u, gt_v),
                    'ae_lk':    self.metric_angle_error(lk_u,  lk_v,  gt_u, gt_v),
                    're_ptl':   self.metric_relative_error(ptl_u, ptl_v, gt_u, gt_v),
                    're_fb':    self.metric_relative_error(fb_u,  fb_v,  gt_u, gt_v),
                    're_lk':    self.metric_relative_error(lk_u,  lk_v,  gt_u, gt_v),
                }
                self.results.append(row)

                quiver_data['frames'].append(frame_idx)
                for comp, val_u, val_v in [
                    ('gt',  gt_u,  gt_v),
                    ('ptl', ptl_u, ptl_v),
                    ('fb',  fb_u,  fb_v),
                    ('lk',  lk_u,  lk_v),
                ]:
                    quiver_data[comp]['u'].append(val_u)
                    quiver_data[comp]['v'].append(val_v)

                print(
                    f"{GRAY}───── Kare {frame_idx:03d} ─────{RESET}\n"
                    f"  Gerçek  GT        : u={gt_u:+8.3f} px   v={gt_v:+8.3f} px\n"
                    f"  {CYAN}PTLFlow tahmini{RESET}   : u={ptl_u:+8.3f} px   v={ptl_v:+8.3f} px   EPE={row['epe_ptl']:.2f} px   AE={row['ae_ptl']:.1f}°\n"
                    f"  {YELLOW}Farneback tahmini{RESET} : u={fb_u:+8.3f} px   v={fb_v:+8.3f} px   EPE={row['epe_fb']:.2f} px   AE={row['ae_fb']:.1f}°\n"
                    f"  {RED}LucasKanade tahmini{RESET}: u={lk_u:+8.3f} px   v={lk_v:+8.3f} px   EPE={row['epe_lk']:.2f} px   AE={row['ae_lk']:.1f}°"
                )

            prev_frame = next_frame.copy()
            frame_idx += 1

        cap.release()
        self._generate_report(quiver_data, timing)

    def _generate_report(self, quiver_data: dict, timing: dict):
        df = pd.DataFrame(self.results)
        if len(df) == 0:
            print(f"{RED}[Hata]{RESET} Hareketli kare bulunamadı.")
            return

        bias_ptl_u = self.metric_bias(df['ptl_u'].tolist(), df['gt_u'].tolist())
        bias_ptl_v = self.metric_bias(df['ptl_v'].tolist(), df['gt_v'].tolist())
        bias_fb_u  = self.metric_bias(df['fb_u'].tolist(),  df['gt_u'].tolist())
        bias_fb_v  = self.metric_bias(df['fb_v'].tolist(),  df['gt_v'].tolist())
        bias_lk_u  = self.metric_bias(df['lk_u'].tolist(),  df['gt_u'].tolist())
        bias_lk_v  = self.metric_bias(df['lk_v'].tolist(),  df['gt_v'].tolist())

        fps_ptl = 1.0 / np.mean(timing['ptl']) if timing['ptl'] else 0
        fps_fb  = 1.0 / np.mean(timing['fb'])  if timing['fb']  else 0
        fps_lk  = 1.0 / np.mean(timing['lk'])  if timing['lk']  else 0

        sep = "=" * 62
        print(f"\n{BOLD}{sep}")
        print(f"   OPTİK AKIŞ BENCHMARK SONUÇLARI  (GT {self.gt_rotation}° düzeltmeli)")
        print(f"{sep}{RESET}")
        header = f"{'Metrik':<22} {'PTLFlow':>12} {'Farneback':>12} {'LucasKanade':>12}"
        print(f"{BOLD}{header}{RESET}")
        print("-" * 62)

        metrics = [
            ("Ort. EPE (px)",     df['epe_ptl'].mean(),  df['epe_fb'].mean(),  df['epe_lk'].mean()),
            ("Ort. RMSE (px)",    df['rmse_ptl'].mean(), df['rmse_fb'].mean(), df['rmse_lk'].mean()),
            ("Ort. Açı Hat. (°)", df['ae_ptl'].mean(),   df['ae_fb'].mean(),   df['ae_lk'].mean()),
            ("Ort. Göreceli (%)", df['re_ptl'].mean(),   df['re_fb'].mean(),   df['re_lk'].mean()),
            ("Bias U (px)",       bias_ptl_u,            bias_fb_u,            bias_lk_u),
            ("Bias V (px)",       bias_ptl_v,            bias_fb_v,            bias_lk_v),
            ("FPS",               fps_ptl,               fps_fb,               fps_lk),
        ]
        for label, v1, v2, v3 in metrics:
            print(f"  {label:<20} {v1:>12.3f} {v2:>12.3f} {v3:>12.3f}")

        print(f"{BOLD}{sep}{RESET}")
        print(f"{GRAY}* Daha düşük EPE / RMSE / AE / Göreceli → daha iyi{RESET}")
        print(f"{GRAY}* Bias ≈ 0 → sistematik önyargı yok{RESET}\n")

        csv_path = os.path.join(self.dataset_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"{GREEN}[OK]{RESET} CSV → {csv_path}")

        self._plot_all(df, quiver_data,
                       bias_ptl_u, bias_ptl_v,
                       bias_fb_u,  bias_fb_v,
                       bias_lk_u,  bias_lk_v,
                       fps_ptl, fps_fb, fps_lk)
        
        self._plot_trajectory(df)

    def _plot_all(self, df, qd,
                  bp_u, bp_v, bf_u, bf_v, bl_u, bl_v,
                  fps_ptl, fps_fb, fps_lk):

        plt.style.use('dark_background')
        C = {'gt': '#FFFFFF', 'ptl': '#00D4FF', 'fb': '#FFD700', 'lk': '#FF6B6B'}
        frames = np.array(qd['frames'])

        fig = plt.figure(figsize=(22, 26), facecolor='#0D1117')
        gs  = gridspec.GridSpec(4, 3, figure=fig,
                                hspace=0.45, wspace=0.35,
                                left=0.07, right=0.96,
                                top=0.94, bottom=0.04)

        fig.suptitle(f"Optik Akış Benchmark Raporu  (GT: AirSim {self.gt_rotation}° düzeltmeli)",
                     fontsize=18, fontweight='bold', color='white', y=0.97)

        ax = fig.add_subplot(gs[0, :])
        for key, label, color in [('epe_ptl','PTLFlow',C['ptl']),
                                   ('epe_fb','Farneback',C['fb']),
                                   ('epe_lk','Lucas-Kanade',C['lk'])]:
            ax.plot(df['frame'], df[key], color=color, lw=1.5, alpha=0.9, label=label)
        self._style_ax(ax, "End-Point Error (EPE) — Kare Zaman Serisi", "Kare", "EPE (piksel)")
        ax.legend(facecolor='#1C2333', edgecolor='#333', labelcolor='white')

        ax = fig.add_subplot(gs[1, :2])
        for key, label, color in [('ae_ptl','PTLFlow',C['ptl']),
                                   ('ae_fb','Farneback',C['fb']),
                                   ('ae_lk','Lucas-Kanade',C['lk'])]:
            ax.plot(df['frame'], df[key], color=color, lw=1.3, alpha=0.85, label=label)
        self._style_ax(ax, "Açısal Hata (°)", "Kare", "Derece")
        ax.legend(facecolor='#1C2333', edgecolor='#333', labelcolor='white')

        ax = fig.add_subplot(gs[1, 2])
        labels_f = ['PTLFlow', 'Farneback', 'LucasKanade']
        vals_f   = [fps_ptl, fps_fb, fps_lk]
        bars = ax.barh(labels_f, vals_f, color=[C['ptl'], C['fb'], C['lk']], alpha=0.85, height=0.5)
        for bar, v in zip(bars, vals_f):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} fps", va='center', color='white', fontsize=9)
        self._style_ax(ax, "İşlem Hızı (FPS)", "Kare/Saniye", "")

        for col_idx, (key, title, color) in enumerate([
            ('gt',  'Ground Truth (düzeltmeli)', C['gt']),
            ('ptl', 'PTLFlow',                   C['ptl']),
            ('fb',  'Farneback',                  C['fb']),
        ]):
            ax = fig.add_subplot(gs[2, col_idx])
            self._draw_quiver(ax, frames, np.array(qd[key]['u']), np.array(qd[key]['v']), color, title)

        ax = fig.add_subplot(gs[3, 0])
        self._draw_quiver(ax, frames, np.array(qd['lk']['u']), np.array(qd['lk']['v']), C['lk'], 'Lucas-Kanade')

        ax = fig.add_subplot(gs[3, 1])
        b_labels = ['PTL U', 'PTL V', 'FB U', 'FB V', 'LK U', 'LK V']
        b_vals   = [bp_u, bp_v, bf_u, bf_v, bl_u, bl_v]
        b_cols   = [C['ptl']]*2 + [C['fb']]*2 + [C['lk']]*2
        bars = ax.bar(b_labels, b_vals, color=b_cols, alpha=0.85, width=0.5)
        ax.axhline(0, color='white', lw=0.8, linestyle='--', alpha=0.5)
        max_abs = max(abs(v) for v in b_vals) if b_vals else 1
        for b, v in zip(bars, b_vals):
            offset = max_abs * 0.03
            ax.text(b.get_x() + b.get_width() / 2, v + (offset if v >= 0 else -offset * 3),
                    f"{v:.2f}", ha='center', color='white', fontsize=7.5)
        self._style_ax(ax, "Sistematik Önyargı (Bias, px)", "", "Ortalama Fark (px)")
        ax.tick_params(axis='x', labelsize=8)

        ax = fig.add_subplot(gs[3, 2])
        bp = ax.boxplot([df['re_ptl'].values, df['re_fb'].values, df['re_lk'].values],
                        patch_artist=True, medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#888'), capprops=dict(color='#888'),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))
        for patch, color in zip(bp['boxes'], [C['ptl'], C['fb'], C['lk']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(['PTLFlow', 'Farneback', 'LK'], color='white')
        self._style_ax(ax, "Göreceli Hata Dağılımı (%)", "", "%")

        out = os.path.join(self.dataset_dir, "benchmark_report.png")
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1117')
        plt.close(fig)
        print(f"{GREEN}[OK]{RESET} Grafik → {out}")

    @staticmethod
    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, color='white', fontsize=12)
        ax.set_xlabel(xlabel, color='#AAAAAA')
        ax.set_ylabel(ylabel, color='#AAAAAA')
        ax.set_facecolor('#0D1117')
        ax.grid(color='#222', linewidth=0.5)
        ax.tick_params(colors='#888')

    @staticmethod
    def _draw_quiver(ax, frames, u_vals, v_vals, color, title):
        n = len(frames)
        if n == 0:
            return
        y   = np.zeros(n)
        mag = np.sqrt(u_vals**2 + v_vals**2) + 1e-9
        u_n = u_vals / mag
        v_n = v_vals / mag
        ax.quiver(frames, y, u_n, v_n, mag,
                  cmap=LinearSegmentedColormap.from_list('flow', ['#111111', color], N=256),
                  scale=30, width=0.005, headwidth=4, headlength=5, alpha=0.9)
        ax.set_title(title, color='white', fontsize=11)
        ax.set_facecolor('#0D1117')
        ax.set_xlabel("Kare", color='#AAAAAA', fontsize=8)
        ax.set_yticks([])
        ax.tick_params(colors='#888')
        ax.set_xlim(frames[0] - 1, frames[-1] + 1)
        ax.set_ylim(-1, 1)

    def _plot_trajectory(self, df):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0D1117')
        ax.set_facecolor('#0D1117')

        # Başlangıç noktası (0,0)
        gt_x, gt_y = [0], [0]
        ptl_x, ptl_y = [0], [0]
        fb_x, fb_y = [0], [0]
        lk_x, lk_y = [0], [0]

        # SLAM Mantığı: Anlık kaymaları uç uca ekleyerek toplam rotayı (kümülatif) buluyoruz
        for _, row in df.iterrows():
            gt_x.append(gt_x[-1] + row['gt_u'])
            gt_y.append(gt_y[-1] - row['gt_v'])  # Kuzey yukarı olsun diye V'yi ters çeviriyoruz
            
            ptl_x.append(ptl_x[-1] + row['ptl_u'])
            ptl_y.append(ptl_y[-1] - row['ptl_v'])
            
            fb_x.append(fb_x[-1] + row['fb_u'])
            fb_y.append(fb_y[-1] - row['fb_v'])
            
            lk_x.append(lk_x[-1] + row['lk_u'])
            lk_y.append(lk_y[-1] - row['lk_v'])

        # Çizimler
        ax.plot(gt_x, gt_y, label='Gerçek Rota (GT)', color='#FFFFFF', lw=3, linestyle='--')
        ax.plot(ptl_x, ptl_y, label='PTLFlow Tahmini', color='#00D4FF', lw=2, alpha=0.9)
        ax.plot(fb_x, fb_y, label='Farneback Tahmini', color='#FFD700', lw=1.5, alpha=0.8)
        ax.plot(lk_x, lk_y, label='Lucas-Kanade Tahmini', color='#FF6B6B', lw=1.5, alpha=0.8)

        # Başlangıç ve Bitiş noktalarını işaretleme
        ax.scatter([0], [0], color='#00FF00', s=100, label='Başlangıç Noktası', zorder=5)
        ax.scatter(gt_x[-1], gt_y[-1], color='#FF0000', s=100, label='Bitiş Noktası (GT)', zorder=5)

        ax.set_title("SLAM Benzeri 2D Uçuş Rotası Karşılaştırması", color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel("X Ekseni Kayması (Piksel)", color='#AAAAAA')
        ax.set_ylabel("Y Ekseni Kayması (Piksel)", color='#AAAAAA')
        ax.legend(facecolor='#1C2333', edgecolor='#333', labelcolor='white')
        ax.grid(color='#222', linewidth=0.5)
        
        # Eksenleri eşitle (Harita yamuk durmasın, oranlar korunsun)
        plt.axis('equal') 

        # Kaydet
        out = os.path.join(self.dataset_dir, "trajectory_map.png")
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1117')
        plt.close(fig)
        print(f"{GREEN}[OK]{RESET} SLAM Rota Haritası → {out}")

if __name__ == "__main__":
    DATASET_KLASORU = os.path.expanduser(
        "/home/ahmet/flight_data/mavigol_200m"  # ← kendi yolunu gir
    )

    benchmarker = OpticalFlowBenchmarker(
        dataset_dir=DATASET_KLASORU,
        ptlflow_model_name='raft_small',
    )
    benchmarker.run_benchmark(max_frames=200)