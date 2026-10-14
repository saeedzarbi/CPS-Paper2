# Paper 2 naming & versioning (OACS)

مقالهٔ اول با **شمارهٔ نسخهٔ محیط** (`v14` → `v18`) و مسیرهایی مثل `results/v18_certified` شناخته می‌شود. مقالهٔ دوم عمداً از این الگو جدا است تا روی سرور و در git اشتباه گرفته نشود.

## خلاصه

| مفهوم | مقاله ۱ (CPS) | مقاله ۲ (OACS) |
|--------|----------------|----------------|
| نام محصول | Certified Perceptual Shield | **OACS** (Online ACI Conformal Shield) |
| تگ ریلیز | `v18` / `master_v18_5g` | **`oacs-r1`** (بعداً `oacs-r2`, …) |
| ورودی محیط در کد جدید | `abr_multi_env_v18.ABREnv` | **`paper2_abr_env.Paper2StreamingEnv`** |
| نتایج ارزیابی | `results/v18_certified/` | **`results/oacs/campaigns/<campaign>/`** |
| چک‌پوینت RL | `results/models/.../v18...` | **`results/models/oacs/<model>/seed_*/`** |
| جداول LaTeX | `macros_v18`, `table_*_v18` | **`paper/oacs/`** (فاز ۵) |

**ثابت می‌ماند (عمداً):** محتوای trace در `data/standardized/*_v18` — همان split اثبات‌شدهٔ Paper 1؛ فقط نام پوشهٔ داده را عوض نمی‌کنیم تا reproducibility نشکند.

## Arms در ارزیابی Paper 2

| نام جدید (استفاده در اسکریپت‌های OACS) | معنی | نام legacy در roadmap اولیه |
|----------------------------------------|------|-----------------------------|
| `raw` | بدون شیلد | `raw` |
| `static_cps` | Paper 1 shield | `cps_v18` |
| `aci_feasibility` | ACI بدون banking | `aci_safety` |
| `aci_oacs` | ACI کامل + per-chunk VMAF + mask | `aci_certified` |

ثابت‌ها: `configs/paper2_release.py`.

## فایل‌های ارزیابی (نقشهٔ roadmap → نام نهایی)

| roadmap (`help.md`) | نام پیشنهادی Paper 2 |
|---------------------|----------------------|
| `eval_aci_shield_v20.py` | `src/evaluation/eval_oacs_campaign.py` |
| `stall_cause_breakdown_v20.py` | `src/evaluation/stall_attribution_oacs.py` |
| `coverage_under_drift_v20.py` | `src/evaluation/coverage_drift_oacs.py` |
| `run_v20_eval.sh` | `experiments/paper2/run_campaign.sh` |
| `train_maskable_v20.py` | `src/training/train_maskable_oacs.py` |
| `make_paper2_assets.py` | `src/paper/make_oacs_assets.py` |

تا زمان پیاده‌سازی، منطق همان roadmap است؛ فقط **مسیر و پیشوند فایل** با OACS یکدست می‌شود.

## Config مرکزی

```python
from configs.paper2_release import (
    PAPER2_CODENAME,
    PAPER2_RELEASE,
    campaign_dir,
    ARM_ACI_FULL,
)
```
