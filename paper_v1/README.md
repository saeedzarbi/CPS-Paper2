# کد اجرای نهایی (v18 / v19)

این پوشه فقط فایل‌هایی است که کمپین نهایی را اجرا کردند و نتایج مقاله از آن‌ها آمده است (`results/v18_certified`، `ablation_eps_alpha`، `perchunk_ablation`). نسخه‌های قدیمی‌تر محیط و شیلد (v9 تا v17 به‌جز زنجیرهٔ وراثت v18) اینجا نیستند.

اجرای واقعی همچنان از ریشهٔ `new/` است، چون داده و نتایج آنجا هستند. این پوشه رونوشت همان فایل‌هاست تا کد نهایی از بقیه جدا باشد.

## ورودی‌ها

- `src/training/run_certified_v18.sh` — آموزش PPO و ارزیابی سه‌بازو
- `run_v19_finalize.sh` و `run_v19_remaining_eval.sh` — تکمیل ارزیابی و ساخت دارایی‌های مقاله
- `run_v19_full_server.sh` — خط لولهٔ سرور (دانلود، انکد، VMAF، تریس، ارزیابی)

## هسته

- سپر: `src/training/certified_perceptual_shield.py`
- ارزیابی: `src/evaluation/eval_certified_shield_v18.py`
- محیط: `src/environment/abr_multi_env_v18.py` (وارث v15 ← v14 ← v12 ← … ← v5)
- تریس: `data/prepare_traces_v18.py`
- مقاله: `src/paper/make_cps_paper_assets.py` و `make_cps_macros.py`
