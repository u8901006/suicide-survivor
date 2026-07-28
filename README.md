# Suicide Survivor · 自殺喪親文獻日報

每日自動從 PubMed 抓取最新的自殺喪親 (Suicide Bereavement) / 自殺損失者 (Survivors of Suicide Loss) 相關研究文獻，由 NVIDIA NIM Nemotron 3 分析彙整後生成繁體中文日報。

## 架構

- **GitHub Actions**: 每天台北時間 23:00 自動執行
- **PubMed E-utilities**: 搜尋 22 本期刊，7 天內新文獻
- **NVIDIA NIM Nemotron 3**: 主要模型 `nvidia/nemotron-3-super-120b-a12b`，備用模型 `nvidia/nemotron-3-nano-30b-a3b`；用於繁體中文摘要、PICO 分析、臨床實用性評估
- **GitHub Pages**: 靜態 HTML 部署

## 網站

https://u8901006.github.io/suicide-survivor/
