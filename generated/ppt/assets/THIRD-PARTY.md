# 第三方資產與授權

這份簡報刻意**不依賴任何 CDN**——字型與圖示全部隨檔案散佈，離線也能完整播放。
以下是隨附資產的來源與授權。

## 字型 · `fonts/`

| 檔案 | 字型 | 版權 | 授權 |
| :--- | :--- | :--- | :--- |
| `notosanstc-{200,300,400,500,700}.woff2` | Noto Sans TC | © Google LLC | [SIL Open Font License 1.1](https://openfontlicense.org/) |
| `inter-{200,300,400,500,600,700,800}.woff2` | Inter | © Rasmus Andersson | [SIL Open Font License 1.1](https://openfontlicense.org/) |
| `jetbrainsmono-{300,400,500,600}.woff2` | JetBrains Mono | © JetBrains s.r.o. | [SIL Open Font License 1.1](https://openfontlicense.org/) |

三者皆為 OFL 1.1，允許自由使用、修改與散佈（含嵌入與商業用途），
唯不得單獨販售字型本身，衍生字型不得使用保留字型名稱。

> **為什麼要自帶完整字型檔？**
> Google Fonts 的中文字型是切成 100 多個 unicode-range 分包供應的，瀏覽器只下載它判斷需要的分包。
> 實測即使已載入 105 個字面，仍會有少數字漏掉、掉回本機字型，造成**同一行標題裡字的粗細不一致**。
> 這裡刻意改用未經分包的完整檔（每個字重涵蓋 20,745 個字符），一次解決掉字與離線兩個問題。

## 函式庫 · `assets/`

| 檔案 | 專案 | 版本 | 授權 |
| :--- | :--- | :--- | :--- |
| `lucide.min.js` | [Lucide](https://lucide.dev/) | 1.28.0（已鎖版） | ISC |
| `motion.min.js` | [Motion One](https://motion.dev/) | 11.11.17 | MIT |

## 插圖 · `images/`

- `01-cover.png`、`54-learning-path.png`：改作自本專案 `docs/assets/` 的既有插圖。
- 其餘 `.svg`：本專案自製的資訊圖，構圖僅用幾何元素（依版式規範，SVG 內不放文字，
  所有標籤由投影片的 HTML 負責，以確保可選取、可翻譯、可縮放）。
