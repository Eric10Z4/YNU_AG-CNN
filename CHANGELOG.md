# 更新日志

## 2026-04-05
- 将对局模块目录从 `env` 重命名迁移为 `gat`。
- 新增 GUI 开局配置流程：先选棋盘大小、再选对战模式、最后选模型。
- 支持 AI vs AI 使用不同模型（黑棋模型与白棋模型可分别选择）。
- 优化 GUI 模型管理：支持自动扫描、刷新与下拉选择。
- 新增旧模型转换脚本：`gat/tools/convert_legacy_model_to_pth.py`。
- 同步 4 个 GUI 测试模型到 `models/gui_models`。
