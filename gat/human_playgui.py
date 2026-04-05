import sys
import os
import argparse
import torch
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QComboBox,
)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPointF, QTimer

# 确保可从任意工作目录直接启动本脚本
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from env.human_play import HumanAIGameSession, ModelPolicyAdapter, StepResult
except ModuleNotFoundError:
    from human_play import HumanAIGameSession, ModelPolicyAdapter, StepResult

try:
    from env.game import Board
    from env.mcts_alphaZero import MCTSPlayer
except ModuleNotFoundError:
    from game import Board
    from mcts_alphaZero import MCTSPlayer


COLOR_BOARD_BG = QColor("#EEDFB2")
COLOR_GRID = QColor("#8A785D")
COLOR_BLACK = QColor("#222222")
COLOR_WHITE = QColor("#F8F8F8")
COLOR_LAST_MOVE = QColor("#E74C3C")

STYLE_SHEET = """
    QMainWindow {
        background-color: #FAFAFA;
    }
    QLabel#Title {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    QLabel#Status {
        font-size: 16px;
        color: #666666;
        font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    QComboBox {
        background-color: #FFFFFF;
        color: #2B2B2B;
        border: 1px solid #D0D0D0;
        border-radius: 6px;
        padding: 6px 28px 6px 10px;
        min-height: 32px;
        font-size: 13px;
    }
    QComboBox:disabled {
        background-color: #F3F3F3;
        color: #7A7A7A;
        border: 1px solid #DADADA;
    }
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #2B2B2B;
        selection-background-color: #EAF4EA;
        selection-color: #1E1E1E;
        border: 1px solid #D0D0D0;
        outline: none;
    }
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #45A049; }
    QPushButton:pressed { background-color: #3E8E41; }
    QPushButton:disabled { background-color: #CCCCCC; color: #888888; }
"""


def _discover_model_paths(model_dirs, extra_models):
    paths = []
    seen = set()

    for m in extra_models:
        p = os.path.abspath(m)
        if os.path.isfile(p) and p.lower().endswith(".pth") and p not in seen:
            paths.append(p)
            seen.add(p)

    for d in model_dirs:
        if not d:
            continue
        abs_dir = os.path.abspath(d)
        if not os.path.isdir(abs_dir):
            continue
        for name in sorted(os.listdir(abs_dir)):
            full = os.path.join(abs_dir, name)
            if os.path.isfile(full) and name.lower().endswith(".pth") and full not in seen:
                paths.append(full)
                seen.add(full)

    return paths


class AIAIAutoSession:
    """AI 对 AI 会话，复用同一模型驱动双方。"""

    def __init__(
        self,
        board_size: int,
        n_in_row: int,
        black_model_path: str,
        white_model_path: str,
        c_puct=5.0,
        n_playout=240,
    ):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.black_model_path = black_model_path
        self.white_model_path = white_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.board = Board(width=board_size, height=board_size, n_in_row=n_in_row)
        self.human_player_id = -1
        self.ai_player_id = 2

        self.black_policy_adapter = ModelPolicyAdapter(board_size=board_size, device=self.device)
        self.black_policy_adapter.load_checkpoint(black_model_path)
        self.white_policy_adapter = ModelPolicyAdapter(board_size=board_size, device=self.device)
        self.white_policy_adapter.load_checkpoint(white_model_path)

        self.ai_black = MCTSPlayer(
            policy_value_function=self.black_policy_adapter.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=0,
        )
        self.ai_white = MCTSPlayer(
            policy_value_function=self.white_policy_adapter.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=0,
        )
        self.players = {1: self.ai_black, 2: self.ai_white}
        self.ai_black.set_player_ind(1)
        self.ai_white.set_player_ind(2)
        self.reset(human_first=False)

    def reset(self, human_first=False):
        self.board.init_board(start_player=0)
        self.ai_black.reset_player()
        self.ai_white.reset_player()

    def is_human_turn(self):
        return False

    def apply_human_move(self, move):
        raise RuntimeError("AI 对 AI 模式不支持人类落子")

    def apply_ai_move(self, temp=1e-3):
        current_player = self.board.current_player
        player_in_turn = self.players[current_player]
        if self.board.last_move != -1:
            player_in_turn.update_with_move(self.board.last_move)

        move = player_in_turn.get_action(self.board, temp=temp, return_prob=0)
        self.board.do_move(move)
        is_end, winner = self.board.game_end()
        r, c = self.board.move_to_location(move)
        return StepResult(move=move, location=(r, c), player=current_player, is_end=is_end, winner=winner)

    def get_snapshot(self):
        grid = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for move, player in self.board.states.items():
            r, c = self.board.move_to_location(move)
            grid[r][c] = player

        is_end, winner = self.board.game_end()
        return {
            "board_size": self.board_size,
            "n_in_row": self.n_in_row,
            "current_player": self.board.current_player,
            "human_player": -1,
            "ai_player": 1,
            "last_move": self.board.last_move,
            "is_end": is_end,
            "winner": winner,
            "availables": list(self.board.availables),
            "grid": grid,
        }


class AIWorker(QThread):
    move_finished = pyqtSignal(object)

    def __init__(self, session, ai_temp: float):
        super().__init__()
        self.session = session
        self.ai_temp = ai_temp

    def run(self):
        try:
            result = self.session.apply_ai_move(temp=self.ai_temp)
            self.move_finished.emit(result)
        except Exception as e:
            self.move_finished.emit(e)


class BoardWidget(QWidget):
    human_move_requested = pyqtSignal(int, int)

    def __init__(self, board_size_preview=8):
        super().__init__()
        self.session = None
        self.preview_board_size = board_size_preview
        self.setMinimumSize(500, 500)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.margin = 40
        self.cell_size = 1.0
        self.offset_x = float(self.margin)
        self.offset_y = float(self.margin)

    def set_session(self, session):
        self.session = session
        self.update()

    def set_preview_board_size(self, size):
        self.preview_board_size = size
        if self.session is None:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QBrush(COLOR_BOARD_BG))

        if self.session is not None:
            snapshot = self.session.get_snapshot()
            board_size = snapshot["board_size"]
            grid_data = snapshot["grid"]
            last_move = snapshot["last_move"]
        else:
            board_size = self.preview_board_size
            grid_data = [[0 for _ in range(board_size)] for _ in range(board_size)]
            last_move = -1

        usable_width = self.width() - 2 * self.margin
        usable_height = self.height() - 2 * self.margin
        self.cell_size = min(usable_width, usable_height) / (board_size - 1)
        self.offset_x = (self.width() - self.cell_size * (board_size - 1)) / 2
        self.offset_y = (self.height() - self.cell_size * (board_size - 1)) / 2

        pen = QPen(COLOR_GRID)
        pen.setWidth(2)
        painter.setPen(pen)

        for i in range(board_size):
            x = self.offset_x + i * self.cell_size
            painter.drawLine(int(x), int(self.offset_y), int(x), int(self.offset_y + (board_size - 1) * self.cell_size))
            y = self.offset_y + i * self.cell_size
            painter.drawLine(int(self.offset_x), int(y), int(self.offset_x + (board_size - 1) * self.cell_size), int(y))

        piece_radius = self.cell_size * 0.42
        for r in range(board_size):
            for c in range(board_size):
                player = grid_data[r][c]
                if player == 0:
                    continue

                cx = self.offset_x + c * self.cell_size
                cy = self.offset_y + r * self.cell_size
                if player == 1:
                    painter.setBrush(QBrush(COLOR_BLACK))
                    painter.setPen(Qt.PenStyle.NoPen)
                else:
                    painter.setBrush(QBrush(COLOR_WHITE))
                    painter.setPen(QPen(QColor("#DDDDDD"), 1))
                painter.drawEllipse(QPointF(cx, cy), piece_radius, piece_radius)

                if self.session is not None:
                    current_move_idx = self.session.board.location_to_move([r, c])
                    if current_move_idx == last_move:
                        painter.setBrush(QBrush(COLOR_LAST_MOVE))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawEllipse(QPointF(cx, cy), piece_radius * 0.25, piece_radius * 0.25)

    def mousePressEvent(self, event: QMouseEvent):
        if self.session is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            c = round((event.pos().x() - self.offset_x) / self.cell_size)
            r = round((event.pos().y() - self.offset_y) / self.cell_size)
            board_size = self.session.get_snapshot()["board_size"]
            if 0 <= r < board_size and 0 <= c < board_size:
                self.human_move_requested.emit(r, c)


class GomokuWindow(QMainWindow):
    def __init__(self, model_dirs=None, candidate_models=None, default_model=None, default_size=8, n_in_row=5):
        super().__init__()
        self.setWindowTitle("AlphaZero 五子棋")
        self.setStyleSheet(STYLE_SHEET)

        self.n_in_row = n_in_row
        self.ai_temp = 1e-3
        self.is_processing = False
        self.game_running = False
        self.ai_worker = None

        self.model_dirs = [os.path.abspath(d) for d in (model_dirs or [])]
        self.model_paths = _discover_model_paths(self.model_dirs, candidate_models or [])
        if default_model:
            dm = os.path.abspath(default_model)
            if os.path.isfile(dm) and dm not in self.model_paths:
                self.model_paths.insert(0, dm)

        self.default_size = default_size
        self.session = None
        self.mode = "human_ai"
        self.human_first = True

        self.init_ui()
        self._refresh_model_combo(select_path=default_model)
        self.update_status("请先设置棋盘大小、模式和模型，然后点击开始对局")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(30, 20, 30, 30)
        layout.setSpacing(12)

        top_layout = QHBoxLayout()
        self.title_label = QLabel("AlphaZero 五子棋")
        self.title_label.setObjectName("Title")
        self.status_label = QLabel("准备就绪")
        self.status_label.setObjectName("Status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_layout.addWidget(self.title_label)
        top_layout.addWidget(self.status_label)
        layout.addLayout(top_layout)

        setup_row1 = QHBoxLayout()
        self.label_size = QLabel("1. 棋盘大小")
        self.label_size.setObjectName("Status")
        self.combo_size = QComboBox()
        for s in [8, 10, 12, 15]:
            self.combo_size.addItem(f"{s} x {s}", s)
        idx = self.combo_size.findData(self.default_size)
        self.combo_size.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_size.currentIndexChanged.connect(self.on_setup_changed)
        setup_row1.addWidget(self.label_size)
        setup_row1.addWidget(self.combo_size)
        layout.addLayout(setup_row1)

        setup_row2 = QHBoxLayout()
        self.label_mode = QLabel("2. 对战模式")
        self.label_mode.setObjectName("Status")
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("人 vs AI", "human_ai")
        self.combo_mode.addItem("AI vs AI", "ai_ai")
        self.combo_mode.currentIndexChanged.connect(self.on_setup_changed)
        setup_row2.addWidget(self.label_mode)
        setup_row2.addWidget(self.combo_mode)
        layout.addLayout(setup_row2)

        setup_row3 = QHBoxLayout()
        self.label_model = QLabel("3. 黑棋模型")
        self.label_model.setObjectName("Status")
        self.combo_model_black = QComboBox()
        self.combo_model_black.currentIndexChanged.connect(self.on_setup_changed)
        self.btn_refresh_models = QPushButton("刷新模型")
        self.btn_refresh_models.clicked.connect(self.on_refresh_models)
        setup_row3.addWidget(self.label_model)
        setup_row3.addWidget(self.combo_model_black, stretch=1)
        setup_row3.addWidget(self.btn_refresh_models)
        layout.addLayout(setup_row3)

        setup_row4 = QHBoxLayout()
        self.label_model_white = QLabel("4. 白棋模型(AI vs AI)")
        self.label_model_white.setObjectName("Status")
        self.combo_model_white = QComboBox()
        self.combo_model_white.currentIndexChanged.connect(self.on_setup_changed)
        setup_row4.addWidget(self.label_model_white)
        setup_row4.addWidget(self.combo_model_white, stretch=1)
        layout.addLayout(setup_row4)

        self.model_path_label = QLabel("")
        self.model_path_label.setObjectName("Status")
        layout.addWidget(self.model_path_label)

        action_row = QHBoxLayout()
        self.btn_start = QPushButton("开始对局")
        self.btn_start.clicked.connect(self.start_new_game)
        self.btn_restart = QPushButton("重新开始(同配置)")
        self.btn_restart.clicked.connect(self.restart_game)
        self.btn_restart.setEnabled(False)
        action_row.addWidget(self.btn_start)
        action_row.addWidget(self.btn_restart)
        layout.addLayout(action_row)

        self.board_widget = BoardWidget(board_size_preview=self.combo_size.currentData())
        self.board_widget.human_move_requested.connect(self.on_human_move)
        layout.addWidget(self.board_widget, stretch=1)
        self.resize(840, 860)
        self.on_setup_changed()

    def _refresh_model_combo(self, select_path=None):
        current_black = self.combo_model_black.currentData() if self.combo_model_black.count() else None
        current_white = self.combo_model_white.currentData() if self.combo_model_white.count() else None

        self.combo_model_black.clear()
        self.combo_model_white.clear()
        self.model_paths = _discover_model_paths(self.model_dirs, self.model_paths)
        for path in self.model_paths:
            text = f"{os.path.basename(path)}  ({os.path.dirname(path)})"
            self.combo_model_black.addItem(text, path)
            self.combo_model_white.addItem(text, path)

        if self.model_paths:
            target = os.path.abspath(select_path) if select_path else self.model_paths[0]
            idx_black = self.combo_model_black.findData(current_black or target)
            if idx_black < 0:
                idx_black = 0
            self.combo_model_black.setCurrentIndex(idx_black)

            idx_white = self.combo_model_white.findData(current_white or target)
            if idx_white < 0:
                idx_white = idx_black
            self.combo_model_white.setCurrentIndex(idx_white)

            self._update_model_path_label()
            self.btn_start.setEnabled(True)
        else:
            self.model_path_label.setText("未找到模型，请放入 models/gui_models 后点击 刷新模型")
            self.btn_start.setEnabled(False)

    def set_setup_enabled(self, enabled: bool):
        self.combo_size.setEnabled(enabled)
        self.combo_mode.setEnabled(enabled)
        self.combo_model_black.setEnabled(enabled)
        self.combo_model_white.setEnabled(enabled)
        self.btn_refresh_models.setEnabled(enabled)
        self.btn_start.setEnabled(enabled and self.combo_model_black.count() > 0)

    def on_refresh_models(self):
        if self.game_running:
            return
        current = self.combo_model_black.currentData() if self.combo_model_black.count() else None
        self._refresh_model_combo(select_path=current)

    def on_setup_changed(self):
        if self.game_running:
            return
        size = self.combo_size.currentData()
        self.board_widget.set_preview_board_size(size)
        is_ai_ai = self.combo_mode.currentData() == "ai_ai"
        self.label_model_white.setVisible(is_ai_ai)
        self.combo_model_white.setVisible(is_ai_ai)
        self._update_model_path_label()

    def _update_model_path_label(self):
        if self.combo_model_black.count() == 0:
            self.model_path_label.setText("未找到模型，请放入 models/gui_models 后点击 刷新模型")
            return
        black_model = self.combo_model_black.currentData()
        if self.combo_mode.currentData() == "ai_ai":
            white_model = self.combo_model_white.currentData()
            self.model_path_label.setText(f"黑棋模型: {black_model} | 白棋模型: {white_model}")
        else:
            self.model_path_label.setText(f"当前选择: {black_model}")

    def _build_session_from_setup(self):
        board_size = int(self.combo_size.currentData())
        mode = self.combo_mode.currentData()
        black_model_path = self.combo_model_black.currentData()
        white_model_path = self.combo_model_white.currentData()
        if not black_model_path:
            raise RuntimeError("请先选择黑棋模型")

        if mode == "human_ai":
            session = HumanAIGameSession(
                board_size=board_size,
                n_in_row=self.n_in_row,
                model_path=black_model_path,
                c_puct=5.0,
                n_playout=400,
                human_first=True,
            )
            human_first = True
        else:
            if not white_model_path:
                raise RuntimeError("AI vs AI 模式下请选择白棋模型")
            session = AIAIAutoSession(
                board_size=board_size,
                n_in_row=self.n_in_row,
                black_model_path=black_model_path,
                white_model_path=white_model_path,
                c_puct=5.0,
                n_playout=240,
            )
            human_first = False
        return session, mode, human_first

    def start_new_game(self):
        if self.is_processing:
            return
        try:
            self.session, self.mode, self.human_first = self._build_session_from_setup()
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))
            return

        self.board_widget.set_session(self.session)
        self.game_running = True
        self.set_setup_enabled(False)
        self.btn_restart.setEnabled(True)
        self.update_status("对局进行中")
        self.board_widget.update()

        if not self.session.is_human_turn():
            self.trigger_ai_turn()

    def restart_game(self):
        if self.session is None or self.is_processing:
            return
        self.session.reset(human_first=self.human_first)
        self.game_running = True
        self.set_setup_enabled(False)
        self.update_status("对局重新开始")
        self.board_widget.update()
        if not self.session.is_human_turn():
            self.trigger_ai_turn()

    def on_human_move(self, r: int, c: int):
        if not self.game_running or self.is_processing or self.session is None:
            return
        if not self.session.is_human_turn():
            return

        move = self.session.board.location_to_move([r, c])
        if move not in self.session.board.availables:
            return
        try:
            result = self.session.apply_human_move(move)
            self.board_widget.update()
            if result.is_end:
                self.handle_game_end(result.winner)
                return
            self.trigger_ai_turn()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"落子异常: {str(e)}")

    def trigger_ai_turn(self):
        if self.session is None or self.is_processing:
            return
        self.is_processing = True
        self.btn_restart.setEnabled(False)
        self.update_status("AI 思考中...")

        self.ai_worker = AIWorker(self.session, self.ai_temp)
        self.ai_worker.move_finished.connect(self.on_ai_finished)
        self.ai_worker.start()

    def on_ai_finished(self, result):
        self.is_processing = False
        self.btn_restart.setEnabled(True)
        self.board_widget.update()

        if isinstance(result, Exception):
            QMessageBox.critical(self, "AI 错误", str(result))
            self.update_status("AI 出错，已停止")
            self.game_running = False
            self.set_setup_enabled(True)
            return

        if result.is_end:
            self.handle_game_end(result.winner)
            return

        if self.mode == "ai_ai" and self.game_running:
            self.update_status("AI vs AI 对局中...")
            QTimer.singleShot(60, self.trigger_ai_turn)
        else:
            self.update_status("你的回合")

    def handle_game_end(self, winner: int):
        self.board_widget.update()
        self.game_running = False
        self.set_setup_enabled(True)

        if winner == -1:
            msg = "游戏结束：平局！"
            color = "#7F8C8D"
        elif self.mode == "human_ai":
            if winner == self.session.human_player_id:
                msg = "游戏结束：你赢了！"
                color = "#27AE60"
            else:
                msg = "游戏结束：AI 获胜！"
                color = "#C0392B"
        else:
            msg = f"游戏结束：AI 玩家 {winner} 获胜！"
            color = "#C0392B"

        self.status_label.setText(msg)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        QMessageBox.information(self, "对局结束", msg)

    def update_status(self, text=None):
        if text is not None:
            self.status_label.setText(text)
            self.status_label.setStyleSheet("color: #666666;")
            return

        self.status_label.setText("准备就绪")
        self.status_label.setStyleSheet("color: #666666;")

    def closeEvent(self, event):
        if self.ai_worker is not None and self.ai_worker.isRunning():
            self.ai_worker.quit()
            self.ai_worker.wait(1000)
        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(description="Gomoku PyQt GUI")
    parser.add_argument("--model", type=str, default=None, help="默认选中的模型")
    parser.add_argument("--models", nargs="*", default=[], help="额外模型文件列表（可传多个 .pth）")
    parser.add_argument("--model-dir", action="append", default=[], help="模型目录（可重复传多次）")
    parser.add_argument("--size", type=int, default=8, help="默认棋盘边长（仅用于初始下拉选中）")
    parser.add_argument("--n-in-row", type=int, default=5, help="连珠数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    default_gui_model_dir = os.path.join(ROOT_DIR, "models", "gui_models")
    os.makedirs(default_gui_model_dir, exist_ok=True)

    model_dirs = [os.path.join(ROOT_DIR, "models"), default_gui_model_dir]
    model_dirs.extend(args.model_dir)

    app = QApplication(sys.argv)
    window = GomokuWindow(
        model_dirs=model_dirs,
        candidate_models=args.models,
        default_model=args.model,
        default_size=args.size,
        n_in_row=args.n_in_row,
    )
    window.show()
    sys.exit(app.exec())