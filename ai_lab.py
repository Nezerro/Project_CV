"""Вкладка «ИИ-лаборатория» Campus Watch: поиск пути, minimax и k-means на данных проекта."""
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSpinBox, QVBoxLayout, QWidget)

from ai_algos import (ALGORITHMS, CAMPUS_MAP, PursuitGame, cluster_events, demo_events, elbow, find_cell)

BG, PANEL, PANEL2, LINE = "#0b1016", "#111923", "#172330", "#263545"
TEXT, MUTED, CYAN, GREEN, AMBER, RED = "#edf4fa", "#91a3b5", "#68d5e7", "#69d7a6", "#f5bd66", "#ff7585"


def _button(text, primary=False):
    btn = QPushButton(text)
    btn.setStyleSheet(f"QPushButton{{background:{CYAN if primary else PANEL2};color:{BG if primary else TEXT};"
                      f"border:1px solid {LINE};border-radius:8px;padding:8px 12px;font-weight:700}}"
                      f"QPushButton:hover{{border-color:{CYAN}}} QPushButton:disabled{{color:{MUTED}}}")
    return btn


class GridView(QWidget):
    """Рисует карту кампуса, обход алгоритма, маршрут и фигуры игры."""
    cell_clicked = Signal(int, int)

    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.rows, self.cols = len(grid), len(grid[0])
        self.visited, self.shown = [], 0          # shown - сколько вершин обхода уже показано
        self.path = []
        self.post = find_cell(grid, "G")
        self.camera = find_cell(grid, "C")
        self.incident = None
        self.guard = self.intruder = None
        self.setMinimumSize(440, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _geometry(self):
        size = min(self.width() / self.cols, self.height() / self.rows)
        return size, (self.width() - size * self.cols) / 2, (self.height() - size * self.rows) / 2

    def _rect(self, cell, size, ox, oy, pad=0.0):
        return QRectF(ox + cell[0] * size + pad, oy + cell[1] * size + pad, size - 2 * pad, size - 2 * pad)

    def mousePressEvent(self, event):
        size, ox, oy = self._geometry()
        pos = event.position()
        x, y = int((pos.x() - ox) // size), int((pos.y() - oy) // size)
        if 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] != "#":
            self.cell_clicked.emit(x, y)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        size, ox, oy = self._geometry()
        for y, row in enumerate(self.grid):
            for x, ch in enumerate(row):
                p.fillRect(self._rect((x, y), size, ox, oy), QColor("#243445" if ch == "#" else "#0f1822"))
        for cell in self.visited[:self.shown]:
            p.fillRect(self._rect(cell, size, ox, oy, 1), QColor(104, 213, 231, 55))
        if self.path:
            pen = QPen(QColor(GREEN), max(2.0, size * 0.22))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            for a, b in zip(self.path, self.path[1:]):
                pa, pb = self._rect(a, size, ox, oy).center(), self._rect(b, size, ox, oy).center()
                p.drawLine(QPointF(pa), QPointF(pb))
        p.setPen(Qt.NoPen)
        marks = [(self.post, "#68d5e7", "П"), (self.camera, "#f5bd66", "К")]
        for cell, color, letter in marks:
            p.setBrush(QColor(color))
            p.drawRoundedRect(self._rect(cell, size, ox, oy, size * .12), 4, 4)
            p.setPen(QColor(BG))
            p.setFont(QFont("Sans", max(7, int(size * .38)), QFont.Bold))
            p.drawText(self._rect(cell, size, ox, oy), Qt.AlignCenter, letter)
            p.setPen(Qt.NoPen)
        if self.incident:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(RED), 2))
            p.drawEllipse(self._rect(self.incident, size, ox, oy, size * .1))
            p.setPen(Qt.NoPen)
        for cell, color in ((self.intruder, RED), (self.guard, CYAN)):
            if cell:
                p.setBrush(QColor(color))
                p.drawEllipse(self._rect(cell, size, ox, oy, size * .2))
        p.end()


class AiLabTab(QWidget):
    def __init__(self, events_provider):
        super().__init__()
        self.events_provider = events_provider
        self.grid = CAMPUS_MAP
        self.view = GridView(self.grid)
        self.view.cell_clicked.connect(self.on_cell)
        self.game = PursuitGame(self.grid)
        self.game_over = False
        self.reveal = QTimer(self)
        self.reveal.timeout.connect(self._reveal_step)
        self.auto = QTimer(self)
        self.auto.timeout.connect(self.game_step)
        self.setStyleSheet(f"QCheckBox, QSpinBox, QComboBox {{ color:{TEXT}; }}"
                           f"QSpinBox, QComboBox {{ background:{BG}; border:1px solid {LINE}; border-radius:6px; padding:5px; }}")
        self._build()
        self.new_game()

    # -- интерфейс ----------------------------------------------------------- #
    def _card(self, title, hint):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        head = QLabel(title)
        head.setStyleSheet("font-weight:900;font-size:12px;letter-spacing:1px")
        note = QLabel(hint)
        note.setWordWrap(True)
        note.setStyleSheet(f"font-size:10px;color:{MUTED}")
        lay.addWidget(head)
        lay.addWidget(note)
        return card, lay

    def _out(self):
        label = QLabel("")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet(f"font-size:11px;background:{PANEL2};padding:8px;border-radius:7px")
        return label

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 12, 0, 0)
        root.setSpacing(16)
        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        legend = QLabel("П - пост охраны · К - камера · красный круг - инцидент · клик по карте задаёт место инцидента")
        legend.setStyleSheet(f"font-size:10px;color:{MUTED}")
        left.addWidget(legend)
        root.addLayout(left, 6)

        right = QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(right, 4)

        card, lay = self._card("МАРШРУТ ОХРАННИКА · BFS / DFS / A*",
                               "Кратчайший путь от поста до места инцидента. При каждом событии камеры маршрут строится сам (A*).")
        row = QHBoxLayout()
        self.algo = QComboBox()
        self.algo.addItems(list(ALGORITHMS))
        self.algo.setCurrentText("A*")
        go = _button("Построить", True)
        go.clicked.connect(lambda: self.find_route(self.algo.currentText()))
        cmp_btn = _button("Сравнить все")
        cmp_btn.clicked.connect(self.compare_algorithms)
        row.addWidget(self.algo)
        row.addWidget(go)
        row.addWidget(cmp_btn)
        lay.addLayout(row)
        self.route_out = self._out()
        lay.addWidget(self.route_out)
        right.addWidget(card)

        card, lay = self._card("ПАТРУЛЬ ПРОТИВ НАРУШИТЕЛЯ · MINIMAX",
                               "Охранник ходит на 2 клетки, нарушитель на 1. Каждый ход - перебор дерева игры на заданную глубину.")
        row = QHBoxLayout()
        row.addWidget(QLabel("Глубина"))
        self.depth = QSpinBox()
        self.depth.setRange(1, 5)
        self.depth.setValue(3)
        self.ab = QCheckBox("alpha-beta")
        self.ab.setChecked(True)
        row.addWidget(self.depth)
        row.addWidget(self.ab)
        row.addStretch()
        lay.addLayout(row)
        row = QHBoxLayout()
        new = _button("Новая игра")
        new.clicked.connect(self.new_game)
        step = _button("Ход", True)
        step.clicked.connect(self.game_step)
        self.auto_btn = _button("Авто")
        self.auto_btn.clicked.connect(self.toggle_auto)
        for w in (new, step, self.auto_btn):
            row.addWidget(w)
        lay.addLayout(row)
        self.game_out = self._out()
        lay.addWidget(self.game_out)
        right.addWidget(card)

        card, lay = self._card("K-MEANS · КЛАСТЕРЫ СОБЫТИЙ",
                               "Группирует журнал по времени суток (по кругу) и типу события: когда и что случается чаще.")
        row = QHBoxLayout()
        row.addWidget(QLabel("k"))
        self.k = QSpinBox()
        self.k.setRange(2, 6)
        self.k.setValue(3)
        self.demo = QCheckBox("тестовые данные")
        run = _button("Кластеризовать", True)
        run.clicked.connect(self.run_kmeans)
        row.addWidget(self.k)
        row.addWidget(self.demo)
        row.addStretch()
        row.addWidget(run)
        lay.addLayout(row)
        self.km_out = self._out()
        self.km_out.setText("Нужно хотя бы несколько событий в журнале, либо включите тестовые данные.")
        lay.addWidget(self.km_out)
        right.addWidget(card)
        right.addStretch()

    # -- маршруты ------------------------------------------------------------ #
    def on_cell(self, x, y):
        self.view.incident = (x, y)
        self.find_route(self.algo.currentText())

    def on_alert(self):
        """Вызывается из главного окна при каждом событии: маршрут к камере, где оно произошло."""
        self.view.incident = self.view.camera
        self.find_route("A*")

    def _run(self, name):
        goal = self.view.incident
        return ALGORITHMS[name](self.grid, self.view.post, goal)

    def find_route(self, name):
        if self.view.incident is None:
            self.view.incident = self.view.camera
        self.stop_game()
        path, order = self._run(name)
        self.view.guard = self.view.intruder = None
        self.view.path, self.view.visited, self.view.shown = path or [], order, 0
        if path:
            self.route_out.setText(f"{name}: маршрут {len(path) - 1} шагов · просмотрено клеток: {len(order)}")
        else:
            self.route_out.setText(f"{name}: маршрута нет")
        self.reveal.start(18)

    def _reveal_step(self):
        self.view.shown = min(self.view.shown + 2, len(self.view.visited))
        self.view.update()
        if self.view.shown >= len(self.view.visited):
            self.reveal.stop()

    def compare_algorithms(self):
        if self.view.incident is None:
            self.view.incident = self.view.camera
        self.find_route(self.algo.currentText())          # рисуем выбранный алгоритм
        lines = []
        for name in ALGORITHMS:
            path, order = self._run(name)
            lines.append(f"{name:<4} шагов: {len(path) - 1 if path else '-':>3} · просмотрено: {len(order)}")
        self.route_out.setText("\n".join(lines) + "\nBFS и A* дают кратчайший путь; A* смотрит меньше клеток; DFS путь находит, но длиннее.")

    # -- игра ---------------------------------------------------------------- #
    def stop_game(self):
        self.auto.stop()
        self.auto_btn.setText("Авто")

    def new_game(self):
        self.stop_game()
        self.reveal.stop()
        self.game_over = False
        self.view.path, self.view.visited, self.view.shown = [], [], 0
        self.view.guard = self.view.post
        self.view.intruder = (len(self.grid[0]) - 2, len(self.grid) - 2)
        self.game_out.setText("Нажмите «Ход» или «Авто».")
        self.view.update()

    def toggle_auto(self):
        if self.auto.isActive():
            self.stop_game()
        else:
            if self.game_over:
                self.new_game()
            self.auto.start(450)
            self.auto_btn.setText("Стоп")

    def game_step(self):
        if self.view.guard is None or self.view.intruder is None:
            self.new_game()
        if self.game_over:
            return
        depth, use_ab = self.depth.value(), self.ab.isChecked()
        g, i = self.view.guard, self.view.intruder
        _, _, nodes_plain = self.game.best_move(g, i, True, depth, alpha_beta=False)
        move, _, nodes_ab = self.game.best_move(g, i, True, depth, alpha_beta=True)
        nodes = nodes_ab if use_ab else nodes_plain
        self.view.guard = move
        if move == i:
            self._finish("Нарушитель пойман")
        else:
            reply, _, _ = self.game.best_move(move, i, False, depth, use_ab)
            self.view.intruder = reply
            if reply == move:
                self._finish("Нарушитель пойман")
        dist = self.game.dist(self.view.guard, self.view.intruder)
        text = (f"Расстояние: {dist} · глубина {depth}\n"
                f"minimax: {nodes_plain} узлов · alpha-beta: {nodes_ab} узлов "
                f"(экономия {100 - 100 * nodes_ab / max(nodes_plain, 1):.0f}%) · оценка ходов одинакова")
        if self.game_over:
            text = "Нарушитель пойман.\n" + text
        self.game_out.setText(text)
        self.view.update()

    def _finish(self, _text):
        self.game_over = True
        self.stop_game()

    # -- k-means ------------------------------------------------------------- #
    def run_kmeans(self):
        events = demo_events() if self.demo.isChecked() else list(self.events_provider())
        k = self.k.value()
        result = cluster_events(events, k)
        if result is None:
            self.km_out.setText(f"Мало данных: событий {len(events)}, а нужно не меньше {max(k, 2)}. "
                                "Включите тестовые данные или накопите журнал.")
            return
        lines = [f"Событий: {len(events)} · k = {k}"]
        for n, c in enumerate(result["clusters"], 1):
            hour, minute = int(c["hour"]), int((c["hour"] % 1) * 60)
            lines.append(f"{n}) {c['count']} шт · около {hour:02d}:{minute:02d} · чаще всего: {c['kind']}")
        curve = elbow(events)
        lines.append("Метод локтя (инерция): " + " · ".join(f"k={kk}: {val:.1f}" for kk, val in curve))
        self.km_out.setText("\n".join(lines))
