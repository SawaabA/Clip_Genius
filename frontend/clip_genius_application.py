import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QComboBox,
    QVBoxLayout, QWidget, QProgressBar, QHBoxLayout, QSlider
)
from PyQt5.QtGui import QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal


class VideoProcessingWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, video_path, highlight_length, transcript_lang, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.highlight_length = highlight_length
        self.transcript_lang = transcript_lang

    def run(self):
        # Run the external processing script (blocking call in a separate thread)
        subprocess.run([
            "python", "process_video.py",
            self.video_path, self.highlight_length, self.transcript_lang
        ])
        self.finished.emit()


class ClipGeniusApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clip Genius")
        self.setWindowIcon(QIcon("hero-img.png"))  # Use your custom icon file
        self.setFixedSize(1150, 500)  # Fixed window size

        # Main layout with left (controls) and right (video player) sections
        main_layout = QHBoxLayout()

        # Left widget: Controls
        left_widget = QWidget()
        left_widget.setFixedWidth(300)
        left_layout = QVBoxLayout(left_widget)

        self.upload_label = QLabel("Upload a Video File:")
        self.upload_button = QPushButton("Select Video")
        self.upload_button.clicked.connect(self.select_video)

        self.length_label = QLabel("Select Highlight Length:")
        self.length_dropdown = QComboBox()
        self.length_dropdown.addItems(["1 Minute", "5 Minutes", "10 Minutes"])

        self.transcript_label = QLabel("Include Transcript?")
        self.transcript_dropdown = QComboBox()
        self.transcript_dropdown.addItems(["No Transcript", "English", "French", "Spanish"])

        self.generate_button = QPushButton("Generate Highlight")
        self.generate_button.clicked.connect(self.generate_highlight)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        left_layout.addWidget(self.upload_label)
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.length_label)
        left_layout.addWidget(self.length_dropdown)
        left_layout.addWidget(self.transcript_label)
        left_layout.addWidget(self.transcript_dropdown)
        left_layout.addWidget(self.generate_button)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch()

        # Right widget: Video Player area
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setFixedWidth(680)

        # Video widget with a black background
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        self.video_widget.setFixedSize(680, 380)

        # Video control buttons and seek slider in a horizontal layout
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(100)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self.seek_video)

        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.seek_slider)

        right_layout.addWidget(QLabel("Preview Video:"))
        right_layout.addWidget(self.video_widget)
        right_layout.addWidget(controls_widget)
        right_layout.addStretch()

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.video_path = ""
        self.highlight_video_path = "game_highlight.mp4"  # Expected output path

        # Timer for progress bar simulation
        self.loading_timer = QTimer()
        self.loading_timer.setInterval(500)  # Update every 500 ms
        self.loading_timer.timeout.connect(self.update_progress)
        self.progress_value = 0

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Videos (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            self.upload_label.setText(f"Selected: {os.path.basename(file_path)}")
            # Immediately load the selected video into the player
            self.load_video(self.video_path)

    def generate_highlight(self):
        if not self.video_path:
            self.upload_label.setText("No file selected!")
            return

        highlight_length = self.length_dropdown.currentText().split(" ")[0]  # minutes as string
        transcript_lang = self.transcript_dropdown.currentText()

        self.progress_value = 0
        self.progress_bar.setValue(self.progress_value)
        self.loading_timer.start()

        self.worker = VideoProcessingWorker(self.video_path, highlight_length, transcript_lang)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def update_progress(self):
        if self.progress_value < 90:
            self.progress_value += 10
            self.progress_bar.setValue(self.progress_value)

    def processing_finished(self):
        self.loading_timer.stop()
        self.progress_bar.setValue(100)
        self.upload_label.setText("Processing Complete! Video is Ready.")
        if os.path.exists(self.highlight_video_path):
            self.load_video(self.highlight_video_path)

    def load_video(self, video_path):
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.media_player.play()

    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def seek_video(self, position):
        duration = self.media_player.duration()
        self.media_player.setPosition(int(duration * (position / 100)))

    def update_slider_position(self, position):
        duration = self.media_player.duration()
        if duration > 0:
            self.seek_slider.setValue(int((position / duration) * 100))

    def update_slider_range(self, duration):
        self.seek_slider.setRange(0, 100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClipGeniusApp()
    window.show()
    sys.exit(app.exec_())

# python clip_genius_application.py
