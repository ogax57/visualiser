import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import threading
import time

# ----------- CONFIG -------------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
COLORS = [
    "#ff4d4d",
    "#ff884d",
    "#ffcc4d",
    "#e6ff4d",
    "#b3ff4d",
    "#4dff88",
    "#4dffc7",
    "#4dd5ff",
    "#4d8cff",
    "#7c4dff",
    "#c44dff",
    "#ff4dd8",
]
SENSITIVITY = 0.85
BUFFER_SIZE = 2048
SAMPLE_RATE = 44100
DOTS = {note: [] for note in NOTES}


# ----------- HELPER FUNCTIONS -------------
def freq_to_note(freq):
    midi = 69 + 12 * np.log2(freq / 440)
    midi_round = int(round(midi))
    name = NOTES[(midi_round + 120) % 12]
    cents = int((midi - midi_round) * 100)
    return name, cents


def autocorrelate(buf):
    # Simple autocorrelation
    buf = buf - np.mean(buf)
    corr = np.correlate(buf, buf, mode="full")
    corr = corr[len(corr) // 2 :]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return -1
    return SAMPLE_RATE / peak


# ----------- VISUAL SETUP -------------
fig, ax = plt.subplots()
plt.axis("off")
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
cells = {}
for i, note in enumerate(NOTES):
    x = i % 4
    y = 2 - (i // 4)
    rect = plt.Rectangle((x, y), 1, 1, fill=False, edgecolor="white", linewidth=1.5)
    ax.add_patch(rect)
    cells[note] = {"x": x + 0.5, "y": y + 0.5, "dots": []}

plt.ion()
plt.show()


# ----------- AUDIO CALLBACK -------------
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    buf = indata[:, 0]
    freq = autocorrelate(buf)
    if freq > 30 and freq < 2000:
        note, cents = freq_to_note(freq)
        accept = 40 * SENSITIVITY
        if abs(cents) <= accept:
            # spawn or grow dot
            x = cells[note]["x"]
            y = cells[note]["y"]
            dot = Circle(
                (x, y),
                radius=0.1 + np.random.rand() * 0.15,
                color=COLORS[NOTES.index(note)],
                alpha=0.8,
            )
            ax.add_patch(dot)
            cells[note]["dots"].append(dot)


# ----------- VISUAL UPDATE FUNCTION (Main Thread Safe) -------------
def update_vis():
    # Shrink existing dots and prepare the next cycle
    for note in NOTES:
        # Create a new list for dots that are still visible
        new_dots = []
        for dot in cells[note]["dots"]:
            dot.set_radius(dot.get_radius() * 0.97)  # Use set_radius (safer)
            dot.set_alpha(dot.get_alpha() * 0.97)  # Use set_alpha
            if dot.get_alpha() > 0.05:
                new_dots.append(dot)
            else:
                dot.remove()  # Remove very faded dots from the axes
        cells[note]["dots"] = new_dots

    # Request a redraw. draw_idle() is thread-safe for scheduling the update.
    fig.canvas.draw_idle()

    # Restart the timer for the next iteration
    timer.start()


# ----------- START AUDIO STREAM & MAIN LOOP -------------

# 1. Create a Matplotlib Timer to handle the animation in the main thread
timer = fig.canvas.new_timer(interval=50)  # 50ms interval (approx 20 FPS)
timer.add_callback(update_vis)

# 2. Start the sound device stream
with sd.InputStream(
    channels=1, callback=audio_callback, blocksize=BUFFER_SIZE, samplerate=SAMPLE_RATE
):
    print("Listening... Play notes!")

    # 3. Start the animation timer and the Matplotlib GUI loop
    timer.start()
    plt.show()
