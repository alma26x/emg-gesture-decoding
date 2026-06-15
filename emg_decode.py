import os, glob, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\arial.ttf")
    plt.rcParams["font.family"] = "Arial"
except Exception:
    pass

ROOT = r"C:\Users\afric\emg_decoding\data\EMG_data_for_gestures-master"
OUT  = r"C:\Users\afric\emg_decoding"
INK, TEAL = "#1d2a30", "#0d7a6f"

CLASSES = [1, 2, 3, 4, 5, 6]
NAMES = {1: "Rest", 2: "Fist", 3: "Wrist\nflexion", 4: "Wrist\nextension",
         5: "Radial\ndev.", 6: "Ulnar\ndev."}
W, S = 40, 20

def features(win):
    f = []
    for c in range(win.shape[1]):
        x = win[:, c]
        d = np.diff(x)
        f += [np.mean(np.abs(x)),
              np.sqrt(np.mean(x ** 2)),
              np.sum(np.abs(d)),
              int(np.sum(x[:-1] * x[1:] < 0)),
              int(np.sum(d[:-1] * d[1:] < 0))]
    return f

def windows(path):
    try:
        data = np.loadtxt(path, skiprows=1)
    except Exception:
        return [], []
    if data.ndim != 2 or data.shape[1] < 10:
        return [], []
    ch, cls = data[:, 1:9], data[:, 9].astype(int)
    X, y, i, n = [], [], 0, len(cls)
    while i < n:
        c, j = cls[i], i
        while j < n and cls[j] == c:
            j += 1
        if c in CLASSES and (j - i) >= W:
            seg = ch[i:j]
            for s in range(0, len(seg) - W + 1, S):
                X.append(features(seg[s:s + W])); y.append(c)
        i = j
    return X, y

subjects = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
accs, all_true, all_pred = [], [], []
for sub in subjects:
    files = sorted(glob.glob(os.path.join(ROOT, sub, "*.txt")))
    if len(files) < 2:
        continue
    Xtr, ytr = windows(files[0])
    Xte, yte = windows(files[1])
    if len(set(ytr)) < 2 or not Xte:
        continue
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(np.array(Xtr), np.array(ytr))
    pred = clf.predict(np.array(Xte))
    a = accuracy_score(yte, pred)
    accs.append(a); all_true += list(yte); all_pred += list(pred)
    print(f"subject {sub}: {a*100:5.1f}%   (train {len(Xtr):4d} / test {len(Xte):4d} windows)")

accs = np.array(accs)
print("\n=== Results ===")
print(f"Subjects evaluated:           {len(accs)}")
print(f"Mean within-subject accuracy: {accs.mean()*100:.1f}%  +/- {accs.std()*100:.1f}%")
print(f"Best / worst subject:         {accs.max()*100:.1f}% / {accs.min()*100:.1f}%")
print(f"Chance level (6 classes):     {100/6:.1f}%")

cm = confusion_matrix(all_true, all_pred, labels=CLASSES).astype(float)
cmn = cm / cm.sum(axis=1, keepdims=True)
labels = [NAMES[c] for c in CLASSES]
fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=160)
ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(labels, fontsize=10); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted gesture", fontsize=12, color=INK)
ax.set_ylabel("True gesture", fontsize=12, color=INK)
ax.set_title(f"Surface-EMG gesture decoding: confusion matrix\n(within-subject, cross-session; {len(accs)} subjects)",
             fontsize=12, color=INK)
for i in range(6):
    for j in range(6):
        v = cmn[i, j]
        ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                color="white" if v > 0.5 else INK, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "emg_confusion_matrix.png"), facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=160)
ax.hist(accs * 100, bins=12, color=TEAL, edgecolor="white")
ax.axvline(accs.mean() * 100, color=INK, lw=1.5, ls="--")
ax.text(accs.mean() * 100, ax.get_ylim()[1] * 0.92, f"  mean {accs.mean()*100:.1f}%",
        color=INK, fontsize=10, va="top")
ax.axvline(100 / 6, color="#b0b0b0", lw=1.2, ls=":")
ax.text(100 / 6, ax.get_ylim()[1] * 0.92, "  chance", color="#888", fontsize=9, va="top")
ax.set_xlabel("Per-subject test accuracy (%)", fontsize=11, color=INK)
ax.set_ylabel("Subjects", fontsize=11, color=INK)
ax.set_title(f"Decoding accuracy across {len(accs)} subjects", fontsize=12, color=INK)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "emg_accuracy_hist.png"), facecolor="white")
plt.close(fig)

print("\nSaved figures to", OUT)
