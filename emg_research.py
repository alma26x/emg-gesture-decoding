import os, glob, json, time, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\arial.ttf"); plt.rcParams["font.family"] = "Arial"
except Exception: pass

ROOT = r"C:\Users\afric\emg_decoding\data\EMG_data_for_gestures-master"
OUT  = r"C:\Users\afric\emg_decoding"
INK, TEAL, AMBER, GREY = "#1d2a30", "#0d7a6f", "#c98a0d", "#9aa3a8"
CLASSES = [1, 2, 3, 4, 5, 6]
NAMES = {1: "Rest", 2: "Fist", 3: "Wrist\nflex", 4: "Wrist\next", 5: "Radial\ndev", 6: "Ulnar\ndev"}

def feats(win):
    f = []
    for c in range(win.shape[1]):
        x = win[:, c]; d = np.diff(x)
        f += [np.mean(np.abs(x)), np.sqrt(np.mean(x**2)), np.sum(np.abs(d)),
              int(np.sum(x[:-1]*x[1:] < 0)), int(np.sum(d[:-1]*d[1:] < 0))]
    return f

def load(path):
    try: data = np.loadtxt(path, skiprows=1)
    except Exception: return None
    if data.ndim != 2 or data.shape[1] < 10: return None
    return data

def windows(data, W, S):
    ch, cls = data[:, 1:9], data[:, 9].astype(int)
    X, y, seg = [], [], []; i, n, sid = 0, len(cls), 0
    while i < n:
        c, j = cls[i], i
        while j < n and cls[j] == c: j += 1
        if c in CLASSES and (j - i) >= W:
            s = ch[i:j]
            for k in range(0, len(s) - W + 1, S):
                X.append(feats(s[k:k+W])); y.append(c); seg.append(sid)
            sid += 1
        i = j
    return np.array(X, float), np.array(y, int), np.array(seg, int)

subjects = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
FS = 200.0
print(f"sampling rate {FS:.0f} Hz; subjects={len(subjects)}")
W, S = 40, 20

res = {"RF": [], "LDA": []}; cm_true, cm_pred = [], []; sm_base, sm_smooth = [], []
t0 = time.time()
for sub in subjects:
    fs = sorted(glob.glob(os.path.join(ROOT, sub, "*.txt")))
    if len(fs) < 2: continue
    d1, d2 = load(fs[0]), load(fs[1])
    if d1 is None or d2 is None: continue
    Xtr, ytr, _ = windows(d1, W, S); Xte, yte, seg = windows(d2, W, S)
    if len(set(ytr)) < 2 or len(Xte) == 0: continue
    for name, clf in [("RF", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
                      ("LDA", LinearDiscriminantAnalysis())]:
        clf.fit(Xtr, ytr); pred = clf.predict(Xte)
        res[name].append(accuracy_score(yte, pred))
        if name == "RF":
            cm_true += list(yte); cm_pred += list(pred)
            sm = pred.copy()
            for sid in np.unique(seg):
                idx = np.where(seg == sid)[0]; p = pred[idx]
                for a in range(len(p)):
                    lo, hi = max(0, a-2), min(len(p), a+3)
                    v, c = np.unique(p[lo:hi], return_counts=True); sm[idx[a]] = v[np.argmax(c)]
            sm_base.append(accuracy_score(yte, pred)); sm_smooth.append(accuracy_score(yte, sm))
rf = np.array(res["RF"]); lda = np.array(res["LDA"])
print(f"[1] within-subject done {time.time()-t0:.0f}s  RF={rf.mean()*100:.1f}%  LDA={lda.mean()*100:.1f}%")

def pool(subs):
    X, y = [], []
    for sub in subs:
        for f in sorted(glob.glob(os.path.join(ROOT, sub, "*.txt"))):
            d = load(f)
            if d is None: continue
            Xi, yi, _ = windows(d, W, S)
            if len(Xi): X.append(Xi); y.append(yi)
    return np.vstack(X), np.concatenate(y)
cut = (len(subjects) * 2) // 3
tr_s, te_s = subjects[:cut], subjects[cut:]
Xtr, ytr = pool(tr_s); Xte, yte = pool(te_s)
clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42); clf.fit(Xtr, ytr)
xsub = accuracy_score(yte, clf.predict(Xte))
print(f"[2] cross-subject train={len(tr_s)} test={len(te_s)} subj: {xsub*100:.1f}%")

sweep_W = [20, 40, 60, 80, 120, 160, 200]; sweep_subj = subjects[:10]; sweep = []
for w in sweep_W:
    a = []
    for sub in sweep_subj:
        fs = sorted(glob.glob(os.path.join(ROOT, sub, "*.txt")))
        if len(fs) < 2: continue
        d1, d2 = load(fs[0]), load(fs[1])
        Xtr, ytr, _ = windows(d1, w, max(1, w//2)); Xte, yte, _ = windows(d2, w, max(1, w//2))
        if len(set(ytr)) < 2 or len(Xte) == 0: continue
        clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=42); clf.fit(Xtr, ytr)
        a.append(accuracy_score(yte, clf.predict(Xte)))
    sweep.append(np.mean(a)); print(f"    win {w} ({w/FS*1000:.0f} ms): {np.mean(a)*100:.1f}%")

P, R, F, _ = precision_recall_fscore_support(cm_true, cm_pred, labels=CLASSES, zero_division=0)

summary = {
    "within_RF_mean": float(rf.mean()*100), "within_RF_std": float(rf.std()*100), "n": int(len(rf)),
    "within_LDA_mean": float(lda.mean()*100), "cross_subject": float(xsub*100),
    "smooth_base": float(np.mean(sm_base)*100), "smooth_after": float(np.mean(sm_smooth)*100),
    "sweep_ms": [round(w/FS*1000) for w in sweep_W], "sweep_acc": [float(a*100) for a in sweep],
    "fs": round(FS), "per_gesture_F1": {NAMES[c].replace(chr(10), " "): float(f*100) for c, f in zip(CLASSES, F)},
}
json.dump(summary, open(os.path.join(OUT, "emg_results.json"), "w"), indent=2)
print("\n==================== SUMMARY ====================")
print(f"Within-subject (cross-session)  RF : {rf.mean()*100:.1f}% +/- {rf.std()*100:.1f}%  (n={len(rf)})")
print(f"Within-subject (cross-session)  LDA: {lda.mean()*100:.1f}%")
print(f"Cross-subject (leave-subjects-out) : {xsub*100:.1f}%")
print(f"Temporal smoothing                 : {np.mean(sm_base)*100:.1f}% -> {np.mean(sm_smooth)*100:.1f}%")
print(f"Chance (6 classes)                 : {100/6:.1f}%")
for c, p, r, f in zip(CLASSES, P, R, F):
    print(f"   {NAMES[c].replace(chr(10),' '):12s} P={p*100:4.0f} R={r*100:4.0f} F1={f*100:4.0f}")

cm = confusion_matrix(cm_true, cm_pred, labels=CLASSES).astype(float)
cmn = cm / cm.sum(axis=1, keepdims=True)
labels = [NAMES[c] for c in CLASSES]
fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=160); fig.patch.set_facecolor("white")
ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(labels, fontsize=10); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted", fontsize=12, color=INK); ax.set_ylabel("True", fontsize=12, color=INK)
ax.set_title(f"Confusion matrix (within-subject, cross-session; {len(rf)} subjects)", fontsize=12, color=INK)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else INK, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "emg_confusion_matrix.png"), facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(12, 8.4), dpi=160); fig.patch.set_facecolor("white")
a = ax[0, 0]
bars = a.bar([0, 1, 2], [rf.mean()*100, lda.mean()*100, xsub*100], color=[TEAL, GREY, AMBER], edgecolor="white", width=0.6)
a.errorbar([0, 1], [rf.mean()*100, lda.mean()*100], yerr=[rf.std()*100, lda.std()*100], fmt="none", ecolor=INK, capsize=4)
a.axhline(100/6, ls=":", color="#888", lw=1.2); a.text(2.4, 100/6+1, "chance", color="#888", fontsize=9, ha="right")
a.set_xticks([0, 1, 2]); a.set_xticklabels(["RF\n(within-subj)", "LDA\n(within-subj)", "RF\n(cross-subj)"], fontsize=10)
a.set_ylabel("Accuracy (%)", fontsize=11, color=INK); a.set_ylim(0, 100)
a.set_title("Classifier comparison & generalization", fontsize=12, color=INK)
for b, v in zip(bars, [rf.mean()*100, lda.mean()*100, xsub*100]): a.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
a.spines[["top", "right"]].set_visible(False)

a = ax[0, 1]
ms = [w/FS*1000 for w in sweep_W]
a.plot(ms, [s*100 for s in sweep], "-o", color=TEAL, lw=2.2, ms=6)
a.set_xlabel("Window length / control latency (ms)", fontsize=11, color=INK)
a.set_ylabel("Accuracy (%)", fontsize=11, color=INK)
a.set_title("Latency vs accuracy tradeoff", fontsize=12, color=INK)
a.axvspan(0, 300, color=TEAL, alpha=0.06); a.text(150, a.get_ylim()[0]+1, "real-time zone\n(<300 ms)", ha="center", fontsize=9, color=TEAL)
a.spines[["top", "right"]].set_visible(False)

a = ax[1, 0]
a.bar(range(6), F*100, color=TEAL, edgecolor="white")
a.set_xticks(range(6)); a.set_xticklabels([NAMES[c] for c in CLASSES], fontsize=9)
a.set_ylabel("F1 score (%)", fontsize=11, color=INK); a.set_ylim(0, 100)
a.set_title("Per-gesture F1 (within-subject)", fontsize=12, color=INK)
for i, v in enumerate(F*100): a.text(i, v+1.5, f"{v:.0f}", ha="center", fontsize=9, color=INK)
a.spines[["top", "right"]].set_visible(False)

a = ax[1, 1]
b = a.bar([0, 1], [np.mean(sm_base)*100, np.mean(sm_smooth)*100], color=[GREY, TEAL], edgecolor="white", width=0.6)
a.set_xticks([0, 1]); a.set_xticklabels(["Raw\nper-window", "+ temporal\nmajority vote"], fontsize=10)
a.set_ylabel("Accuracy (%)", fontsize=11, color=INK); a.set_ylim(0, 100)
a.set_title("Temporal smoothing (real-time post-processing)", fontsize=12, color=INK)
for bb, v in zip(b, [np.mean(sm_base)*100, np.mean(sm_smooth)*100]): a.text(bb.get_x()+bb.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
a.spines[["top", "right"]].set_visible(False)

fig.suptitle("Surface-EMG gesture decoding — evaluation  (Lobov et al. 2018, 8-ch MYO, 6 gestures)", fontsize=13, color=INK, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "emg_results_panel.png"), facecolor="white"); plt.close(fig)
print("\nsaved: emg_results_panel.png, emg_confusion_matrix.png, emg_results.json")
