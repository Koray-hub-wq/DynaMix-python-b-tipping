# Halvorsen mit festen zufälligen Expertengewichten

Sechs Trainings ohne Context und ohne Gating-Netzwerk. Der bekannte Zielparameter
φ bleibt als additiver Eingang **Cφ** erhalten. Bei jedem Trainingsbatch werden
neue Fenster mit **200 Zeitschritten** aus den vollen Trajektorien gezogen.

| Split | Trainings-φ | Experten | PWL pro Experte |
| --- | --- | --- | --- |
| `negative3` | -1, -0.75, -0.5 | 10 und 40 | 2 |
| `spanning5` | -1, -0.5, 0, 0.5, 1 | 10 und 40 | 2 |
| `interleaved3` | -0.25, 0.25, 0.75 | 10 und 40 | 2 |

Implementierung: [fixed_mixture.py](../../src/dynamix/model/fixed_mixture.py).
Die bisherigen DynaMix-Modelle und Context-Experimente bleiben unverändert.

## Modell und Interpretation

Wie in DynaMix bekommen alle Experten denselben latenten Zustand z:

```text
f_i(z, φ) = A_i * z + W_i Φ_P(z) + h_i + C_i φ
z_next    = Σ_i w_i f_i(z, φ)
x_next    = z_next[:3]
```

Die positiven Gewichte werden einmal aus `Dirichlet(1,...,1)` gezogen
(unabhängige Exponential-Zufallszahlen, auf Summe 1 normiert). Sie sind ein
PyTorch-Buffer, keine trainierbaren Parameter, und Bestandteil jedes Checkpoints.
Ein Gewichtssatz gilt für alle Zeitpunkte, Trajektorien und φ-Werte, auch nach
dem Laden. Für eine feste Expertenzahl werden bei allen drei Splits dieselben
Gewichte und dieselbe Modellinitialisierung verwendet. Alle Expertenparameter,
C-Matrizen und die Initialisierungsmatrix B werden trainiert.

**Diese feste Mischung ist mathematisch selbst ein einzelnes AL-RNN**, denn
die ReLU-Koordinaten sind für alle Experten gleich:
`A_eff=Σw_i A_i`, entsprechend für `W_eff`, `h_eff`, `C_eff`. Es entstehen bei
40 Experten somit weiterhin nur zwei ReLU-Koordinaten, keine 80 unabhängigen
nichtlinearen Einheiten. Der Code nutzt diese exakte Zusammenfassung für die
Rechenschritte, erhält aber den Gradientenfluss zu allen originalen Experten.
Er speichert auch alle 10 bzw. 40 Experten. Der Unterschied betrifft die
Parametrisierung, Initialisierung und Optimierung, nicht die Modellklasse.

## Daten und Training

Standardquelle auf dem Server:

```text
/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories
```

Benötigt werden `trajectories.npy` `(Serie, Zeit, 3)`, `trajectory_phi.npy` und
`trajectory_phi_raw.npy` `(Serie, Zeit, 1)`. Das Unterverzeichnis `trajectories/`
wird ebenfalls akzeptiert. Falls die `*_constants.npy`-Dateien vorhanden sind,
werden sie gegen die vollen Parameterarrays geprüft. `metadata.json` wird,
falls vorhanden, eingelesen und über seinen Hash dokumentiert.

- M=30 latente Dimensionen, 2 PWL-Einheiten, Cφ pro Experte; deterministische Experten.
- Chronologischer Split wie bei den Single-ALRNNs: Training `[0,80000)`, Test
  `[80000,T)`. Kein Trainingsfenster berührt den Testbereich.
- 200 Übergänge pro Fenster: x0 plus 200 Zielpunkte. Batchgröße 8, 20 Batches
  pro Epoche, 500 Epochen. Ausgewogene Auswahl der Trainings-φ; zufällige Serie
  und Startzeit innerhalb des jeweiligen φ.
- Sparse Teacher Forcing alle 10 Schritte, α=1; Startzustand aus dem Ziel-x0,
  beobachtete Koordinaten direkt kopiert, versteckte Koordinaten mit B initialisiert.
- RAdam, exponentielle Lernrate 0.004 bis 0.00001, Gradient-Clipping bei 10.
- Zusätzliche Noise 0.02 nur auf Trainingsinputs, außer x0. Die Targets bleiben
  unverändert; die Quelldaten sind bereits verrauscht und standardisiert.
- Keine zusätzliche Standardisierung anhand eines Contexts. Kein Gating und
  daher auch kein Gating-σ-Regularisierungsterm. Loss: MSE der drei Beobachtungen.
- Seed 8, separater Gewichtsseed 8, 4 Threads. Fenster- und Noise-RNG sind von
  der Modellinitialisierung getrennt: 10 und 40 Experten erhalten dieselben
  Fenster und dieselbe Augmentation bei gleicher Laufkonfiguration.
- Checkpoints alle 25 Epochen. Das finale Modell wird ausgewertet; keine Auswahl
  anhand der Fehler auf unbekannten Parametern.

M=30 entspricht dem allgemeinen DynaMix-Standard. Das referenzierte
`halvorsen_context`-Experiment verwendet dagegen M=10, 30 Experten und den Split
vor dem letzten 10000-Punkte-Testbereich. Für einen isolierten Vergleich des
Gatings müssen M, Datensplit, Trainingsaufwand und Auswertungsstartpunkte der
Vergleichsläufe angeglichen werden. Die neuen Standardeinstellungen sind deshalb
nicht als bereits vollständig gepaarter Vergleich zu diesen alten Läufen zu lesen.

## Serverstart

Die neuen Dateien müssen im Server-Repository liegen: der Ordner
`experiments/halvorsen_fixed_mixture/` und `src/dynamix/model/fixed_mixture.py`.
Die vorhandene DynaMix-Pythonumgebung wird benutzt; `venv`/`.venv` wird bei Bedarf
automatisch aktiviert. Es werden keine zusätzlichen Pakete benötigt.

```bash
cd /export/home/klenkeit/DynaMix-python-b-tipping
GPUS="0 1" bash experiments/halvorsen_fixed_mixture/run_all.sh
```

`run_all.sh` arbeitet die Expertenzahlen vollständig nacheinander ab:

1. Alle drei Parametersätze mit **10 Experten** trainieren.
2. Ihre Testtrajektorien, 3D-/Zeitreihengrafiken und Bifurkationsdiagramme erzeugen.
3. Alle drei Parametersätze mit **40 Experten** trainieren.
4. Ihre Testtrajektorien und Diagramme erzeugen.
5. Die direkten 10-vs-40-Bifurkationsvergleiche aus den geprüften Caches erstellen,
   ohne erneute ODE-Simulation oder Modellrollouts.

Erst wenn Training und beide Auswertungen der 10er-Stufe erfolgreich beendet
sind, startet die 40er-Stufe. Innerhalb einer Stufe werden die drei Parametersätze
auf die angegebenen GPUs verteilt, mit höchstens einem Training pro GPU.
Auch `train.sh` beendet die 10er-Trainings vor den 40er-Trainings.
Mit einer GPU laufen alle Trainings nacheinander:

```bash
GPU_ID=0 bash experiments/halvorsen_fixed_mixture/run_all.sh
```

Die Analyse jeder Stufe verwendet die erste angegebene GPU. GPU-Nummern gelten
innerhalb eines eventuell gesetzten `CUDA_VISIBLE_DEVICES`. `run_all.sh` prüft
Daten, GPUs und Analysebackend vor dem Training. Für Bifurkationen wird das
bestehende Backend samt seiner Hilfsdateien verwendet:
`/export/home/klenkeit/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py`.

Andere Pfade und Hyperparameter können als Umgebungsvariablen oder CLI-Flags
gesetzt werden. PWL bleibt fest bei 2:

```bash
GPUS="0 1 2" LATENT_DIM=10 EPOCHS=1000 BATCH_SIZE=16 \
EXPERIMENT_ROOT=/export/home/klenkeit/fixed_mixture_m10 \
ANALYSIS_DIR=/export/home/klenkeit/diffrentbutconst_prams \
bash experiments/halvorsen_fixed_mixture/run_all.sh

# Dieselben Einstellungen bei getrennten Stufen erneut angeben/exportieren:
bash experiments/halvorsen_fixed_mixture/train.sh --gpus 0 1
bash experiments/halvorsen_fixed_mixture/trajectories.sh --gpus 0
bash experiments/halvorsen_fixed_mixture/bifurcations.sh --gpus 0

# Nur negative3 und 10 Experten, beispielsweise als Vorlauf:
bash experiments/halvorsen_fixed_mixture/run_all.sh --splits negative3 --expert-counts 10
```

Weitere Optionen: `--source`, `--split-index`, `--sequence-length`,
`--teacher-forcing-interval`, `--batches-per-epoch`, `--noise-level`,
`--trajectory-steps`, `--bifurcation-steps`, `--num-parameters`, `--rollouts`,
`--burn-in-steps`, `--discard-steps`, `--components x y z`, `--weights-seed`.
`python experiments/halvorsen_fixed_mixture/run.py --help` zeigt alle Optionen.
Die Option `--phi-dim 0` erlaubt zusätzlich ein vollständig autonomes Modell;
dessen Sweep variiert nur den aus Ground Truth entnommenen Startzustand, nicht
die Dynamik über einen φ-Eingang. Standard und vereinbartes Experiment ist φ-Dim 1.

Beim Wiederholen werden vollständige Trainings nur bei identischen Daten,
Code und Einstellungen übersprungen. Unterbrochene Trainings werden aus
`last_training_state.pt` fortgesetzt, inklusive Optimizer, Scheduler und RNGs.
Seit dem letzten Checkpoint nicht gespeicherte Epochen werden reproduziert.
Bei geänderten Einstellungen einen neuen `EXPERIMENT_ROOT` verwenden.

## Ergebnisse

Standardausgabe:
`results/halvorsen_fixed_mixture/seed_8_w8_m30_p2_phi1/`

```text
experiment_plan.json
logs/<split>_experts_<10|40>.log
models/<split>/experts_<10|40>/
  config.json                    # Gewichte, Parameter, Daten- und Code-Hashes
  training_history.json
  checkpoints/final_model.pt
  checkpoints/last_training_state.pt
trajectories/
  experts_<10|40>/               # eigene Auswertung pro abgeschlossener Stufe
    test_reference.npz
    summary.csv
    <split>_experts_<10|40>/
      predictions.npz            # alle Serien, freie Vorhersagen
      metrics.json               # Fehler pro Serie
      attractors_9params.png      # 3x3-Übersicht mit Ground Truth
      timeseries_9params.png
      trajectory_phi_*.png       # je φ: Ground Truth und Modell nebeneinander
bifurcations/
  bifurcation_cache.npz          # zusammengeführter Vergleichscache
  sweep_initial_conditions.npz
  diagnostics.json
  all/                           # alle sechs Modelle und Ground Truth
  experts_10/                    # vollständige 10er-Stufe samt eigenem Cache
  experts_40/                    # vollständige 40er-Stufe samt eigenem Cache
  negative3/                     # direkter Vergleich 10 gegen 40
  spanning5/
  interleaved3/
```

Die einzeln gestarteten `trajectories.sh` und `bifurcations.sh` werten wie bisher
alle ausgewählten Expertenzahlen zusammen aus. Ihre Ausgabe liegt direkt unter
`trajectories/` bzw. `bifurcations/`; `--expert-counts 10` begrenzt sie auf die 10er.

Die Testauswertung umfasst alle neun Parameter und alle vorhandenen Serien
(lokal 72, je acht pro φ), standardmäßig 10000 freie Vorhersageschritte ab
`x[80000]`. Erster bewerteter Punkt ist `x[80001]`. Die 3D-Grafiken zeigen pro
φ die erste Quellserie, für alle Modelle dieselbe. Die NPZ-Dateien enthalten
alle Serien. `summary.csv` enthält Median-Dstsp, DH, MASE, MSE über die ersten
200 Schritte sowie die Anzahl nichtendlicher und divergierter Vorhersagen und den Trainingsstatus.

Als divergiert gilt eine Vorhersage ab dem ersten nichtendlichen Wert oder einer
Beobachtungskoordinate mit Betrag über `1e6` (die Daten sind standardisiert).
`metrics.json` dokumentiert `diverged`, den ersten betroffenen Vorhersageschritt
(ab 1 gezählt) und den maximalen endlichen Betrag. Die Grafiken zeigen nur den
Präfix davor und markieren den Abbruch ausdrücklich. Die vollständigen
Rohvorhersagen in `predictions.npz` bleiben unverändert. Achsengrenzen werden
in Float64 berechnet, damit große Float32-Werte keinen Plot-Abbruch verursachen.
Dstsp und DH werden für divergierte Rollouts nicht berechnet; MASE über die
ersten 10 und MSE über die ersten 200 Schritte bleiben erhalten, sofern der
jeweilige kurze Horizont noch nicht divergiert. `summary.csv` zählt betroffene
Serien in `n_diverged`; fehlende Langzeitmetriken sind kein erfolgreicher Rollout.
Auch die Bifurkationsanalyse kennzeichnet solche Rollouts in `diagnostics.json`
und nimmt ihre Maxima nicht in das Diagramm auf.

Nach einem Auswertungsabbruch denselben `run_all.sh`-Befehl erneut starten,
sobald die korrigierte `evaluate.py` auf dem Server liegt. Bereits abgeschlossene
Trainings mit passenden Einstellungen werden übersprungen. Der Plot-Fix ändert
keine Trainingsparameter oder Trainingsdateien.

Die Bifurkationsanalyse simuliert standardmäßig 320 φ-Werte von -1 bis 1 mit
dem bestehenden Halvorsen-ODE-Backend. Sie verwendet die tatsächliche affine
Zuordnung aus den Quelldaten (lokal `a = 1.75 + 0.35 φ`, also bis 2.1 trotz
des Ordnernamens `14to209`). 5000 ODE-Burn-in-Schritte; 15000 freie Modellschritte,
die ersten 1000 werden vor der Maximaextraktion verworfen. Ground Truth wird
wie im bisherigen Backend pro simulierter Trajektorie standardisiert und nicht
verrauscht. Alle Modelle erhalten denselben einzelnen GT-Startpunkt und φ,
keinen Context. Bei mehreren Rollouts gibt es verschiedene ODE-Startbedingungen.
Trainingsparameter sind in den Diagrammen markiert. Standardkomponente ist z;
die Maxima aller drei Koordinaten werden gecacht.

Passende Bifurkationscaches werden wiederverwendet. Mit `--plot-only` werden
ausschließlich passende Caches geplottet; `--force-recompute` erzwingt eine
neue Simulation. Cacheidentität berücksichtigt Daten, Checkpoints, Einstellungen
und die verwendeten Modell-, Treiber- und Backenddateien.

## Lokale Prüfung

```bash
python -m unittest discover -s experiments/halvorsen_fixed_mixture -p test_experiment.py -v

python experiments/halvorsen_fixed_mixture/smoke_test.py \
  --source ../generate_trainingdata/diffrentbutconst_prams/single_halvorsen/14to209_noisy_9params/trajectories \
  --analysis-dir ../generate_trainingdata/diffrentbutconst_prams \
  --output results/halvorsen_fixed_mixture_smoke
```

Der Smoke-Test trainiert alle sechs Modelle für je einen Batch mit 200 Schritten
auf echten Daten und erzeugt kurze Test- und Bifurkationsgrafiken auf CPU.
Er prüft die Pipeline und liefert keine Aussage über die Modellqualität.
