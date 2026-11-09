# Halvorsen: fester Context versus echter Context

Branch: `experiment/halvorsen-fixed-context`, abgezweigt von `include_Cphi_t`.

Es werden sechs Modelle mit externem phi-Eingang trainiert:

| Parametersatz | Trainings-phi | Context-Bedingungen |
| --- | --- | --- |
| `negative3` | -1, -0.75, -0.5 | `fixed`, `matched` |
| `interleaved3` | -0.25, 0.25, 0.75 | `fixed`, `matched` |
| `spanning5` | -1, -0.5, 0, 0.5, 1 | `fixed`, `matched` |

`fixed` verwendet zufaellige Context-Fenster ausschliesslich von phi=-0.75,
im Training und in der Vorhersage. Es ist ein fester **Parameterwert**, nicht
ein einziges unveraendertes Fenster. Die Referenztrajektorie und der Zeitpunkt
werden unabhaengig vom Zielparameter gezogen. `matched` verwendet den echten,
unmittelbar vorangehenden Context derselben Zieltrajektorie.

Beide Varianten erhalten exakt dieselben Ground-Truth-Zielfenster und denselben
Ziel-phi-Eingang. Der Context-phi wird als Metadatum gespeichert, aber nicht ins
Gating oder in die Experten eingespeist. Die Architektur bleibt unveraendert.

## Fenster und Hyperparameter

- Context: 500 Punkte. Overlap: 0.
- Ziel: x0 plus 200 vorherzusagende Schritte, also `data.npy` mit 201 Punkten.
  Context `[s, s+500)`, x0 bei `s+500`, Loss gegen `[s+501, s+701)`.
- Initialisierung aus x0 des Zielfensters, auch bei fremdem Context. x0 erhaelt
  keine zusaetzliche Trainingsaugmentation; fuer die restlichen Inputs bleibt
  die bisherige Noise-Augmentation erhalten. Die Quelldaten selbst sind verrauscht.
- 30 AL-RNN-Experten, 10 latente Dimensionen pro Experte, 2 PWL-Einheiten.
- 500 Epochen, Batchgroesse 8, 20 Batches/Epoche, Seed 8, 4 Threads.
- RAdam, Lernrate 0.004 bis 0.00001, alpha=1, Teacher Forcing alle 10 Schritte.
- Noise 0.02, Regularisierung 0.1, Auswertung/Checkpoint alle 25 Epochen.
- 16 zufaellige Trainingsfenster pro Quelltrajektorie.

Die bisherigen Halvorsen-Konfigurationen haben 10 Experten und 30 latente
Dimensionen. Hier gilt ausdruecklich die gewuenschte umgekehrte Aufteilung.
Die uebrigen Werte stammen aus den vorhandenen Halvorsen-Laeufen.

Alle Zielfenster werden neu aus dem gemeinsamen Neun-Parameter-Datensatz
gesampelt. `negative3` entspricht der Parameterauswahl von `14to158_3params`;
die alten, teilweise ueberlappenden Fenster werden nicht wiederverwendet.
Die letzten 10000 Schritte jeder Quelltrajektorie bleiben fuer die Auswertung
reserviert. Auch feste Trainings-Contexts stammen nur aus dem davorliegenden
Bereich. Quellindizes und Zeitindizes werden fuer jedes Fenster gespeichert.
Falls ein fester Context und ein Ziel aus derselben Quelltrajektorie stammen,
wird ihre zeitliche Ueberlappung explizit ausgeschlossen.

Die Daten sind bereits standardisiert. Deshalb erfolgt keine weitere Skalierung
anhand des Contexts; sonst wuerde der Context-Tausch auch die Koordinatenskala
veraendern. Das gilt fuer beide Bedingungen und fuer alle Auswertungen.

## Auf den Server bringen

Falls die Aenderungen noch nicht committed/gepusht sind, lokal im Repository:

```powershell
git add .gitignore .gitattributes README.md experiments/halvorsen_context src/dynamix/training/training.py src/dynamix/training/training_setup.py src/dynamix/utilities/dataset.py
git commit -m "Prepare Halvorsen fixed-context ablation"
git push -u origin experiment/halvorsen-fixed-context
```

Dann auf dem Server, bei sauberem Working Tree:

```bash
cd /export/home/klenkeit/DynaMix-python-b-tipping
git fetch origin
git switch --track origin/experiment/halvorsen-fixed-context
# Wenn der Branch bereits lokal existiert: git switch experiment/halvorsen-fixed-context
# Danach bei Bedarf: git pull --ff-only
source venv/bin/activate
```

Die Skripte verwenden dieses Repository ueber `PYTHONPATH`. Eine vorhandene
virtuelle Umgebung wird genutzt; andernfalls wird `venv` oder `.venv` aktiviert.
Shell-Dateien sind ueber `.gitattributes` auf LF festgelegt. `bash script.sh`
funktioniert ohne `chmod`.

## Daten auf dem Server

Standardquelle:
`/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params`.

Darin oder im Unterverzeichnis `trajectories` werden benoetigt:

```text
trajectories.npy          # (72, 100000, 3) in den vorhandenen lokalen Daten
trajectory_phi.npy       # (72, 100000, 1)
trajectory_phi_raw.npy    # (72, 100000, 1)
metadata.json            # empfohlen, fuer Herkunft und Regime-Metadaten
```

Falls auf dem Server nur `data.npy`, `context.npy` und `test.npy` liegen,
zuerst das lokale Verzeichnis
`../generate_trainingdata/diffrentbutconst_prams/single_halvorsen/14to209_noisy_9params/trajectories`
auf den Server kopieren oder `DATA_SOURCE` auf die vorhandenen vollen
Trajektorien setzen. Aus bereits ueberlappenden Fenstern laesst sich der
gewuenschte kontrollierte Datensplit nicht verlaesslich rekonstruieren.

## Starten

Alles: zwei Trainings parallel, danach Tests und Bifurkationen:

```bash
cd /export/home/klenkeit/DynaMix-python-b-tipping
export GPU_FIXED=0 GPU_MATCHED=1 GPU_ID=0
export ANALYSIS_DIR=/export/home/klenkeit/diffrentbutconst_prams
bash experiments/halvorsen_context/run_all.sh
```

Oder einzeln:

```bash
GPU_FIXED=0 GPU_MATCHED=1 bash experiments/halvorsen_context/train.sh
GPU_ID=0 bash experiments/halvorsen_context/trajectories.sh
GPU_ID=0 bash experiments/halvorsen_context/bifurcations.sh
```

Die drei Parametersaetze laufen nacheinander, `fixed` und `matched` jeweils
parallel. Mit nur einer GPU laufen standardmaessig beide Prozesse auf `GPU_ID`;
bei wenig Speicher seriell starten:

```bash
PARALLEL=0 GPU_ID=0 bash experiments/halvorsen_context/train.sh
```

Die GPU-Nummern beziehen sich auf die fuer den Prozess sichtbaren GPUs
(also nach einem eventuellen `CUDA_VISIBLE_DEVICES`).

Mit `GPUS` lassen sich alle sechs Trainings auf einen Pool von einer bis sechs
GPUs verteilen. Eine GPU bearbeitet dabei jeweils ein eigenstaendiges Modell;
ein einzelnes Modell wird nicht auf mehrere GPUs aufgeteilt.

```bash
# Alle sechs Trainings gleichzeitig, ein Modell pro GPU
GPUS="0,1,2,3,4,5" GPU_ID=0 bash experiments/halvorsen_context/run_all.sh
# Drei Trainings gleichzeitig, pro GPU zwei Modelle nacheinander
GPUS="0,2,4" GPU_ID=0 bash experiments/halvorsen_context/run_all.sh
```

Die Reihenfolge der sechs Jobs ist `negative3/fixed`, `negative3/matched`,
`interleaved3/fixed`, `interleaved3/matched`, `spanning5/fixed`,
`spanning5/matched`. Sie werden reihum auf die angegebenen GPUs verteilt.
Leerzeichen statt Kommas sind ebenfalls erlaubt. `GPUS` hat Vorrang vor
`GPU_FIXED` und `GPU_MATCHED`. Mit `PARALLEL=0` laufen die Jobs weiterhin seriell.
Ohne `GPUS` bleibt es beim paarweisen Start mit `GPU_FIXED` und `GPU_MATCHED`.

Die anschliessenden Trajektorien- und Bifurkationsauswertungen verwenden
weiterhin die einzelne GPU aus `GPU_ID`; der Pool betrifft die Trainings.

Optional im Hintergrund:

```bash
GPU_FIXED=0 GPU_MATCHED=1 GPU_ID=0 nohup bash experiments/halvorsen_context/run_all.sh > halvorsen_context.log 2>&1 &
tail -f halvorsen_context.log
```

Vollstaendige Trainingslogs liegen separat in `EXPERIMENT_ROOT/logs`.
Abgeschlossene Modelle werden beim erneuten Start uebersprungen. Unvollstaendige
Runs werden nicht ueberschrieben; dafuer einen neuen Experimentordner waehlen
oder den betroffenen Run vorher verschieben. Bei anderen Trainingshyperparametern
immer einen neuen `EXPERIMENT_ROOT` verwenden.

## Ergebnisse

Standardordner:
`results/halvorsen_context_phi_-0.75/seed_8/`.

```text
datasets/manifest.json                # Einstellungen, Herkunft, Parametersaetze
datasets/{split}/{fixed,matched}/     # identische Ziele, unterschiedliche Contexts
datasets/evaluation/                 # Testtails aller 9 Parameter und Referenzpool
models/{split}/{fixed,matched}/       # config.json, Checkpoints, Trainingsplots
logs/                                # sechs Trainingslogs
trajectories_trained/summary.csv      # Medianmetriken je Modell und phi
trajectories_trained/{run}/           # alle Vorhersagen, Expertgewichte, Metriken, Plots
bifurcations_trained/                 # Cache, Diagnostik und Diagramme
```

Die Testauswertung simuliert **alle** Testtrajektorien bei allen neun Werten,
in den vorhandenen Daten also 8 je phi und 72 je Modell. Der erste Punkt nach
dem 500-Schritte-Context ist x0; die folgenden 9499 Schritte werden frei vorhergesagt.
Keine Teacher-Forcing-Korrekturen in den Auswertungs-Rollouts. Gespeichert werden
Rohvorhersagen, mittlere Expertgewichte sowie Dstsp, DH, MASE und MSE der ersten
200 Schritte. Die Uebersichtsplots zeigen je phi eine Trajektorie mit medianem
Dstsp. Nicht-endliche Rollouts werden als divergiert ausgewiesen.
Die Tests aller neun Parameter werden auch waehrend des Trainings zur Diagnose
berechnet; sie gehen nicht in Gradienten, Lernrate oder Checkpoint-Auswahl ein.

Das Bifurkationsskript nutzt die Funktionen aus
`diffrentbutconst_prams/compare_model_bifurcation_diagrams.py`, mit explizitem
experimentgerechtem Context und Ziel-x0. Es simuliert Ground Truth fuer 320
Parameterwerte im tatsaechlichen Rohparameterbereich des Quelldatensatzes,
mit dt=0.01 und 5000 Burn-in-Schritten. Pro Modell folgen 15000 freie Schritte;
die ersten 1000 werden vor der Extraktion lokaler Maxima verworfen, ebenso bei
der Ground Truth. Standardkomponente ist z. Es entstehen gemeinsame Diagramme
fuer alle sechs Modelle sowie je eines fuer die drei `fixed`- und drei
`matched`-Modelle, zusaetzlich Einzelplots und Trainingsparameter-Markierungen.
Der Cache prueft Einstellungen und Checkpoint-Hashes vor Wiederverwendung.

Die Referenz im Bifurkationssweep stammt aus dem zurueckgehaltenen
phi=-0.75-Contextpool. Pro Rollout wird fuer alle Sweep-Parameter dasselbe
Referenzfenster benutzt. `ROLLOUTS=8` nutzt beispielsweise alle acht
Referenztrajektorien. Der Zielstartpunkt ist fuer alle sechs Modelle identisch.

Haeufige Anpassungen:

```bash
# Laengerer Test-Rollout bzw. kleinere Sweep-Aufloesung
TRAJECTORY_STEPS=5000 bash experiments/halvorsen_context/trajectories.sh
NUM_PARAMETERS=80 BIFURCATION_STEPS=5000 bash experiments/halvorsen_context/bifurcations.sh
# Alle drei Komponenten
bash experiments/halvorsen_context/bifurcations.sh --components x y z
# Vorhandenen Sweep nur neu plotten (gleiche Sweep-Einstellungen beibehalten)
bash experiments/halvorsen_context/bifurcations.sh --plot-only
# Andere Wiederholung: getrennte Daten, Modelle und Ergebnisse
SEED=9 bash experiments/halvorsen_context/run_all.sh
```

## Interpretation und Context-Tausch

Ein erfolgreicher Lauf mit festem Context zeigt, dass ein zum Zielparameter
passender Context fuer diese Aufgabe nicht notwendig ist. Er beweist noch nicht,
dass ausschliesslich der C*phi-Term das Verhalten steuert: Das Gating haengt
auch vom aktuellen latenten Zustand z ab, der wiederum von phi beeinflusst wird.
Ausserdem bleibt ein Context zur Erkennung der Halvorsen-Familie vorhanden.
Bei `interleaved3` und `spanning5` ist der Referenzparameter -0.75 kein Ziel im
Training; seine Beobachtungen stehen dem Modell trotzdem als Context zur Verfuegung.

Fuer einen direkteren Test der Context-Abhaengigkeit koennen dieselben sechs
Checkpoints zusaetzlich mit getauschtem Context ausgewertet werden:

```bash
EVAL_CONTEXT=fixed bash experiments/halvorsen_context/trajectories.sh
EVAL_CONTEXT=matched bash experiments/halvorsen_context/trajectories.sh
# Entsprechend auch mit bifurcations.sh; separate Ausgabeordner.
```

Die mittleren Expertgewichte sind innerhalb desselben Checkpoints vergleichbar;
Expertennummern verschiedener Trainings haben keine feste gemeinsame Bedeutung.
Mehrere Seeds und Referenzfenster helfen, Unterschiede abzusichern.

## Lokale Pruefungen

```bash
python experiments/halvorsen_context/test_experiment.py
python experiments/halvorsen_context/smoke_test.py \
  --source /path/to/14to209_noisy_9params \
  --analysis-dir /path/to/diffrentbutconst_prams \
  --output results/halvorsen_context_smoke
```

Der Smoke-Test trainiert alle sechs Modelle fuer je einen Batch auf echten
Halvorsen-Daten mit 500 Context- und 200 Vorhersageschritten. Er prueft danach
die neun Testparameter und einen winzigen Bifurkationssweep auf der CPU.
Diese kurzen Laeufe pruefen die Pipeline, nicht die Modellguete.
