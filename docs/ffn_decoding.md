## Decoding body kinematics from striatal population activity
 
### Data and trial selection
 
Kinematic decoding was performed separately for each recording session in three rats (Bidoof, Xerneas, Onix).
 
- **Units:** single units were included if they fired at least 500 spikes over the session and had a median waveform amplitude of at most 50 (the `amplitude_median` field in the unit table).
- **Session-level gates:** the session had to have video and pose estimates, and its video-to-ephys frame alignment had to be linear. Video frames for each choice poke were obtained from an LED-based alignment, and frame index was regressed against ephys time; sessions with a nonlinear fit were excluded.
- **Trial-level gates:**
  - the poke had a verified video frame;
  - the snout position at the poke matched the chosen port (per-trial port check);
  - all behavioural fields were present;
  - the last exit from the reward port preceded the choice poke by 0–3,000 ms;
  - entry to the reward port followed the poke by 0–5,500 ms.
- **Further exclusions:** trials whose decoding window extended outside the tracked part of the video, or contained missing pose values, were removed. Sessions needed at least 50 remaining trials.
 
The final dataset comprised:
 
| rat | sessions | trials | units (median per session, range) |
|---|---|---|---|
| Bidoof | 13 | 4,645 | 357 (24, 4–53) |
| Xerneas | 22 | 14,308 | 579 (28, 9–46) |
| Onix | 28 | 13,469 | 267 (8, 4–16) |
 
### Neural features
 
All event times were taken from the acquisition system's hardware sample counter (30 kHz), corrected for samples dropped at the start of a recording.
 
- **Trial window:** from 200 ms before the animal last left the reward port (end of the previous trial) to 200 ms after it next entered the reward port. The window was referenced to the choice poke and divided into 30 ms bins.
- **Spike counts:** each unit's spikes were counted in every bin.
- **Input vector:** each bin *t* to be decoded was represented by the counts of all N units in bins *t − 1*, *t* and *t + 1*, giving a 90 ms window centred on the target bin and 3N features.
- **Edge bins:** the first and last bin of each trial lacked a full context window and were not decoded.
 
### Kinematic targets
 
**Pose estimation.** Nine body points were tracked with a single-instance SLEAP model: nose, snout base, implant, left and right ear, left and right body markers, back centre and tail base. Video was 90.0356 frames/s; this rate was measured from the frame-versus-ephys-time fit, with s.d. 0.0009 across sessions.
 
**Pose processing,** applied to the continuous session-long trace rather than per trial, so that a frame received the same value regardless of trial:
- no confidence threshold was applied, since the nose is often occluded at the port and low-scoring;
- single-frame excursions larger than 50 px that immediately reversed were removed;
- gaps inside the tracked span were filled by linear interpolation;
- traces were smoothed with a Savitzky–Golay filter (9-frame window, cubic polynomial);
- velocity (px/s) was the first derivative of the same local polynomial fit.
 
**Decoding target.** For each bin, the target was the position or velocity of all nine points in x and y (18 values) at the video frame nearest the bin centre. That frame was computed as the poke frame plus the bin centre's offset from the poke converted to frames. Position and velocity were decoded in separate models.
 
### Decoder
 
The decoder was a fully connected feed-forward network with the architecture of the `DenseNNDecoder` in the KordingLab Neural_Decoding package, implemented in Keras 3.14 (TensorFlow 2.21 backend).
 
**Architecture:**
- input: 3N spike-count features;
- two hidden layers of 400 rectified linear units, each followed by dropout (p = 0.25);
- a linear output layer of 18 units.
 
**Preprocessing:** input features and targets were z-scored using the mean and standard deviation of the training set only. Predictions were transformed back to pixel units before evaluation.
 
**Training:** mean squared error loss, Adam optimiser (default settings, learning rate 10⁻³), 50 epochs, mini-batches of 128 bins. A batch size of 128 was chosen after a pilot comparison on one session (29 units, 704 trials). Batch sizes of 32, 64, 128 and 256 gave R² within the variation across cross-validation folds (position 0.586–0.593; velocity 0.425–0.435), and 128 trained 2.2× faster than 32. Random seeds were fixed per fold, and a new network was trained for every fold.
 
### Cross-validation
 
Two partitioning schemes were used. Folds were defined once per session over its trials, so all models, variables and unit counts for a session were evaluated on identical splits.
 
- **Trial-wise cross-validation:** trials were randomly assigned to three folds. Each model was trained on two folds and tested on the third, so every trial was predicted exactly once by a model that had not seen it.
- **Leave-one-port-out:** for each choice port, the model was trained on trials to the other ports and tested on trials to that port. This tests whether the neural code for movement generalises to an approach the decoder never encountered in training.
 
### Performance measures
 
Predictions were compared with the true kinematics over all decoded bins of the test trials in each fold.
 
- **R²:** the coefficient of determination, variance-weighted across the 18 targets (scikit-learn 1.7). R² was also computed for each target separately, so that values could be reported for individual keypoints or keypoint sets.
- **Shape R²:** the same measure after subtracting each trial's own mean from both the true and predicted traces of every target. It measures how well the movement within a trial is decoded, regardless of a constant offset over the trial, such as the location of the chosen port. Under leave-one-port-out, plain R² is dominated by the held-out port's spatial offset, which the decoder cannot learn by construction; shape R² was therefore the primary measure for that scheme. For velocity, whose trial mean is close to zero, the two measures were nearly identical.
- **Session value:** each measure was averaged over folds to give one value per session.
 
### Dependence on population size
 
To measure how decoding accuracy scales with the number of recorded units, the decoder was retrained on random subsets of each session's units: 2, 4, 6, … units, and the full population. One subset was drawn per size, with a fixed seed.
 
**Same sessions at every point.** Sessions differ in how many units they have, so averaging across sessions at each subset size would change the set of sessions along the curve. To avoid this, a minimum unit count was set for each rat: Bidoof 20, Xerneas 24 and Onix 8, which keeps 11, 16 and 19 sessions respectively. Only sessions with at least that many units were included, and curves were shown only up to that size, so every point of a rat's curve averages the same sessions.
 
**Across-rat summaries.** For figures summarising all three rats, each rat's curve was first averaged over its sessions, and the curves were then averaged across rats (mean ± s.e.m., n = 3 rats). In these summaries, each rat included only sessions with at least as many units as the largest subset size shown.
 
### Example visualisations
 
Example heatmaps show true and predicted trajectories for held-out trials from the cross-validation models.
- **Layout:** each row spans one trial's decoded window, aligned to the exit from the reward port. Trials are grouped by port and sorted by duration, with an equal number of trials per port.
- **Pooled heatmaps:** to show all four ports, trials from several sessions of the same rat were combined. Each trial is still a held-out prediction from its own session's model. The top-K sessions by R² were used, plus the next-best sessions with trials at any port that was otherwise under-represented. Trials were drawn equally from the contributing sessions within each port.
- **Pooling is for display only:** all reported accuracies are per-session values, never computed on pooled trials.
 
---
 
Three things to check before using this text:
 
1. **The amplitude criterion is an upper bound,** and its units aren't recorded in the code. The old code called it `min_amplitude` but applied it as a maximum. Confirm that this is what you intended and give the units in the text.
2. **The nose is mostly interpolated.** Only about 20–27% of nose frames within the tracked span are actually observed, and the decoder was trained on the interpolated values. If you report nose-specific results, it's worth saying so.
3. **Session counts:** the minimum-unit numbers (11, 16 and 19 sessions) come from cross-validation. Leave-one-port-out has slightly fewer for Xerneas (14 at the 24-unit cutoff), because two of its sessions have no leave-one-port-out results.