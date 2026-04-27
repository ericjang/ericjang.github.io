// components/p4_experiments.jsx
// Part 4 · slide 12 — experiment timeline with interactive figure preview.
// Every experiment in alphago-tutorial/experiments/ is plotted as a dot;
// only figure-bearing dots are hoverable. During auto-play, the figure for
// the most-recently-revealed figure-bearing dot stays shown and flashes
// briefly when a new one arrives.

const EXP = [
  {ts:"2025-12-24_11-15", exp:"2025-12-24_11-15-wider-shallower-model", file:null, short:"wider shallower model"},
  {ts:"2025-12-24_20-54", exp:"2025-12-24_20-54-data-scaling-wider-shallower", file:null, short:"data scaling wider shallower"},
  {ts:"2025-12-24_21-38", exp:"2025-12-24_21-38-data-scaling-larger-batch", file:null, short:"data scaling larger batch"},
  {ts:"2025-12-24_22-05", exp:"2025-12-24_22-05-batch-512", file:null, short:"batch 512"},
  {ts:"2025-12-24_23-09", exp:"2025-12-24_23-09-muon-vs-adamw", file:null, short:"muon vs adamw"},
  {ts:"2025-12-25_13-30", exp:"2025-12-25_13-30-scaling-laws-param-count", file:null, short:"scaling laws param count"},
  {ts:"2025-12-26_10-30", exp:"2025-12-26_10-30-resnet-scaling-laws", file:null, short:"resnet scaling laws"},
  {ts:"2025-12-26_19-13", exp:"2025-12-26_19-13-resnet-scaling-laws", file:null, short:"resnet scaling laws"},
  {ts:"2025-12-26_20-30", exp:"2025-12-26_20-30-data-deduplication-analysis", file:null, short:"data deduplication analysis"},
  {ts:"2025-12-26_21-00", exp:"2025-12-26_21-00-epsilon-greedy-data-generation", file:null, short:"epsilon greedy data generation"},
  {ts:"2025-12-27_02-17", exp:"2025-12-27_02-17-three-dataset-dedup-comparison", file:null, short:"three dataset dedup comparison"},
  {ts:"2025-12-27_14-02", exp:"2025-12-27_14-02-nn-policy-visualization", file:null, short:"nn policy visualization"},
  {ts:"2025-12-27_14-06", exp:"2025-12-27_14-06-critical-batch-size-analysis", file:null, short:"critical batch size analysis"},
  {ts:"2025-12-27_14-10", exp:"2025-12-27_14-10-resnet-100m-baseline-training", file:null, short:"resnet 100m baseline training"},
  {ts:"2025-12-27_14-25", exp:"2025-12-27_14-25-four-dataset-dedup-comparison", file:null, short:"four dataset dedup comparison"},
  {ts:"2025-12-27_15-06", exp:"2025-12-27_15-06-finetune-and-visualize-vs-random", file:null, short:"finetune and visualize vs random"},
  {ts:"2025-12-27_15-13", exp:"2025-12-27_15-13-mup-hyperparameter-search-partial-analysis", file:null, short:"mup hyperparameter search partial analysis"},
  {ts:"2025-12-27_15-33", exp:"2025-12-27_15-33-finetuned-policy-eval-vs-random", file:null, short:"finetuned policy eval vs random"},
  {ts:"2025-12-27_15-42", exp:"2025-12-27_15-42-finetuned-policy-eval-vs-gnugo10", file:null, short:"finetuned policy eval vs gnugo10"},
  {ts:"2025-12-27_16-03", exp:"2025-12-27_16-03-nn-self-play-eval", file:null, short:"nn self play eval"},
  {ts:"2025-12-27_16-15", exp:"2025-12-27_16-15-mup-training-vs-baseline", file:null, short:"mup training vs baseline"},
  {ts:"2025-12-27_16-30", exp:"2025-12-27_16-30-gotransformer-vs-resnet-eval", file:null, short:"gotransformer vs resnet eval"},
  {ts:"2025-12-27_17-26", exp:"2025-12-27_17-26-gotransformer-vs-finetuned-baseline", file:null, short:"gotransformer vs finetuned baseline"},
  {ts:"2025-12-27_17-28", exp:"2025-12-27_17-28-mup-ablation-study", file:null, short:"mup ablation study"},
  {ts:"2025-12-27_18-24", exp:"2025-12-27_18-24-training-metrics-comparison", file:null, short:"training metrics comparison"},
  {ts:"2025-12-27_18-30", exp:"2025-12-27_18-30-mup-lr-sweep", file:null, short:"mup lr sweep"},
  {ts:"2025-12-27_18-46", exp:"2025-12-27_18-46-dataset-comparison", file:null, short:"dataset comparison"},
  {ts:"2025-12-27_19-07", exp:"2025-12-27_19-07-baseline-lr-sweep", file:null, short:"baseline lr sweep"},
  {ts:"2025-12-27_19-25", exp:"2025-12-27_19-25-five-dataset-dedup-comparison", file:null, short:"five dataset dedup comparison"},
  {ts:"2025-12-27_19-41", exp:"2025-12-27_19-41-small-batch-lr-sweep", file:null, short:"small batch lr sweep"},
  {ts:"2025-12-27_20-42", exp:"2025-12-27_20-42-mup-sweep-analysis", file:null, short:"mup sweep analysis"},
  {ts:"2025-12-28_11-05", exp:"2025-12-28_11-05-mup-transfer-validation", file:null, short:"mup transfer validation"},
  {ts:"2025-12-28_11-45", exp:"2025-12-28_11-45-transformer-vs-resnet", file:null, short:"transformer vs resnet"},
  {ts:"2025-12-28_12-30", exp:"2025-12-28_12-30-circuits-analysis", file:null, short:"circuits analysis"},
  {ts:"2025-12-29_10-13", exp:"2025-12-29_10-13-normalized-deduplication-analysis", file:null, short:"normalized deduplication analysis"},
  {ts:"2025-12-29_12-00", exp:"2025-12-29_12-00-nn-selfplay-training-comparison", file:null, short:"nn selfplay training comparison"},
  {ts:"2025-12-29_14-00", exp:"2025-12-29_14-00-iterative-selfplay-training", file:null, short:"iterative selfplay training"},
  {ts:"2025-12-29_17-15", exp:"2025-12-29_17-15-skill-conditioned-training", file:null, short:"skill conditioned training"},
  {ts:"2025-12-29_19-30", exp:"2025-12-29_19-30-skill-conditioning-investigation", file:null, short:"skill conditioning investigation"},
  {ts:"2025-12-29_20-18", exp:"2025-12-29_20-18-selfplay-game-analysis", file:null, short:"selfplay game analysis"},
  {ts:"2025-12-29_21-00", exp:"2025-12-29_21-00-advantage-conditioned-training", file:null, short:"advantage conditioned training"},
  {ts:"2025-12-29_21-22", exp:"2025-12-29_21-22-mcts-implementation", file:null, short:"mcts implementation"},
  {ts:"2025-12-29_21-30", exp:"2025-12-29_21-30-advantage-sigmoid-fix", file:null, short:"advantage sigmoid fix"},
  {ts:"2025-12-29_21-38", exp:"2025-12-29_21-38-advantage-vs-baseline", file:null, short:"advantage vs baseline"},
  {ts:"2025-12-29_22-36", exp:"2025-12-29_22-36-mcts-fast-python", file:null, short:"mcts fast python"},
  {ts:"2025-12-30_00-01", exp:"2025-12-30_00-01-mcts-white-vs-raw-black", file:null, short:"mcts white vs raw black"},
  {ts:"2025-12-30_00-15", exp:"2025-12-30_00-15-mcts-debug", file:null, short:"mcts debug"},
  {ts:"2025-12-30_00-27", exp:"2025-12-30_00-27-onestep-q-estimation", file:null, short:"onestep q estimation"},
  {ts:"2025-12-30_00-30", exp:"2025-12-30_00-30-mcts-trace-debug", file:null, short:"mcts trace debug"},
  {ts:"2025-12-30_01-00", exp:"2025-12-30_01-00-self-play-nn-mix", file:null, short:"self play nn mix"},
  {ts:"2025-12-30_19-14", exp:"2025-12-30_19-14-mcts-vs-raw-policy", file:null, short:"mcts vs raw policy"},
  {ts:"2025-12-30_19-38", exp:"2025-12-30_19-38-mcts-vs-policy-with-value-passing", file:"figures/game_stats.png", short:"mcts vs policy with value passing"},
  {ts:"2025-12-30_20-00", exp:"2025-12-30_20-00-mcts-800-sims-analysis", file:"figures/puct_distributions.png", short:"mcts 800 sims analysis"},
  {ts:"2025-12-30_20-29", exp:"2025-12-30_20-29-mcts-q-value-evolution", file:"figures/early_simulations.png", short:"mcts q value evolution"},
  {ts:"2025-12-30_20-52", exp:"2025-12-30_20-52-mcts-unusual-decisions-analysis", file:"figures/mcts_analysis.png", short:"mcts unusual decisions analysis"},
  {ts:"2025-12-30_21-45", exp:"2025-12-30_21-45-retrain-10k-mcts-analysis", file:"figures/training_curves.png", short:"retrain 10k mcts analysis"},
  {ts:"2025-12-31_00-18", exp:"2025-12-31_00-18-mcts-rollout-verification", file:"figures/value_per_move.png", short:"mcts rollout verification"},
  {ts:"2025-12-31_01-11", exp:"2025-12-31_01-11-mcts-vs-raw-policy", file:"figures/summary.png", short:"mcts vs raw policy"},
  {ts:"2025-12-31_01-25", exp:"2025-12-31_01-25-mcts-vs-raw-100sim", file:"figures/summary.png", short:"mcts vs raw 100sim"},
  {ts:"2025-12-31_12-05", exp:"2025-12-31_12-05-mcts-shared-lib-validation", file:"figures/summary.png", short:"mcts shared lib validation"},
  {ts:"2025-12-31_14-07", exp:"2025-12-31_14-07-nn-vs-cpp-mcts-parity", file:"figures/summary.png", short:"nn vs cpp mcts parity"},
  {ts:"2025-12-31_15-20", exp:"2025-12-31_15-20-cppmcts-vs-raw-policy", file:null, short:"cppmcts vs raw policy"},
  {ts:"2025-12-31_15-24", exp:"2025-12-31_15-24-nn-vs-cpp-mcts-parallel", file:"figures/summary.png", short:"nn vs cpp mcts parallel"},
  {ts:"2025-12-31_15-42", exp:"2025-12-31_15-42-mcts-vs-raw-selfplay", file:null, short:"mcts vs raw selfplay"},
  {ts:"2025-12-31_16-34", exp:"2025-12-31_16-34-cppmcts-vs-raw-selfplay", file:null, short:"cppmcts vs raw selfplay"},
  {ts:"2025-12-31_16-47", exp:"2025-12-31_16-47-cppmcts-bug-investigation", file:"figures/summary.png", short:"cppmcts bug investigation"},
  {ts:"2025-12-31_17-27", exp:"2025-12-31_17-27-mcts-sim-to-games", file:null, short:"mcts sim to games"},
  {ts:"2026-01-02_11-14", exp:"2026-01-02_11-14-mcts-sim-to-games-verification", file:null, short:"mcts sim to games verification"},
  {ts:"2026-01-02_14-23", exp:"2026-01-02_14-23-mcts-sim-paths-verification", file:null, short:"mcts sim paths verification"},
  {ts:"2026-01-02_15-45", exp:"2026-01-02_15-45-mcts-vs-raw-policy", file:"figures/results.png", short:"mcts vs raw policy"},
  {ts:"2026-01-02_17-47", exp:"2026-01-02_17-47-mcts-selfplay-data-collection", file:null, short:"mcts selfplay data collection"},
  {ts:"2026-01-02_19-21", exp:"2026-01-02_19-21-train-mcts-selfplay-data", file:"figures/accuracy_curves.png", short:"train mcts selfplay data"},
  {ts:"2026-01-02_20-03", exp:"2026-01-02_20-03-katago-score-histogram", file:"figures/score_histogram.png", short:"katago score histogram"},
  {ts:"2026-01-02_21-05", exp:"2026-01-02_21-05-train-katago-data", file:"figures/accuracy.png", short:"train katago data"},
  {ts:"2026-01-02_21-25", exp:"2026-01-02_21-25-train-mcts-augmented-katago", file:"figures/accuracy.png", short:"train mcts augmented katago"},
  {ts:"2026-01-02_22-57", exp:"2026-01-02_22-57-mcts-sims-vs-winrate", file:"figures/winrate_vs_sims.png", short:"mcts sims vs winrate"},
  {ts:"2026-01-02_23-39", exp:"2026-01-02_23-39-train-combined-gamedata", file:"figures/accuracy.png", short:"train combined gamedata"},
  {ts:"2026-01-03_12-20", exp:"2026-01-03_12-20-replay-buffer-online-training", file:null, short:"replay buffer online training"},
  {ts:"2026-01-03_15-10", exp:"2026-01-03_15-10-replay-buffer-20k-12workers", file:"figures/win_rate_vs_steps.png", short:"replay buffer 20k 12workers"},
  {ts:"2026-01-03_18-00", exp:"2026-01-03_18-00-eval-mup-10m-vs-gnugo10", file:"figures/results.png", short:"eval mup 10m vs gnugo10"},
  {ts:"2026-01-05_10-30", exp:"2026-01-05_10-30-winrate-matrix-visualization", file:"figures/winrate_matrix.png", short:"winrate matrix visualization"},
  {ts:"2026-01-05_12-16", exp:"2026-01-05_12-16-katago-final-board-analysis", file:"figures/final_board_states.png", short:"katago final board analysis"},
  {ts:"2026-01-05_12-30", exp:"2026-01-05_12-30-katago-final-board-analysis-v2", file:"figures/final_board_states.png", short:"katago final board analysis v2"},
  {ts:"2026-01-05_14-02", exp:"2026-01-05_14-02-optimal-komi-analysis", file:"figures/winrate_vs_komi.png", short:"optimal komi analysis"},
  {ts:"2026-01-05_15-12", exp:"2026-01-05_15-12-replay-buffer-with-offline-pusher", file:"figures/Screenshot from 2026-01-06 20-16-47.png", short:"replay buffer with offline pusher"},
  {ts:"2026-01-05_16-33", exp:"2026-01-05_16-33-nn-vs-katago-analysis", file:"figures/close_games_vs_katago.png", short:"nn vs katago analysis"},
  {ts:"2026-01-06_20-33", exp:"2026-01-06_20-33-checkpoint-winrate-analysis", file:"figures/policy_accuracy_vs_winrate.png", short:"checkpoint winrate analysis"},
  {ts:"2026-01-06_22-39", exp:"2026-01-06_22-39-checkpoint-winrate-reanalysis", file:"figures/loss_vs_winrate.png", short:"checkpoint winrate reanalysis"},
  {ts:"2026-01-06_23-03", exp:"2026-01-06_23-03-score-diff-heatmap", file:"figures/score_diff_heatmaps.png", short:"score diff heatmap"},
  {ts:"2026-01-08_11-00", exp:"2026-01-08_11-00-debug-random-white-upsets", file:"figures/upset_0_full_replay.png", short:"debug random white upsets"},
  {ts:"2026-01-08_13-45", exp:"2026-01-08_13-45-katago-advantage-analysis", file:"figures/katago_advantage_recommendation.png", short:"katago advantage analysis"},
  {ts:"2026-01-12_11-10", exp:"2026-01-12_11-10-eval-visualization", file:"figures/win_rate_by_color.png", short:"eval visualization"},
  {ts:"2026-01-12_21-43", exp:"2026-01-12_21-43-policy-eval-results", file:"figures/win_rate_by_color.png", short:"policy eval results"},
  {ts:"2026-01-13_12-51", exp:"2026-01-13_12-51-compare-experiments-by-color", file:"figures/win_rate_overall_comparison.png", short:"compare experiments by color"},
  {ts:"2026-01-14_10-30", exp:"2026-01-14_10-30-mcts-katago-winrate-by-color", file:"figures/win_rate_by_color.png", short:"mcts katago winrate by color"},
  {ts:"2026-01-14_11-00", exp:"2026-01-14_11-00-compare-10m-100m-flops", file:"figures/policy_accuracy_by_flops_and_step.png", short:"compare 10m 100m flops"},
  {ts:"2026-01-14_16-51", exp:"2026-01-14_16-51-10m-1petaflop-4epoch", file:"figures/accuracy_curves.png", short:"10m 1petaflop 4epoch"},
  {ts:"2026-01-14_18-20", exp:"2026-01-14_18-20-compare-10m-100m-score-advantage", file:"figures/score_advantage_combined.png", short:"compare 10m 100m score advantage"},
  {ts:"2026-01-14_23-37", exp:"2026-01-14_23-37-mcts-tune-exp01-baseline", file:"figures/win_rate_summary.png", short:"mcts tune exp01 baseline"},
  {ts:"2026-01-14_23-45", exp:"2026-01-14_23-45-mcts-tune-exp02-sim-scaling", file:"figures/combined_progression.png", short:"mcts tune exp02 sim scaling"},
  {ts:"2026-01-15_11-16", exp:"2026-01-15_11-16-mcts-baseline-20sims-120depth", file:"figures/win_rate.png", short:"mcts baseline 20sims 120depth"},
  {ts:"2026-01-15_11-47", exp:"2026-01-15_11-47-mcts-tune-exp04-policy-temperature", file:"figures/win_rate_vs_temp.png", short:"mcts tune exp04 policy temperature"},
  {ts:"2026-01-15_12-12", exp:"2026-01-15_12-12-mcts-tune-exp05-cpuct", file:"figures/win_rate_vs_cpuct.png", short:"mcts tune exp05 cpuct"},
  {ts:"2026-01-15_12-29", exp:"2026-01-15_12-29-mcts-tune-exp06-vs-gnugo", file:"figures/win_rate_vs_gnugo_level.png", short:"mcts tune exp06 vs gnugo"},
  {ts:"2026-01-15_12-42", exp:"2026-01-15_12-42-mcts-tune-exp07-color", file:"figures/win_rate_by_color.png", short:"mcts tune exp07 color"},
  {ts:"2026-01-15_12-53", exp:"2026-01-15_12-53-mcts-tune-exp08-dirichlet", file:"figures/win_rate_by_noise.png", short:"mcts tune exp08 dirichlet"},
  {ts:"2026-01-15_14-39", exp:"2026-01-15_14-39-mcts-tuning-final-report", file:null, short:"mcts tuning final report"},
  {ts:"2026-01-15_15-51", exp:"2026-01-15_15-51-mcts-vs-katago-winrate", file:"figures/win_rate_bar_chart.png", short:"mcts vs katago winrate"},
  {ts:"2026-01-15_16-29", exp:"2026-01-15_16-29-checkpoint-winrate-vs-katago", file:"figures/winrate_vs_checkpoint.png", short:"checkpoint winrate vs katago"},
  {ts:"2026-01-15_17-31", exp:"2026-01-15_17-31-mcts-tune-gnugo-baseline", file:"figures/baseline_results.png", short:"mcts tune gnugo baseline"},
  {ts:"2026-01-15_17-43", exp:"2026-01-15_17-43-mcts-tune-num-simulations", file:"figures/num_simulations_comparison.png", short:"mcts tune num simulations"},
  {ts:"2026-01-15_18-19", exp:"2026-01-15_18-19-mcts-tune-temperature", file:"figures/temperature_comparison.png", short:"mcts tune temperature"},
  {ts:"2026-01-15_18-39", exp:"2026-01-15_18-39-mcts-tune-cpuct", file:"figures/cpuct_comparison.png", short:"mcts tune cpuct"},
  {ts:"2026-01-15_18-58", exp:"2026-01-15_18-58-mcts-tune-dirichlet", file:"figures/dirichlet_comparison.png", short:"mcts tune dirichlet"},
  {ts:"2026-01-15_19-20", exp:"2026-01-15_19-20-mcts-tune-noise-weight", file:"figures/noise_weight_comparison.png", short:"mcts tune noise weight"},
  {ts:"2026-01-15_19-41", exp:"2026-01-15_19-41-mcts-validation", file:"figures/validation_results.png", short:"mcts validation"},
  {ts:"2026-01-15_20-16", exp:"2026-01-15_20-16-mcts-sims-no-noise", file:"figures/sims_no_noise.png", short:"mcts sims no noise"},
  {ts:"2026-01-16_17-17", exp:"2026-01-16_17-17-mcts-optimal-config", file:"figures/baseline_vs_optimal.png", short:"mcts optimal config"},
  {ts:"2026-01-16_17-55", exp:"2026-01-16_17-55-game-length-histograms", file:"figures/game_length_comparison.png", short:"game length histograms"},
  {ts:"2026-01-16_18-15", exp:"2026-01-16_18-15-mcts-vs-katago-blunder-analysis", file:"figures/move_heatmaps.png", short:"mcts vs katago blunder analysis"},
  {ts:"2026-01-16_19-25", exp:"2026-01-16_19-25-stronger-katago-checkpoint-eval", file:"figures/winrate_vs_checkpoint.png", short:"stronger katago checkpoint eval"},
  {ts:"2026-01-16_22-36", exp:"2026-01-16_22-36-mcts-vs-katago-analysis", file:"figures/win_rate_by_color.png", short:"mcts vs katago analysis"},
  {ts:"2026-01-17_11-03", exp:"2026-01-17_11-03-compare-passaction-training-flops", file:"figures/policy_accuracy_vs_flops.png", short:"compare passaction training flops"},
  {ts:"2026-01-17_11-30", exp:"2026-01-17_11-30-analyze-passaction-training", file:"figures/policy_accuracy_vs_flops.png", short:"analyze passaction training"},
  {ts:"2026-01-17_12-00", exp:"2026-01-17_12-00-compare-distill-model-sizes", file:"figures/metrics_vs_flops.png", short:"compare distill model sizes"},
  {ts:"2026-01-17_12-00", exp:"2026-01-17_12-00-redo-mcts", file:"figures/recovery_attempts.png", short:"redo mcts"},
  {ts:"2026-01-17_12-30", exp:"2026-01-17_12-30-redo-mcts-vs-katago", file:"figures/recovery_summary.png", short:"redo mcts vs katago"},
  {ts:"2026-01-17_13-00", exp:"2026-01-17_13-00-redo-mcts-20games", file:"figures/alternative_analysis.png", short:"redo mcts 20games"},
  {ts:"2026-01-17_13-30", exp:"2026-01-17_13-30-redo-mcts-data-collection", file:"figures/data_summary.png", short:"redo mcts data collection"},
  {ts:"2026-01-19_15-30", exp:"2026-01-19_15-30-finetune-redo-mcts", file:"figures/matchup_summary.png", short:"finetune redo mcts"},
  {ts:"2026-01-19_17-30", exp:"2026-01-19_17-30-critical-drop-shift-analysis", file:"figures/critical_drop_comparison.png", short:"critical drop shift analysis"},
  {ts:"2026-01-19_18-00", exp:"2026-01-19_18-00-iterative-recovery-training", file:"figures/win_rate_vs_base.png", short:"iterative recovery training"},
  {ts:"2026-01-20_10-42", exp:"2026-01-20_10-42-base-finetune-recovery-training", file:"figures/win_rate_vs_base.png", short:"base finetune recovery training"},
  {ts:"2026-01-20_11-58", exp:"2026-01-20_11-58-base-finetune-10-alternates", file:"figures/recovery_rate_comparison.png", short:"base finetune 10 alternates"},
  {ts:"2026-01-20_14-00", exp:"2026-01-20_14-00-update-frequency-sweep", file:"figures/final_win_rate_comparison.png", short:"update frequency sweep"},
  {ts:"2026-01-20_20-35", exp:"2026-01-20_20-35-distributed-collector-scaling", file:"figures/scaling_throughput.png", short:"distributed collector scaling"},
  {ts:"2026-01-20_20-56", exp:"2026-01-20_20-56-local-gpu-collector-scaling", file:"figures/comparison.png", short:"local gpu collector scaling"},
  {ts:"2026-01-20_21-22", exp:"2026-01-20_21-22-local-inference-server-per-collector", file:"figures/comparison.png", short:"local inference server per collector"},
  {ts:"2026-01-20_22-48", exp:"2026-01-20_22-48-shared-server-scaling-6-collectors", file:"figures/scaling_throughput.png", short:"shared server scaling 6 collectors"},
  {ts:"2026-01-20_23-10", exp:"2026-01-20_23-10-inference-server-batch-tuning", file:"figures/throughput_vs_batch_size.png", short:"inference server batch tuning"},
  {ts:"2026-01-21_13-06", exp:"2026-01-21_13-06-collector-throughput-optimization", file:"03-collector-worker-scaling/figures/collector_scaling.png", short:"collector throughput optimization"},
  {ts:"2026-01-21_23-50", exp:"2026-01-21_23-50-iterative-recovery-vs-katago", file:null, short:"iterative recovery vs katago"},
  {ts:"2026-01-22_01-19", exp:"2026-01-22_01-19-recovery-vs-gnugo", file:"mcts_comparison/figures/recovery_comparison.png", short:"recovery vs gnugo"},
  {ts:"2026-01-22_10-30", exp:"2026-01-22_10-30-recovery-v2-parallel", file:"figures/progress.png", short:"recovery v2 parallel"},
  {ts:"2026-01-22_12-45", exp:"2026-01-22_12-45-recovery-v3-katago-opponents", file:"figures/progress.png", short:"recovery v3 katago opponents"},
  {ts:"2026-01-22_17-30", exp:"2026-01-22_17-30-debug-katago-asymmetry", file:"figures/game_outcome_analysis.png", short:"debug katago asymmetry"},
  {ts:"2026-01-22_19-45", exp:"2026-01-22_19-45-train-10m-evaluate-checkpoints", file:"figures/training_metrics.png", short:"train 10m evaluate checkpoints"},
  {ts:"2026-01-25_10-25", exp:"2026-01-25_10-25-lambda-vs-sims-sweep", file:"figures/win_rate_vs_sims.png", short:"lambda vs sims sweep"},
  {ts:"2026-01-25_14-56", exp:"2026-01-25_14-56-mcts-sims-sweep-lambda0", file:"figures/win_rate_vs_sims.png", short:"mcts sims sweep lambda0"},
  {ts:"2026-01-25_15-30", exp:"2026-01-25_15-30-cpuct-sweep", file:"figures/win_rate_vs_sims_cpuct.png", short:"cpuct sweep"},
  {ts:"2026-01-25_18-35", exp:"2026-01-25_18-35-train-10m-concurrent-eval", file:"figures/checkpoint_win_rates_ray.png", short:"train 10m concurrent eval"},
  {ts:"2026-01-25_19-24", exp:"2026-01-25_19-24-checkpoint-pair-eval", file:"figures/advantage_matrix.png", short:"checkpoint pair eval"},
  {ts:"2026-01-26_00-30", exp:"2026-01-26_00-30-train-10m-100m-dataset-v3", file:"figures/model_comparison_flops.png", short:"train 10m 100m dataset v3"},
  {ts:"2026-01-26_10-00", exp:"2026-01-26_10-00-eval-10m-100m-dataset-v3", file:"figures/combined_scaling.png", short:"eval 10m 100m dataset v3"},
  {ts:"2026-01-26_12-00", exp:"2026-01-26_12-00-scaling-law-500m-1b", file:"figures/accuracy_vs_flops.png", short:"scaling law 500m 1b"},
  {ts:"2026-01-27_00-30", exp:"2026-01-27_00-30-10m-checkpoint-pair-eval", file:"figures/advantage_matrix.png", short:"10m checkpoint pair eval"},
  {ts:"2026-02-01_12-00", exp:"2026-02-01_12-00-eval-scaling-law-500m-1b", file:"figures/combined_flops_scaling.png", short:"eval scaling law 500m 1b"},
  {ts:"2026-02-02_17-09", exp:"2026-02-02_17-09-datasetv4-scaling-laws", file:"figures/accuracy_vs_flops.png", short:"datasetv4 scaling laws"},
  {ts:"2026-02-03_12-00", exp:"2026-02-03_12-00-eval-datasetv4-scaling-laws", file:"figures/combined_flops_scaling.png", short:"eval datasetv4 scaling laws"},
  {ts:"2026-02-03_18-00", exp:"2026-02-03_18-00-datasetv4-checkpoint-pairs", file:"figures/cross_size_comparison.png", short:"datasetv4 checkpoint pairs"},
  {ts:"2026-02-05_20-45", exp:"2026-02-05_20-45-datasetv5-10m-training", file:"figures/loss_curves.png", short:"datasetv5 10m training"},
  {ts:"2026-02-05_23-30", exp:"2026-02-05_23-30-hyperparam-search-v5", file:"figures/batch_00_val_loss.png", short:"hyperparam search v5"},
  {ts:"2026-02-06_09-15", exp:"2026-02-06_09-15-hyperparam-refinement-v7", file:"batch-01-architecture/figures/train_val_comparison.png", short:"hyperparam refinement v7"},
  {ts:"2026-02-06_09-30", exp:"2026-02-06_09-30-hyperparam-refinement-v8", file:"figures/architecture_comparison.png", short:"hyperparam refinement v8"},
  {ts:"2026-02-06_09-30", exp:"2026-02-06_09-30-iterative-self-play-training", file:"figures/combined_loss_curves.png", short:"iterative self play training"},
  {ts:"2026-02-06_19-30", exp:"2026-02-06_19-30-iterative-self-play-7M-d4", file:"figures/combined_loss_curves.png", short:"iterative self play 7M d4"},
  {ts:"2026-02-07_18-00", exp:"2026-02-07_18-00-eval-selfplay-7M-vs-katago", file:"figures/win_rate_vs_round.png", short:"eval selfplay 7M vs katago"},
  {ts:"2026-02-07_20-00", exp:"2026-02-07_20-00-iterative-selfplay-7M-earlystop", file:"figures/val_loss_comparison.png", short:"iterative selfplay 7M earlystop"},
  {ts:"2026-02-08_09-00", exp:"2026-02-08_09-00-eval-iterative-selfplay-earlystop-vs-katago", file:"figures/win_rate_vs_round.png", short:"eval iterative selfplay earlystop vs katago"},
  {ts:"2026-02-08_09-00", exp:"2026-02-08_09-00-scratch-each-round-7M", file:"figures/win_rate_vs_round.png", short:"scratch each round 7M"},
  {ts:"2026-02-08_18-30", exp:"2026-02-08_18-30-strategy-landscape-viz", file:"figures/katago_winrates.png", short:"strategy landscape viz"},
  {ts:"2026-02-08_19-30", exp:"2026-02-08_19-30-iterative-selfplay-allckpt-eval", file:"figures/val_loss_comparison.png", short:"iterative selfplay allckpt eval"},
  {ts:"2026-02-09_13-00", exp:"2026-02-09_13-00-selfplay-optimization-series", file:null, short:"selfplay optimization series"},
  {ts:"2026-02-09_16-45", exp:"2026-02-09_16-45-allckpt-strategy-diversity", file:"figures/behavioral_embeddings_by_epoch.png", short:"allckpt strategy diversity"},
  {ts:"2026-02-09_17-30", exp:"2026-02-09_17-30-iterative-selfplay-v5v6-dataset", file:"figures/val_loss_comparison.png", short:"iterative selfplay v5v6 dataset"},
  {ts:"2026-02-10_09-00", exp:"2026-02-10_09-00-iterative-selfplay-weighted-continuous", file:"figures/val_loss_comparison.png", short:"iterative selfplay weighted continuous"},
  {ts:"2026-02-10_11-00", exp:"2026-02-10_11-00-iterative-selfplay-uniform-continuous", file:"figures/val_loss_comparison.png", short:"iterative selfplay uniform continuous"},
  {ts:"2026-02-10_17-50", exp:"2026-02-10_17-50-train-datasetv5v7-10m-6epochs", file:"figures/win_rate_curve.png", short:"train datasetv5v7 10m 6epochs"},
  {ts:"2026-02-13_18-42", exp:"2026-02-13_18-42-optimize-val-loss", file:null, short:"optimize val loss"},
  {ts:"2026-03-29_04-00", exp:"2026-03-29_04-00-optimize-valloss", file:null, short:"optimize valloss"},
  {ts:"2026-04-03_12-00", exp:"2026-04-03_12-00-optimize-katago-valloss", file:null, short:"optimize katago valloss"},
  {ts:"2026-04-04_00-00", exp:"2026-04-04_00-00-optimize-v10-valloss", file:"figures/improvement_waterfall.png", short:"optimize v10 valloss"},
  {ts:"2026-04-06_00-00", exp:"2026-04-06_00-00-optimize-v11-valloss", file:null, short:"optimize v11 valloss"},
  {ts:"2026-04-06_12-00", exp:"2026-04-06_12-00-optimize-transformer-valloss", file:null, short:"optimize transformer valloss"},
  {ts:"2026-04-07_00-00", exp:"2026-04-07_00-00-optimize-v12-valloss", file:null, short:"optimize v12 valloss"},
  {ts:"2026-04-07_09-00", exp:"2026-04-07_09-00-tts-scaling-tuned", file:null, short:"tts scaling tuned"},
  {ts:"2026-04-07_09-00", exp:"2026-04-07_09-00-tts-scaling-tuned-c_puct0.5", file:null, short:"tts scaling tuned c_puct0.5"},
  {ts:"2026-04-07_10-00", exp:"2026-04-07_10-00-convnet-vs-transformer", file:"figures/running_best_comparison.png", short:"convnet vs transformer"},
  {ts:"2026-04-07_17-30", exp:"2026-04-07_17-30-optimize-v12-valloss", file:null, short:"optimize v12 valloss"},
  {ts:"2026-04-08_06-45", exp:"2026-04-08_06-45-learngo-v0", file:null, short:"learngo v0"},
  {ts:"2026-04-09_00-58", exp:"2026-04-09_00-58-learngo-v1", file:null, short:"learngo v1"},
  {ts:"2026-04-10_15-55", exp:"2026-04-10_15-55-optimize-valloss", file:null, short:"optimize valloss"},
  {ts:"2026-04-10_23-47", exp:"2026-04-10_23-47-learngo-v2", file:null, short:"learngo v2"},
  {ts:"2026-04-11_21-05", exp:"2026-04-11_21-05-throughput-opt", file:null, short:"throughput opt"},
  {ts:"2026-04-12_00-08", exp:"2026-04-12_00-08-learngo-v3", file:null, short:"learngo v3"},
  {ts:"2026-04-12_04-49", exp:"2026-04-12_04-49-playout-cap-randomization", file:null, short:"playout cap randomization"},
  {ts:"2026-04-12_19-51", exp:"2026-04-12_19-51-cpp-mcts-throughput", file:null, short:"cpp mcts throughput"},
  {ts:"2026-04-12_23-32", exp:"2026-04-12_23-32-numsim-sweep", file:null, short:"numsim sweep"},
  {ts:"2026-04-13_00-43", exp:"2026-04-13_00-43-numworkers-knee", file:null, short:"numworkers knee"},
  {ts:"2026-04-13_00-51", exp:"2026-04-13_00-51-learngo-local-teacher", file:null, short:"learngo local teacher"},
  {ts:"2026-04-13_08-00", exp:"2026-04-13_08-00-learngo-local-teacher-v3", file:null, short:"learngo local teacher v3"},
  {ts:"2026-04-13_16-21", exp:"2026-04-13_16-21-learngo-local-teacher-v4", file:null, short:"learngo local teacher v4"},
  {ts:"2026-04-13_17-35", exp:"2026-04-13_17-35-learngo-ssh-cluster-v5", file:null, short:"learngo ssh cluster v5"},
  {ts:"2026-04-15_02-11", exp:"2026-04-15_02-11-iter6-recovery", file:null, short:"iter6 recovery"},
  {ts:"2026-04-15_07-21", exp:"2026-04-15_07-21-learngo-ssh-cluster-v6", file:null, short:"learngo ssh cluster v6"},
  {ts:"2026-04-15_08-00", exp:"2026-04-15_08-00-learngo-decen-collect-v7", file:null, short:"learngo decen collect v7"},
  {ts:"2026-04-17_04-25", exp:"2026-04-17_04-25-ben-replay-iter6", file:null, short:"ben replay iter6"},
  {ts:"2026-04-17_04-40", exp:"2026-04-17_04-40-throughput-tuning", file:null, short:"throughput tuning"},
  {ts:"2026-04-17_05-35", exp:"2026-04-17_05-35-autoresearch-valloss", file:null, short:"autoresearch valloss"},
  {ts:"2026-04-17_17-08", exp:"2026-04-17_17-08-autoresearch-transformer-valloss", file:null, short:"autoresearch transformer valloss"},
  {ts:"2026-04-18_06-16", exp:"2026-04-18_06-16-learngo-fast", file:null, short:"learngo fast"},
  {ts:"2026-04-19_05-42", exp:"2026-04-19_05-42-mcts-label-throughput", file:null, short:"mcts label throughput"},
  {ts:"2026-04-19_06-35", exp:"2026-04-19_06-35-mcts-self-distill", file:null, short:"mcts self distill"},
  {ts:"2026-04-20_08-30", exp:"2026-04-20_08-30-mcts-label-2048", file:null, short:"mcts label 2048"},
  {ts:"2026-04-20_12-30", exp:"2026-04-20_12-30-mcts-rootq-stabilization", file:null, short:"mcts rootq stabilization"},
  {ts:"2026-04-20_23-35", exp:"2026-04-20_23-35-learngo-decen-collect-19x19-v0", file:null, short:"learngo decen collect 19x19 v0"},
  {ts:"2026-04-21_21-17", exp:"2026-04-21_21-17-19x19-arch-search", file:null, short:"19x19 arch search"},
  {ts:"2026-04-21_22-02", exp:"2026-04-21_22-02-mcts-rootq-stabilization-100k", file:null, short:"mcts rootq stabilization 100k"},
  {ts:"2026-04-22_00-15", exp:"2026-04-22_00-15-size-invariant-resnet", file:null, short:"size invariant resnet"},
  {ts:"2026-04-22_07-56", exp:"2026-04-22_07-56-19x19-arch-search", file:null, short:"19x19 arch search"},
  {ts:"2026-04-22_12-11", exp:"2026-04-22_12-11-learngo-19x19-9x9-v0", file:null, short:"learngo 19x19 9x9 v0"},
  {ts:"2026-04-22_12-22", exp:"2026-04-22_12-22-learngo-refine-onpolicy", file:null, short:"learngo refine onpolicy"},
  {ts:"2026-04-23_00-01", exp:"2026-04-23_00-01-offline-vs-onpolicy-postmortem", file:null, short:"offline vs onpolicy postmortem"},
  {ts:"2026-04-23_00-02", exp:"2026-04-23_00-02-learngo-refine-onpolicy-balanced-value", file:null, short:"learngo refine onpolicy balanced value"},
  {ts:"2026-04-24_09-12", exp:"2026-04-24_09-12-eval-9x9-vs-19x19-on-9x9", file:"figures/win_rate.png", short:"eval 9x9 vs 19x19 on 9x9"},
  {ts:"2026-04-25_14-16", exp:"2026-04-25_14-16-mcts-throughput-bench", file:"figures/moves_per_sec.png", short:"mcts throughput bench"},
  {ts:"2026-04-25_20-09", exp:"2026-04-25_20-09-web-mcts-cluster-bench", file:null, short:"web mcts cluster bench"},
];
const N_EXP = EXP.length;
const N_WITH_FIG = EXP.filter(e => e.file).length;

function parseExpTime(s) {
  const [d, t] = s.split('_');
  const [y, m, day] = d.split('-').map(Number);
  const [h, mi] = t.split('-').map(Number);
  return Date.UTC(y, m - 1, day, h, mi);
}

function P4_ExperimentTimeline() {
  const { localTime: lt } = useSprite();

  // ── Plot geometry ────────────────────────────────────────────────────────
  const PLOT_X = 220, PLOT_Y = 260, PLOT_W = 460, PLOT_H = 320;
  const PAD_L = 50, PAD_R = 18, PAD_T = 36, PAD_B = 36;
  const PA_L = PLOT_X + PAD_L;
  const PA_R = PLOT_X + PLOT_W - PAD_R;
  const PA_T = PLOT_Y + PAD_T;
  const PA_B = PLOT_Y + PLOT_H - PAD_B;

  const points = React.useMemo(() => {
    const ts = EXP.map(e => parseExpTime(e.ts));
    const t0 = ts[0], tEnd = ts[ts.length - 1];
    return EXP.map((e, i) => ({
      ...e, ts_ms: ts[i],
      x: PA_L + (ts[i] - t0) / (tEnd - t0) * (PA_R - PA_L),
      y: PA_B - (i + 1) / N_EXP * (PA_B - PA_T),
    }));
  }, []);

  // Animation: dots fade in over ~12 s starting at lt = 1.0
  const DOT_T0 = 1.0;
  const DOT_DT = 12.0 / N_EXP;
  const visibleDots = clamp(Math.floor((lt - DOT_T0) / DOT_DT) + 1, 0, N_EXP);

  // Hover state
  const [hovered, setHovered] = React.useState(null);
  const latestIdx = Math.max(0, Math.min(visibleDots - 1, N_EXP - 1));

  // Auto-cycle figure pane: last figure-bearing index ≤ latestIdx.
  // If we haven't crossed any figure-bearing dot yet, fall back to the first
  // figure-bearing experiment so the pane is never empty.
  const autoFigIdx = React.useMemo(() => {
    for (let i = latestIdx; i >= 0; i--) {
      if (points[i].file) return i;
    }
    return points.findIndex(p => p.file);
  }, [latestIdx, points]);

  // Only figure-bearing dots are hoverable, so `hovered` (when set) always
  // points at a figure-bearing entry.
  const selectedIdx = hovered != null ? hovered : latestIdx;
  const figureIdx = hovered != null ? hovered : autoFigIdx;

  // Flash when a figure-bearing dot was just revealed and user isn't hovering.
  const lastDotRevealAt = DOT_T0 + latestIdx * DOT_DT;
  const sinceReveal = lt - lastDotRevealAt;
  const flash = (hovered == null && points[latestIdx]?.file
                 && sinceReveal >= 0 && sinceReveal < 0.5)
    ? (1 - sinceReveal / 0.5)
    : 0;

  // ── China-travel gap ─────────────────────────────────────────────────────
  const gapStartIdx = points.findIndex(p => p.ts_ms >= parseExpTime('2026-02-11_00-00'));
  const gapEndIdx   = points.findIndex(p => p.ts_ms >= parseExpTime('2026-04-01_00-00'));
  const gapLeftX  = gapStartIdx >= 0 ? points[gapStartIdx].x : PA_L;
  const gapRightX = gapEndIdx   >= 0 ? points[gapEndIdx  ].x : PA_R;
  const gapMidX = (gapLeftX + gapRightX) / 2;

  const headerOp = Easing.easeOutCubic(clamp(lt / 0.6, 0, 1));
  const plotOp   = Easing.easeOutCubic(clamp((lt - 0.6) / 0.6, 0, 1));
  const figOp    = Easing.easeOutCubic(clamp((lt - 1.0) / 0.6, 0, 1));
  // The gap annotation appears once dots have crossed into Feb territory.
  const gapAnnoOp = Easing.easeOutCubic(clamp((lt - 8.5) / 0.7, 0, 1));

  // X-axis month tick marks
  const months = [
    { ms: parseExpTime('2025-12-15_00-00'), label: 'Dec' },
    { ms: parseExpTime('2026-01-01_00-00'), label: 'Jan' },
    { ms: parseExpTime('2026-02-01_00-00'), label: 'Feb' },
    { ms: parseExpTime('2026-03-01_00-00'), label: 'Mar' },
    { ms: parseExpTime('2026-04-01_00-00'), label: 'Apr' },
  ];
  const t0Ms = points[0].ts_ms;
  const tEndMs = points[points.length - 1].ts_ms;
  const xForMs = (ms) => PA_L + (ms - t0Ms) / (tEndMs - t0Ms) * (PA_R - PA_L);

  return (
    <>
      <div style={{ opacity: headerOp }}>
        <SlideHeader
          num="11"
          title="Lessons from a four-month research log"
          maxWidth={1040}
          subtitle={<>
            We ran <strong>{N_EXP}</strong> experiments over four months, attempting to build a Go AI from scratch. Here were the most important lessons we learned:
          </>}
        />
      </div>

      {/* LEFT eyebrow */}
      <div style={{
        position: 'absolute', left: PLOT_X + PAD_L - 4, top: PLOT_Y + 4,
        opacity: plotOp,
        fontFamily: 'var(--mono)', fontSize: 10.5,
        letterSpacing: '0.14em', textTransform: 'uppercase',
        color: 'var(--ink-soft)',
      }}>
        Cumulative experiments — {visibleDots} / {N_EXP}
      </div>

      {/* Plot SVG (interactive — gets mouse events) */}
      <svg style={{
        position: 'absolute', inset: 0,
        opacity: plotOp,
      }} width="100%" height="100%">
        {/* Axis frame */}
        <line x1={PA_L} y1={PA_B} x2={PA_R} y2={PA_B}
              stroke="var(--ink-soft)" strokeWidth={0.8} opacity={0.6} />
        <line x1={PA_L} y1={PA_T} x2={PA_L} y2={PA_B}
              stroke="var(--ink-soft)" strokeWidth={0.8} opacity={0.6} />

        {/* Y gridlines + labels */}
        {[0, 50, 100, 150, 200].map(v => {
          const y = PA_B - (v / N_EXP) * (PA_B - PA_T);
          return (
            <g key={`y${v}`}>
              <line x1={PA_L} y1={y} x2={PA_R} y2={y}
                    stroke="var(--ink-soft)" strokeWidth={0.5}
                    strokeDasharray="2 4" opacity={v === 0 ? 0 : 0.18} />
              <text x={PA_L - 8} y={y + 3.5} textAnchor="end"
                    fontFamily="var(--mono)" fontSize={10}
                    fill="var(--ink-soft)" fontVariantNumeric="tabular-nums">
                {v}
              </text>
            </g>
          );
        })}

        {/* X month gridlines + labels */}
        {months.map((m, i) => {
          const x = xForMs(m.ms);
          if (x < PA_L - 2 || x > PA_R + 2) return null;
          return (
            <g key={`m${i}`}>
              <line x1={x} y1={PA_T} x2={x} y2={PA_B}
                    stroke="var(--ink-soft)" strokeWidth={0.5}
                    strokeDasharray="2 4" opacity={0.15} />
              <text x={x} y={PA_B + 16} textAnchor="middle"
                    fontFamily="var(--mono)" fontSize={10}
                    fill="var(--ink-soft)">
                {m.label}
              </text>
            </g>
          );
        })}

        {/* Step polyline through visible dots */}
        {visibleDots > 1 && (() => {
          const pts = [`${PA_L},${PA_B}`];
          for (let i = 0; i < Math.min(visibleDots, points.length); i++) {
            const p = points[i];
            const prev = i === 0 ? { x: PA_L, y: PA_B } : points[i - 1];
            pts.push(`${p.x},${prev.y}`);
            pts.push(`${p.x},${p.y}`);
          }
          return (
            <polyline points={pts.join(' ')} fill="none"
                      stroke="var(--accent-mcts)" strokeWidth={1.4} opacity={0.85}
                      strokeLinejoin="round" strokeLinecap="round" />
          );
        })()}

        {/* China-travel gap annotation */}
        <g opacity={gapAnnoOp}>
          <rect x={gapLeftX} y={PA_T} width={Math.max(0, gapRightX - gapLeftX)} height={PA_B - PA_T}
                fill="var(--accent-mcts-bg)" opacity={0.55} />
          <text x={gapMidX} y={(PA_T + PA_B) / 2 + 4}
                textAnchor="middle"
                fontFamily="var(--mono)" fontSize={11}
                letterSpacing="0.08em"
                fill="var(--accent-mcts)" fontWeight={600}>
            Eric traveling
          </text>
        </g>

        {/* Dots — each is its own hover target. Smaller radius to handle density. */}
        {points.slice(0, visibleDots).map((p, i) => {
          const since = lt - (DOT_T0 + i * DOT_DT);
          const fade = Easing.easeOutCubic(clamp(since / 0.25, 0, 1));
          const isLatest = i === latestIdx;
          const isHovered = hovered === i;
          const isSelected = selectedIdx === i;
          const hasFig = !!p.file;
          const baseR = hasFig ? 2.6 : 1.8;
          const r = baseR + (isSelected ? 1.8 : 0);
          return (
            <g key={`d${i}`} opacity={fade}
               style={{ cursor: hasFig ? 'pointer' : 'default' }}
               onMouseEnter={hasFig ? () => setHovered(i) : undefined}
               onMouseLeave={hasFig ? () => setHovered(prev => (prev === i ? null : prev)) : undefined}>
              {/* Hit target — only figure-bearing dots get an enlarged target. */}
              {hasFig && <circle cx={p.x} cy={p.y} r={7} fill="transparent" />}
              {/* Latest-dot pulse ring (only when nothing is being hovered) */}
              {isLatest && hovered == null && hasFig && (
                <circle cx={p.x} cy={p.y} r={r + 5}
                        fill="none" stroke="var(--accent-mcts)"
                        strokeWidth={1.0} opacity={0.4 * fade} />
              )}
              {/* Hover halo */}
              {isHovered && (
                <circle cx={p.x} cy={p.y} r={r + 7}
                        fill="none" stroke="var(--accent-mcts)"
                        strokeWidth={1.2} opacity={0.55} />
              )}
              <circle cx={p.x} cy={p.y} r={r}
                      fill={hasFig ? 'var(--accent-mcts)' : 'rgba(31,26,20,0.40)'}
                      stroke={isSelected ? 'var(--bg)' : 'none'}
                      strokeWidth={isSelected ? 1.0 : 0} />
            </g>
          );
        })}

        {/* X-axis caption */}
        <text x={(PA_L + PA_R) / 2} y={PA_B + 32} textAnchor="middle"
              fontFamily="var(--mono)" fontSize={10}
              letterSpacing="0.08em"
              fill="var(--ink-soft)">
          date · 2025–26
        </text>

        {/* Y-axis caption */}
        <text x={PA_L - 32} y={(PA_T + PA_B) / 2} textAnchor="middle"
              fontFamily="var(--mono)" fontSize={10}
              letterSpacing="0.08em"
              fill="var(--ink-soft)"
              transform={`rotate(-90 ${PA_L - 32} ${(PA_T + PA_B) / 2})`}>
          experiments
        </text>
      </svg>

      {/* RIGHT: featured figure pane (driven by figureIdx; text-card fallback for hovered no-figure dots) */}
      <div style={{
        position: 'absolute', left: 720, top: 245,
        width: 510, height: 350,
        opacity: figOp,
      }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 10.5,
          letterSpacing: '0.12em', textTransform: 'uppercase',
          color: hovered != null ? 'var(--accent-mcts)' : 'var(--ink-soft)',
          marginBottom: 6,
          transition: 'color 220ms',
        }}>
          {hovered != null ? 'Hovered' : 'Featured'} · {EXP[selectedIdx].ts}
        </div>

        <div style={{
          position: 'relative',
          width: '100%', height: 290,
          background: 'rgba(31,26,20,0.04)',
          border: `1px solid ${flash > 0 ? 'var(--accent-mcts)' : 'rgba(31,26,20,0.10)'}`,
          borderRadius: 6,
          overflow: 'hidden',
          boxShadow: flash > 0
            ? `0 0 0 ${1 + flash * 3}px rgba(176, 79, 50, ${0.18 * flash})`
            : 'none',
          transition: 'box-shadow 200ms ease, border-color 200ms ease',
        }}>
          {/* Stacked figure images — only the figure-bearing experiments. */}
          {EXP.map((f, i) => f.file ? (
            <img key={i}
              src={`/alphago-tutorial/experiments/${f.exp}/${f.file}`}
              alt={f.short}
              loading={Math.abs(i - figureIdx) > 8 ? 'lazy' : 'eager'}
              style={{
                position: 'absolute', inset: 8,
                width: 'calc(100% - 16px)', height: 'calc(100% - 16px)',
                objectFit: 'contain',
                opacity: i === figureIdx ? 1 : 0,
                transition: 'opacity 240ms ease',
                pointerEvents: 'none',
              }} />
          ) : null)}
        </div>

        <div style={{
          marginTop: 8,
          fontFamily: 'var(--serif)', fontSize: 13,
          color: 'var(--ink)', lineHeight: 1.4,
          fontStyle: 'italic',
          minHeight: 30,
        }}>
          {EXP[figureIdx]?.short}
        </div>
      </div>
    </>
  );
}

Object.assign(window, { P4_ExperimentTimeline });
