# Related work notes: "Pixels Without Geometry? Auditing the 3D Consistency of Video World Models with Structure-from-Motion"

Compiled 2026-10-11. Every entry below was checked against its arXiv record (title, authors, submission date, ID) via the arXiv API. Venues come from the arXiv comment or journal-ref field, Crossref (IEEE/Springer proceedings), Semantic Scholar, or the ICLR 2026 listing. Where none of these confirmed a venue, the entry is listed as arXiv only. Text marked "(abstract)" is based only on the abstract; text marked "(full text checked)" means I read the relevant section.

---

## A. Novelty assessment (closest works)

1. **Li et al. 2024, "Sora Generates Videos with Stunning Geometrical Consistency" (2402.17403)** (full text checked). This paper runs SIFT matching with fundamental-matrix RANSAC on frame pairs, reporting inlier counts, keep-ratio and epipolar error, then COLMAP plus 3DGS. It covers 6 to 10 text-to-video clips from Sora, Gen-2 and Pika. It is the closest precedent for our two-view and SfM metrics. It uses no controls, no real reference, and no robotics data, and its keep-ratio is confounded by how much the camera moves. Our slowed-down and blurred controls target exactly that confound.
2. **RoboPhys-3D (2608.28718).** The closest robotics benchmark. It passes generated and ground-truth robot videos through the same reconstruction pipeline (VGGT, COLMAP and 3DGS, plus DROID-SLAM for the camera path) and evaluates Wan 2.2, CogVideoX 1.5, Cosmos 3 and RoboDreamer. It differs from us in three ways: the data is simulated (RoboTwin 2.0), it has no perturbed-real calibration ladder, and no wrist camera is mentioned.
3. **GeoT2V-Bench (2606.24829).** Tests whether camera-prompted T2V clips can be reconstructed as a single rigid scene, using VGGT poses with static-versus-deformable Gaussian fits. It is not robotic, not action-conditioned and has no real continuation.
4. **MEt3R (2501.06336) and WorldScore (2504.00983).** The standard reconstruction-based consistency metrics: DUSt3R warping for MEt3R, DROID-SLAM reprojection for WorldScore.
5. **DepthWorld (CoRL 2026, 2610.08780)** (full text checked). Makes our motivating claim about DROID world models and uses Ctrl-World as its baseline. It measures cross-view reprojection with ground-truth depth, not SfM consistency of rollouts.

**Positioning.** Our paper is the first *calibrated* SfM, epipolar and VGGT audit on *real* DROID wrist-camera clips that compares general I2V models against an *action-conditioned* DROID world model and scores the camera path against the true path. Claim the control ladder (non-rigid warps, unrelated clips, motion- and blur-matched real clips) as the main methodological contribution. Do not claim "first to use SfM on generated video."

---

## B. Verified papers by group

### Group 1: Geometric / 3D / multi-view consistency evaluation of generated video

**Closest-work details (novelty threats)**

- **Sora Generates Videos with Stunning Geometrical Consistency.** Li, X. et al., arXiv 2024, **2402.17403**. (full text checked) Takes Sora clips, regenerates Pika and Gen-2 videos from the first frame, then computes SIFT, RANSAC-F, num_pts, num_inliers_F, keep_ratio and symmetric epipolar error (RMSE/MAE) at 30-frame intervals, plus COLMAP SfM and 3DGS visualizations. *Overlap:* the same family of two-view and SfM metrics. *Difference:* we add a real-continuation reference, a calibration-control ladder, VGGT camera-path error against ground truth, robot wrist-camera data and an action-conditioned model. Their metric, which favours "more matches", would reward a frozen camera. We explicitly control for that.
- **MEt3R: Measuring Multi-View Consistency in Generated Images.** Asim, M. et al., CVPR 2025, **2501.06336**. A DUSt3R-based pairwise consistency metric that warps DINO features between views. It is a feed-forward-reconstruction metric we should report or compare against. It is not robot-specific and has no camera-path reference.
- **WorldScore: A Unified Evaluation Benchmark for World Generation.** Duan, H. et al., ICCV 2025, **2504.00983**. A world-generation benchmark with camera-control and 3D-consistency metrics (3D consistency via DROID-SLAM reprojection error, as reported in secondary sources). It is a general-scene benchmark with no robotics data and no perturbation controls.
- **GeoT2V-Bench: Benchmarking 3D Consistency in Text-to-Video Models via 3D Reconstruction.** Fan, C. and Favaro, P., arXiv 2026, **2606.24829**. (abstract) Asks the same question as ours: can generated frames be explained as one static 3D scene? It answers with VGGT-style poses, DeformableGS, a static MedianGS proxy and flow agreement, over 12 T2V configurations. *Difference:* text-to-video with prompted camera paths, no ground-truth continuation, no robotics, and no action-conditioned models.
- **GeCo: Evaluating Geometric Consistency for Video Generation via Motion and Structure.** Gu, L. et al., arXiv 2025, **2512.22274**. Dense deformation and occlusion-consistency maps for static scenes, built from residual motion and depth priors. It is a complementary dense metric that we could cite as an alternative to SfM.
- **Measuring 3D Spatial Geometric Consistency in Dynamic Video Generation (SGC).** Dou, W. et al., ECCV 2026, **2603.19048**. Measures divergence among camera poses estimated from separate static sub-regions. It handles dynamic foregrounds, which matters because our robot arm moves. Secondary sources say it notes that COLMAP often fails on generated video.
- **Quantitative Video World Model Evaluation for Geometric-Consistency (PDI-Bench).** Wu, J. et al., arXiv 2026, **2605.15185**. Computes object-centric projective-geometry residuals (scale-depth, 3D motion, rigidity) using SAM 2, MegaSaM and CoTracker3. It audits geometry of "video world models" but works per object and not on robotics.
- **WorldOlympiad: Can Your World Model Survive a Triathlon?** Zhao, Y. et al., arXiv 2026, **2606.11129**. (partial full text) Its geometry track reconstructs generated videos with Depth Anything 3 Gaussians and includes camera-trajectory alignment against the reference video's camera. It covers a robotics scenario (RoboCOIN; Cosmos-Predict-2.5, WoW). This partially overlaps with our camera-path error, but it uses no SfM or epipolar metrics, no DROID or Ctrl-World, and an MLLM judge for the reconstructions.

**Other evaluation and benchmark papers**

- **4DWorldBench.** Lu, Y. et al., arXiv 2025, **2511.19836**. A 3D/4D world-generation benchmark whose 4D consistency dimension includes reprojection-based geometric consistency.
- **WBench.** Ying, K. et al., arXiv 2026, **2605.25874**. An interactive world-model benchmark with consistency sub-metrics (RoboPhys-3D says these include reconstruction- and camera-trajectory-based self-consistency).
- **JOG3R: Towards 3D-Consistent Video Generators.** Huang, C.-H. P. et al., arXiv 2025, **2501.01409**. Finds that OpenSora frames "are not truly 3D-consistent" when probed with SfM/DUSt3R-style pose estimation. This is early evidence for our hypothesis.
- **Long-Term Photometric Consistent Novel View Synthesis with Diffusion Models.** Yu, J. J. et al., ICCV 2023, **2304.10700**. Introduces the thresholded symmetric epipolar distance (TSED) consistency metric. It is the closest prior work to our Sampson-error metric (from memory of the paper; the abstract does not name TSED, so check before citing for that detail).
- **Controlling Space and Time with Diffusion Models (4DiM).** Watson, D. et al., ICLR 2025, **2407.07860**. A pose-conditioned generator evaluated with epipolar and pose-based metrics. Cite for the precedent of using SfM and epipolar metrics on generated views (from memory; abstract-level verification only).
- **CameraCtrl.** He, H. et al., arXiv 2024, **2404.02101**. (full text checked) Runs COLMAP on generated videos to compute RotErr and TransErr against the ground-truth camera path. This is the direct precedent for our camera-path error.
- **MotionCtrl.** Wang, Z. et al., SIGGRAPH 2024, **2312.03641**. A camera- and object-motion-controllable generator. It is the standard baseline in camera-control evaluations.
- **CamCo.** Xu, D. et al., arXiv 2024, **2406.02509**. (full text checked) Reports the COLMAP error rate (failure to register all frames), the number of reconstructed points and summed rotation/translation error. This is the direct precedent for our registration-rate metric.
- **AC3D.** Bahmani, S. et al., CVPR 2025, **2411.18673**. Analyses 3D camera control in video DiTs. Related camera-control evaluation.
- **VD3D.** Bahmani, S. et al., ICLR 2025, **2407.12781**. Camera control for large video transformers. Same evaluation tradition.
- **CameraCtrl II.** He, H. et al., arXiv 2025, **2503.10592**. Camera-controlled dynamic scene exploration.
- **VBench.** Huang, Z. et al., CVPR 2024, **2311.17982**. The standard perceptual benchmark. Its consistency dimensions are feature-based, not geometric, which motivates our audit.
- **VBench-2.0.** Zheng, D. et al., arXiv 2025, **2503.21755**. Intrinsic faithfulness, including physics and commonsense. Still not SfM-based.
- **WorldModelBench.** Li, D. et al., NeurIPS 2025, **2502.20694**. Judges video models as world models with human and VLM judges, including a robotics domain. It is judgment-based rather than geometric.
- **PhyGenBench (Towards World Simulator...).** Meng, F. et al., arXiv 2024, **2410.05363**. A physical-commonsense benchmark.
- **Physics-IQ (Do generative video models understand physical principles?).** Motamed, S. et al., arXiv 2025, **2501.09038**. Real-continuation comparison for physics. This is methodologically close to our "real continuation" reference.
- **VideoPhy.** Bansal, H. et al., ICLR 2025, **2406.03520**. Physical commonsense evaluation.
- **VideoPhy-2.** Bansal, H. et al., arXiv 2025, **2503.06800**. Action-centric physical commonsense.
- **How Far is Video Generation from World Model: A Physical Law Perspective.** Kang, B. et al., ICML 2025, **2411.02385**. A "world-simulator" critique showing that models fail to generalise physical laws out of distribution.
- **Is Sora a World Simulator? A Comprehensive Survey...** Zhu, Z. et al., arXiv 2024, **2405.03520**. The survey behind the "Sora as world simulator" framing.
- **Video Models are Zero-Shot Learners and Reasoners.** Wiedemer, T. et al., arXiv 2025, **2509.20328**. Representative of the claim that video models carry world knowledge. Useful as a counterpoint.

### Group 2: Video world models for robotics

- **Ctrl-World.** Guo, Y. et al., ICLR 2026, **2510.10125**. Our action-conditioned model: multi-view, trained on DROID (95k trajectories), with pose-conditioned memory.
- **DepthWorld: 3D World Model for Robot Manipulation.** Bardhan, J. et al., CoRL 2026, **2610.08780**. (full text checked) States that RGB world models "do not compose into a consistent 3D world". It builds DROID-3D (calibrated depth and extrinsics) and trains an SVD-based RGB-D world model against a Ctrl-World baseline. Its geometric checks are ground-truth-depth cross-view reprojection and depth accuracy, not SfM, epipolar or VGGT pose error on rollouts. It is complementary (a fix, where we provide an audit), and its DROID-3D could supply ground-truth geometry for our future work.
- **Mem-World.** Zheng, Z. et al., CoRL 2026, **2606.18960**. A wrist-view-centred surfel memory world model on DROID that outperforms Ctrl-World for policy evaluation. It targets wrist-camera persistence, which is related to the detail loss we observe in Ctrl-World.
- **RoGSW4RLD.** Kim, J. H. et al., arXiv 2026, **2609.35311**. Lifts DROID and Cosmos 3 world-model rollouts into a metric 4D Gaussian field. It assumes rollouts can be spatialised, whereas we test whether they can be.
- **Geometry-aware 4D Video Generation for Robot Manipulation.** Liu, Z. et al., ICLR 2026, **2507.01099**. Uses cross-view pointmap supervision for 3D-consistent robot video, measured with cross-view consistency metrics.
- **GEM-4D.** Zhou, K. et al., arXiv 2026, **2605.22882**. Distils 4D correspondence from a geometry foundation model into a robot video world model.
- **PAIWorld.** Huang, Y. et al., arXiv 2026, **2606.18375**. A 3D-consistent multi-view robot world foundation model (Latent 3D-REPA). Evidence that geometric inconsistency is a recognised failure.
- **UniSim (Learning Interactive Real-World Simulators).** Yang, S. et al., ICLR 2024, **2310.06114**. The seminal action-conditioned video simulator.
- **IRASim.** Zhu, F. et al., ICCV 2025, **2406.14540**. A fine-grained action-conditioned robot video world model.
- **Cosmos World Foundation Model Platform for Physical AI.** NVIDIA, arXiv 2025, **2501.03575**. A world foundation model platform.
- **World Simulation with Video Foundation Models for Physical AI (Cosmos-Predict2.5).** NVIDIA, arXiv 2025, **2511.00062**. The next Cosmos generation, used in WorldOlympiad.
- **Genie: Generative Interactive Environments.** Bruce, J. et al., ICML 2024, **2402.15391**. A latent-action interactive world model.
- **Genie Envisioner.** Liao, Y. et al., arXiv 2025, **2508.05635**. A world foundation platform for manipulation.
- **DreamGen.** Jang, J. et al., arXiv 2025, **2505.12705**. Video world models as robot data generators.
- **Vid2World.** Huang, S. et al., arXiv 2025, **2505.14357**. Converts pretrained video diffusion models into action-conditioned world models.
- **RoboScape.** Shang, Y. et al., NeurIPS 2025, **2506.23135**. A physics-informed world model with depth and keypoint supervision.
- **UniPi (Learning Universal Policies via Text-Guided Video Generation).** Du, Y. et al., NeurIPS 2023, **2302.00111**. Video generation as planning.
- **Video Language Planning.** Du, Y. et al., ICLR 2024, **2310.10625**.
- **AVDC (Learning to Act from Actionless Videos through Dense Correspondences).** Ko, P.-C. et al., ICLR 2024, **2310.08576**. Recovers actions from generated video via correspondences. Geometric validity directly matters here.
- **Video Prediction Policy.** Hu, Y. et al., ICML 2025, **2412.14803**.
- **Unified World Models.** Zhu, C. et al., arXiv 2025, **2504.02792**. Video and action diffusion pretrained on large robot datasets.
- **Unified Video Action Model.** Li, S. et al., arXiv 2025, **2503.00200**.
- **1X World Model Challenge technical report.** Mereu, R. et al., arXiv 2025, **2510.07092**. Adapts Wan2.2-TI2V-5B (one of our generators) into a robot-state-conditioned world model.

### Group 3: Evaluating robot policies / world-model fidelity with generated video

- **RoboPhys-3D.** Wang, T. et al., arXiv 2026, **2608.28718**. (partial full text) Built on RoboTwin 2.0 simulation. Generated and ground-truth videos go through the same 3D reconstruction (VGGT, VGGT-Ω, 4DGS, 4C4D, COLMAP plus 3DGS), with DROID-SLAM trajectories. It evaluates Wan 2.2, CogVideoX 1.5, Cosmos 3 and RoboDreamer. *Overlap:* a reconstruction-based, ground-truth-referenced geometry evaluation of robot world models, including two of our model families. *Difference:* simulated data rather than real DROID, no perturbed-real calibration controls (only a ground-truth ceiling), wrist cameras not mentioned, and no action-conditioned real-robot model such as Ctrl-World.
- **WorldGym.** Quevedo, J. et al., arXiv 2025, **2506.00613**. A world model as a policy-evaluation environment.
- **WorldEval.** Li, Y. et al., arXiv 2025, **2505.19017**. A world model as a real-world policy evaluator.
- **Evaluating Gemini Robotics Policies in a Veo World Simulator.** Gemini Robotics Team, arXiv 2025, **2512.10675**. A frontier-video-model policy evaluator with multi-view consistency.
- **Reference-Free Assessment of Physical Consistency in World Model-based Video Generation.** Oh, Y. and Yun, S., CVPR 2026 3D-LLM/VLA Workshop, **2606.22363**. Uses DROID-SLAM and SEA-RAFT inconsistency to filter WorldGym/WorldEval-style rollouts, improving task-success agreement. It is close in spirit (geometry-based filtering of robot world-model video) but reference-free and without SfM or epipolar calibration.
- **TriWorldBench.** Liu, X. et al., arXiv 2026, **2609.26314**. Head and wrist tri-view consistency for embodied world models, including 3D coherence metrics.
- **EWMBench.** Yue, H. et al., arXiv 2025, **2505.09694**. Scene, motion and semantic quality of embodied world models.
- **WorldArena.** Shang, Y. et al., arXiv 2026, **2602.08971**. Perception and functional utility, finding a "perception-functionality gap".
- **RoboWM-Bench.** Jiang, F. et al., arXiv 2026, **2604.19092**. Executability of generated manipulation video. It reports "non-physical geometric distortions".
- **RoboTrustBench.** Li, H. et al., arXiv 2026 (EMNLP 2026 Findings per arXiv), **2606.01600**. Built on DROID episodes and evaluated by human and MLLM judges.
- **World-in-World.** Zhang, J. et al., ICLR 2026, **2510.18135**. Closed-loop evaluation of world models.
- **WorldSimBench.** Qin, Y. et al., arXiv 2024, **2410.18072**. Includes embodied scenarios.

### Group 4: Generators

- **LTX-Video.** HaCohen, Y. et al., arXiv 2024, **2501.00103**.
- **CogVideoX.** Yang, Z. et al., ICLR 2025, **2408.06072**. The 5B-I2V model is from this family.
- **Wan: Open and Advanced Large-Scale Video Generative Models.** Team Wan, arXiv 2025, **2503.20314**. This is the Wan 2.1 report. No separate Wan2.2 technical report was found on arXiv, so cite this report plus the model card.
- **Stable Video Diffusion.** Blattmann, A. et al., arXiv 2023, **2311.15127**. The backbone of DepthWorld (per its abstract).

### Group 5: Reconstruction tools

- **Structure-from-Motion Revisited.** Schönberger, J. L. and Frahm, J.-M., CVPR 2016, pp. 4104-4113, DOI 10.1109/CVPR.2016.445 (Crossref). This is COLMAP.
- **Distinctive Image Features from Scale-Invariant Keypoints.** Lowe, D. G., IJCV 60(2):91-110, 2004, DOI 10.1023/B:VISI.0000029664.99615.94 (Crossref). This is SIFT.
- **MAGSAC.** Barath, D. et al., CVPR 2019, **1803.07469**.
- **MAGSAC++.** Barath, D. et al., CVPR 2020, **1912.05909**.
- **DUSt3R.** Wang, S. et al., CVPR 2024, **2312.14132**.
- **MASt3R.** Leroy, V. et al., ECCV 2024, **2406.09756**.
- **VGGT.** Wang, J. et al., CVPR 2025, **2503.11651**.
- **MonST3R.** Zhang, J. et al., ICLR 2025, **2410.03825**. Geometry under motion. Relevant because the robot arm is a dynamic foreground.

### Group 6: Data

- **DROID.** Khazatsky, A. et al., RSS 2024, **2403.12945**.

---

## D. Searched for but NOT verified (do not cite as papers)

- **Genie 2 / Genie 3** (DeepMind): no arXiv or venue paper found. These are blog posts only. Cite as web pages, if at all.
- **1X world model** (company model): no paper from 1X itself. Only the third-party challenge report (2510.07092) is verified.
- **Wan2.2 technical report**: no separate arXiv report. Use the Wan report (2503.20314) plus the GitHub/HF model card.
- **Sora technical report** ("Video generation models as world simulators", OpenAI 2024): a blog/tech page, not checked here.
- **MotionCtrl's own evaluation protocol** (which pose estimator it uses): not verified, so do not describe it.
- **Exact WorldScore 3D-consistency implementation** (DROID-SLAM reprojection): taken from secondary sources only. Check the WorldScore paper before stating it.
- **TSED in Yu et al. 2023 and the epipolar metrics in 4DiM**: from memory. The abstracts do not mention them, so check the full text before attributing.
- **Venues for UWM (2504.02792) and UVA (2503.00200)**: Semantic Scholar lists "Robotics" (likely RSS 2025) but this was not confirmed, so they are cited as arXiv.
- **Physics-IQ venue**: Semantic Scholar suggests WACV, which was not confirmed, so it is cited as arXiv.
- **"RBench"** (mentioned in TriWorldBench): no matching arXiv record found.
- **AstraNav-World (2512.21714)** and **VGGT-World (2603.12655)** are real arXiv papers, but secondary-source claims about their VGGT pose-error protocols were not verified, so they were excluded.

---

## C. BibTeX (identical to refs.bib)

```bibtex
% refs.bib -- verified 2026-10-11 against the arXiv API record (title, authors, date, ID)
% and, for venues, the arXiv comment/journal-ref, Crossref, Semantic Scholar, or the ICLR 2026 proceedings listing.
% Where no venue could be confirmed, the entry is an arXiv @article with eprint fields.

% ===================== Group 1: geometric / 3D consistency evaluation of generated video =====================

@article{li2024sora,
  title         = {Sora Generates Videos with Stunning Geometrical Consistency},
  author        = {Li, Xuanyi and Zhou, Daquan and Zhang, Chenxu and Wei, Shaodong and Hou, Qibin and Cheng, Ming-Ming},
  journal       = {arXiv preprint arXiv:2402.17403},
  year          = {2024},
  eprint        = {2402.17403},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{asim2025met3r,
  title     = {{MEt3R}: Measuring Multi-View Consistency in Generated Images},
  author    = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {6034--6044},
  year      = {2025},
  eprint    = {2501.06336},
  archivePrefix = {arXiv}
}

@inproceedings{duan2025worldscore,
  title     = {{WorldScore}: A Unified Evaluation Benchmark for World Generation},
  author    = {Duan, Haoyi and Yu, Hong-Xing and Chen, Sirui and Fei-Fei, Li and Wu, Jiajun},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  eprint    = {2504.00983},
  archivePrefix = {arXiv}
}

@article{fan2026geot2v,
  title         = {{GeoT2V-Bench}: Benchmarking {3D} Consistency in Text-to-Video Models via {3D} Reconstruction},
  author        = {Fan, Chenrui and Favaro, Paolo},
  journal       = {arXiv preprint arXiv:2606.24829},
  year          = {2026},
  eprint        = {2606.24829},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{gu2025geco,
  title         = {{GeCo}: Evaluating Geometric Consistency for Video Generation via Motion and Structure},
  author        = {Gu, Leslie and Hur, Junhwa and Herrmann, Charles and Zhan, Fangneng and Zickler, Todd and Sun, Deqing and others},
  journal       = {arXiv preprint arXiv:2512.22274},
  year          = {2025},
  eprint        = {2512.22274},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{dou2026sgc,
  title     = {Measuring {3D} Spatial Geometric Consistency in Dynamic Video Generation},
  author    = {Dou, Weijia and Zheng, Wenzhao and Chen, Weiliang and Zheng, Yu and Zhou, Jie and Lu, Jiwen},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  eprint    = {2603.19048},
  archivePrefix = {arXiv},
  note      = {Accepted at ECCV 2026 per arXiv comment}
}

@article{wu2026pdibench,
  title         = {Quantitative Video World Model Evaluation for Geometric-Consistency},
  author        = {Wu, Jiaxin and Pi, Yihao and Zhang, Yinling and Li, Yuheng and Zou, Xueyan},
  journal       = {arXiv preprint arXiv:2605.15185},
  year          = {2026},
  eprint        = {2605.15185},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{zhao2026worldolympiad,
  title         = {{WorldOlympiad}: Can Your World Model Survive a Triathlon?},
  author        = {Zhao, Yuke and Zhao, Wangbo and Wang, Weijie and Zhang, Zeyu and An, Dakai and Liu, Akide and others},
  journal       = {arXiv preprint arXiv:2606.11129},
  year          = {2026},
  eprint        = {2606.11129},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{lu20254dworldbench,
  title         = {{4DWorldBench}: A Comprehensive Evaluation Framework for {3D/4D} World Generation Models},
  author        = {Lu, Yiting and Luo, Wei and Tu, Peiyan and Li, Haoran and Zhu, Hanxin and Yu, Zihao and others},
  journal       = {arXiv preprint arXiv:2511.19836},
  year          = {2025},
  eprint        = {2511.19836},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{ying2026wbench,
  title         = {{WBench}: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation},
  author        = {Ying, Kaining and Hu, Hengrui and Ren, Siyu and Li, Jiamu and Chen, Fengjiao and Wang, Ziwen and others},
  journal       = {arXiv preprint arXiv:2605.25874},
  year          = {2026},
  eprint        = {2605.25874},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{huang2025jog3r,
  title         = {{JOG3R}: Towards {3D}-Consistent Video Generators},
  author        = {Huang, Chun-Hao Paul and Mitra, Niloy and Jeong, Hyeonho and Yoon, Jae Shin and Ceylan, Duygu},
  journal       = {arXiv preprint arXiv:2501.01409},
  year          = {2025},
  eprint        = {2501.01409},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{yu2023photoconsistent,
  title     = {Long-Term Photometric Consistent Novel View Synthesis with Diffusion Models},
  author    = {Yu, Jason J. and Forghani, Fereshteh and Derpanis, Konstantinos G. and Brubaker, Marcus A.},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  eprint    = {2304.10700},
  archivePrefix = {arXiv}
}

@inproceedings{watson2025controlling,
  title     = {Controlling Space and Time with Diffusion Models},
  author    = {Watson, Daniel and Saxena, Saurabh and Li, Lala and Tagliasacchi, Andrea and Fleet, David J.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2407.07860},
  archivePrefix = {arXiv}
}

@article{he2024cameractrl,
  title         = {{CameraCtrl}: Enabling Camera Control for Text-to-Video Generation},
  author        = {He, Hao and Xu, Yinghao and Guo, Yuwei and Wetzstein, Gordon and Dai, Bo and Li, Hongsheng and Yang, Ceyuan},
  journal       = {arXiv preprint arXiv:2404.02101},
  year          = {2024},
  eprint        = {2404.02101},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{wang2024motionctrl,
  title     = {{MotionCtrl}: A Unified and Flexible Motion Controller for Video Generation},
  author    = {Wang, Zhouxia and Yuan, Ziyang and Wang, Xintao and Chen, Tianshui and Xia, Menghan and Luo, Ping and Shan, Ying},
  booktitle = {SIGGRAPH 2024 Conference Proceedings},
  year      = {2024},
  eprint    = {2312.03641},
  archivePrefix = {arXiv}
}

@article{xu2024camco,
  title         = {{CamCo}: Camera-Controllable {3D}-Consistent Image-to-Video Generation},
  author        = {Xu, Dejia and Nie, Weili and Liu, Chao and Liu, Sifei and Kautz, Jan and Wang, Zhangyang and Vahdat, Arash},
  journal       = {arXiv preprint arXiv:2406.02509},
  year          = {2024},
  eprint        = {2406.02509},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{bahmani2025ac3d,
  title     = {{AC3D}: Analyzing and Improving {3D} Camera Control in Video Diffusion Transformers},
  author    = {Bahmani, Sherwin and Skorokhodov, Ivan and Qian, Guocheng and Siarohin, Aliaksandr and Menapace, Willi and Tagliasacchi, Andrea and Lindell, David B. and Tulyakov, Sergey},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {22875--22889},
  year      = {2025},
  eprint    = {2411.18673},
  archivePrefix = {arXiv}
}

@inproceedings{huang2024vbench,
  title     = {{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
  author    = {Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {21807--21818},
  year      = {2024},
  eprint    = {2311.17982},
  archivePrefix = {arXiv}
}

@article{zheng2025vbench2,
  title         = {{VBench-2.0}: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness},
  author        = {Zheng, Dian and Huang, Ziqi and Liu, Hongbo and Zou, Kai and He, Yinan and Zhang, Fan and Gu, Lulu and Zhang, Yuanhan and others},
  journal       = {arXiv preprint arXiv:2503.21755},
  year          = {2025},
  eprint        = {2503.21755},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{li2025worldmodelbench,
  title     = {{WorldModelBench}: Judging Video Generation Models As World Models},
  author    = {Li, Dacheng and Fang, Yunhao and Chen, Yukang and Yang, Shuo and Cao, Shiyi and Wong, Justin and Luo, Michael and Wang, Xiaolong and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2502.20694},
  archivePrefix = {arXiv}
}

@article{meng2024phygenbench,
  title         = {Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation},
  author        = {Meng, Fanqing and Liao, Jiaqi and Tan, Xinyu and Shao, Wenqi and Lu, Quanfeng and Zhang, Kaipeng and Cheng, Yu and Li, Dianqi and others},
  journal       = {arXiv preprint arXiv:2410.05363},
  year          = {2024},
  eprint        = {2410.05363},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{motamed2025physicsiq,
  title         = {Do Generative Video Models Understand Physical Principles?},
  author        = {Motamed, Saman and Culp, Laura and Swersky, Kevin and Jaini, Priyank and Geirhos, Robert},
  journal       = {arXiv preprint arXiv:2501.09038},
  year          = {2025},
  eprint        = {2501.09038},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{bansal2025videophy,
  title     = {{VideoPhy}: Evaluating Physical Commonsense for Video Generation},
  author    = {Bansal, Hritik and Lin, Zongyu and Xie, Tianyi and Zong, Zeshun and Yarom, Michal and Bitton, Yonatan and Jiang, Chenfanfu and Sun, Yizhou and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2406.03520},
  archivePrefix = {arXiv}
}

@article{bansal2025videophy2,
  title         = {{VideoPhy-2}: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation},
  author        = {Bansal, Hritik and Peng, Clark and Bitton, Yonatan and Goldenberg, Roman and Grover, Aditya and Chang, Kai-Wei},
  journal       = {arXiv preprint arXiv:2503.06800},
  year          = {2025},
  eprint        = {2503.06800},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{kang2025howfar,
  title     = {How Far is Video Generation from World Model: A Physical Law Perspective},
  author    = {Kang, Bingyi and Yue, Yang and Lu, Rui and Lin, Zhijie and Zhao, Yang and Wang, Kaixin and Huang, Gao and Feng, Jiashi},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  eprint    = {2411.02385},
  archivePrefix = {arXiv}
}

@article{zhu2024sorasurvey,
  title         = {Is {Sora} a World Simulator? A Comprehensive Survey on General World Models and Beyond},
  author        = {Zhu, Zheng and Wang, Xiaofeng and Zhao, Wangbo and Min, Chen and Li, Bohan and Deng, Nianchen and Dou, Min and Wang, Yuqi and others},
  journal       = {arXiv preprint arXiv:2405.03520},
  year          = {2024},
  eprint        = {2405.03520},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{wiedemer2025videozeroshot,
  title         = {Video Models are Zero-Shot Learners and Reasoners},
  author        = {Wiedemer, Thadd{\"a}us and Li, Yuxuan and Vicol, Paul and Gu, Shixiang Shane and Matarese, Nick and Swersky, Kevin and Kim, Been and Jaini, Priyank and others},
  journal       = {arXiv preprint arXiv:2509.20328},
  year          = {2025},
  eprint        = {2509.20328},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

% ===================== Group 2: video world models for robotics =====================

@inproceedings{guo2026ctrlworld,
  title     = {{Ctrl-World}: A Controllable Generative World Model for Robot Manipulation},
  author    = {Guo, Yanjiang and Shi, Lucy Xiaoyang and Chen, Jianyu and Finn, Chelsea},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2510.10125},
  archivePrefix = {arXiv}
}

@inproceedings{yang2024unisim,
  title     = {Learning Interactive Real-World Simulators},
  author    = {Yang, Sherry and Du, Yilun and Ghasemipour, Kamyar and Tompson, Jonathan and Kaelbling, Leslie and Schuurmans, Dale and Abbeel, Pieter},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2310.06114},
  archivePrefix = {arXiv}
}

@inproceedings{zhu2025irasim,
  title     = {{IRASim}: A Fine-Grained World Model for Robot Manipulation},
  author    = {Zhu, Fangqi and Wu, Hongtao and Guo, Song and Liu, Yuxiao and Cheang, Chilam and Kong, Tao},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {9834--9844},
  year      = {2025},
  eprint    = {2406.14540},
  archivePrefix = {arXiv}
}

@article{nvidia2025cosmos,
  title         = {Cosmos World Foundation Model Platform for Physical {AI}},
  author        = {{NVIDIA} and Agarwal, Niket and Ali, Arslan and Bala, Maciej and Balaji, Yogesh and Barker, Erik and Cai, Tiffany and others},
  journal       = {arXiv preprint arXiv:2501.03575},
  year          = {2025},
  eprint        = {2501.03575},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{nvidia2025cosmospredict25,
  title         = {World Simulation with Video Foundation Models for Physical {AI}},
  author        = {{NVIDIA} and Ali, Arslan and Bai, Junjie and Bala, Maciej and Balaji, Yogesh and others},
  journal       = {arXiv preprint arXiv:2511.00062},
  year          = {2025},
  eprint        = {2511.00062},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{bruce2024genie,
  title     = {Genie: Generative Interactive Environments},
  author    = {Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and Hughes, Edward and Lai, Matthew and Mavalankar, Aditi and others},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2402.15391},
  archivePrefix = {arXiv}
}

@article{liao2025genieenvisioner,
  title         = {Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation},
  author        = {Liao, Yue and Zhou, Pengfei and Huang, Siyuan and Yang, Donglin and Chen, Shengcong and Jiang, Yuxin and Hu, Yue and Cai, Jingbin and others},
  journal       = {arXiv preprint arXiv:2508.05635},
  year          = {2025},
  eprint        = {2508.05635},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{jang2025dreamgen,
  title         = {{DreamGen}: Unlocking Generalization in Robot Learning through Video World Models},
  author        = {Jang, Joel and Ye, Seonghyeon and Lin, Zongyu and Xiang, Jiannan and Bjorck, Johan and Fang, Yu and Hu, Fengyuan and Huang, Spencer and others},
  journal       = {arXiv preprint arXiv:2505.12705},
  year          = {2025},
  eprint        = {2505.12705},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{huang2025vid2world,
  title         = {{Vid2World}: Crafting Video Diffusion Models to Interactive World Models},
  author        = {Huang, Siqiao and Wu, Jialong and Zhou, Qixing and Miao, Shangchen and Long, Mingsheng},
  journal       = {arXiv preprint arXiv:2505.14357},
  year          = {2025},
  eprint        = {2505.14357},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@inproceedings{shang2025roboscape,
  title     = {{RoboScape}: Physics-informed Embodied World Model},
  author    = {Shang, Yu and Zhang, Xin and Tang, Yinzhou and Jin, Lei and Gao, Chen and Wu, Wei and Li, Yong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2506.23135},
  archivePrefix = {arXiv}
}

@inproceedings{du2023unipi,
  title     = {Learning Universal Policies via Text-Guided Video Generation},
  author    = {Du, Yilun and Yang, Mengjiao and Dai, Bo and Dai, Hanjun and Nachum, Ofir and Tenenbaum, Joshua B. and Schuurmans, Dale and Abbeel, Pieter},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2302.00111},
  archivePrefix = {arXiv}
}

@inproceedings{du2024vlp,
  title     = {Video Language Planning},
  author    = {Du, Yilun and Yang, Mengjiao and Florence, Pete and Xia, Fei and Wahid, Ayzaan and Ichter, Brian and Sermanet, Pierre and Yu, Tianhe and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2310.10625},
  archivePrefix = {arXiv}
}

@inproceedings{ko2024avdc,
  title     = {Learning to Act from Actionless Videos through Dense Correspondences},
  author    = {Ko, Po-Chen and Mao, Jiayuan and Du, Yilun and Sun, Shao-Hua and Tenenbaum, Joshua B.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2310.08576},
  archivePrefix = {arXiv}
}

@inproceedings{hu2025vpp,
  title     = {Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations},
  author    = {Hu, Yucheng and Guo, Yanjiang and Wang, Pengchao and Chen, Xiaoyu and Wang, Yen-Jen and Zhang, Jianke and Sreenath, Koushil and Lu, Chaochao and Chen, Jianyu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  eprint    = {2412.14803},
  archivePrefix = {arXiv}
}

@article{zhu2025uwm,
  title         = {Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets},
  author        = {Zhu, Chuning and Yu, Raymond and Feng, Siyuan and Burchfiel, Benjamin and Shah, Paarth and Gupta, Abhishek},
  journal       = {arXiv preprint arXiv:2504.02792},
  year          = {2025},
  eprint        = {2504.02792},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{li2025uva,
  title         = {Unified Video Action Model},
  author        = {Li, Shuang and Gao, Yihuai and Sadigh, Dorsa and Song, Shuran},
  journal       = {arXiv preprint arXiv:2503.00200},
  year          = {2025},
  eprint        = {2503.00200},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{mereu20251xwm,
  title         = {Generative World Modelling for Humanoids: {1X} World Model Challenge Technical Report},
  author        = {Mereu, Riccardo and Scannell, Aidan and Hou, Yuxin and Zhao, Yi and Jitta, Aditya and Dominguez, Antonio and others},
  journal       = {arXiv preprint arXiv:2510.07092},
  year          = {2025},
  eprint        = {2510.07092},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@inproceedings{bardhan2026depthworld,
  title     = {{DepthWorld}: {3D} World Model for Robot Manipulation},
  author    = {Bardhan, Jai and Sivic, Josef and Petrik, Vladimir},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  eprint    = {2610.08780},
  archivePrefix = {arXiv},
  note      = {Accepted at CoRL 2026 per arXiv comment}
}

@inproceedings{zheng2026memworld,
  title     = {{Mem-World}: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation},
  author    = {Zheng, Zirui and Yu, Jiaqian and Peng, Xiongfeng and Shi, Jun and Li, Mingyi and Zhang, Chao and others},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  eprint    = {2606.18960},
  archivePrefix = {arXiv},
  note      = {CoRL 2026 per arXiv comment}
}

@article{kim2026rogsw4rld,
  title         = {{RoGSW4RLD}: Feed-Forward {4D} {Gaussian} Lifting for Robot World Model Rollouts},
  author        = {Kim, Jin Hyun and Kim, Min Young and Song, Soohwan and Kim, Daekyum},
  journal       = {arXiv preprint arXiv:2609.35311},
  year          = {2026},
  eprint        = {2609.35311},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@inproceedings{liu2026geometry4d,
  title     = {Geometry-aware {4D} Video Generation for Robot Manipulation},
  author    = {Liu, Zeyi and Li, Shuang and Cousineau, Eric and Feng, Siyuan and Burchfiel, Benjamin and Song, Shuran},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2507.01099},
  archivePrefix = {arXiv}
}

@article{zhou2026gem4d,
  title         = {{GEM-4D}: Geometry-Enhanced Video World Models for Robot Manipulation},
  author        = {Zhou, Kaichen and Chen, Yuzhen and Zhan, Fangneng and Hua, Hang and Chen, Grace and Chang, Xinhai and others},
  journal       = {arXiv preprint arXiv:2605.22882},
  year          = {2026},
  eprint        = {2605.22882},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{huang2026paiworld,
  title         = {{PAIWorld}: A {3D}-Consistent World Foundation Model for Robotic Manipulation},
  author        = {Huang, Yuhang and Lv, Xuan and Xu, Junyan and Yu, Zhiyuan and Zhang, Jiazhao and Hu, Ruizhen and others},
  journal       = {arXiv preprint arXiv:2606.18375},
  year          = {2026},
  eprint        = {2606.18375},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

% ===================== Group 3: evaluating policies / world-model fidelity for robotics =====================

@article{quevedo2025worldgym,
  title         = {{WorldGym}: World Model as An Environment for Policy Evaluation},
  author        = {Quevedo, Julian and Sharma, Ansh Kumar and Sun, Yixiang and Suryavanshi, Varad and Liang, Percy and Yang, Sherry},
  journal       = {arXiv preprint arXiv:2506.00613},
  year          = {2025},
  eprint        = {2506.00613},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{li2025worldeval,
  title         = {{WorldEval}: World Model as Real-World Robot Policies Evaluator},
  author        = {Li, Yaxuan and Zhu, Yichen and Wen, Junjie and Shen, Chaomin and Xu, Yi},
  journal       = {arXiv preprint arXiv:2505.19017},
  year          = {2025},
  eprint        = {2505.19017},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{gemini2025veo,
  title         = {Evaluating {Gemini} Robotics Policies in a {Veo} World Simulator},
  author        = {{Gemini Robotics Team} and Choromanski, Krzysztof and Devin, Coline and Du, Yilun and Dwibedi, Debidatta and Gao, Ruiqi and others},
  journal       = {arXiv preprint arXiv:2512.10675},
  year          = {2025},
  eprint        = {2512.10675},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{wang2026robophys3d,
  title         = {{RoboPhys-3D}: A Comprehensive Embodied World Model Evaluation via {3D} Reconstruction},
  author        = {Wang, Tianyi and Chen, Jiazhou and Xu, Yiming and Li, Xiangyu and Zeng, Tianyi and Chou, Chih-Hsien and others},
  journal       = {arXiv preprint arXiv:2608.28718},
  year          = {2026},
  eprint        = {2608.28718},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{yue2025ewmbench,
  title         = {{EWMBench}: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models},
  author        = {Yue, Hu and Huang, Siyuan and Liao, Yue and Chen, Shengcong and Zhou, Pengfei and Chen, Liliang and Yao, Maoqing and Ren, Guanghui},
  journal       = {arXiv preprint arXiv:2505.09694},
  year          = {2025},
  eprint        = {2505.09694},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{shang2026worldarena,
  title         = {{WorldArena}: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models},
  author        = {Shang, Yu and Li, Zhuohang and Ma, Yiding and Su, Weikang and Jin, Xin and Wang, Ziyou and others},
  journal       = {arXiv preprint arXiv:2602.08971},
  year          = {2026},
  eprint        = {2602.08971},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{jiang2026robowmbench,
  title         = {{RoboWM-Bench}: A Benchmark for Evaluating World Models in Robotic Manipulation},
  author        = {Jiang, Feng and Chen, Yang and Xu, Kyle and Liu, Yuchen and Wang, Haifeng and Shen, Zhenhao and others},
  journal       = {arXiv preprint arXiv:2604.19092},
  year          = {2026},
  eprint        = {2604.19092},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{liu2026triworldbench,
  title         = {{TriWorldBench}: A Tri-View Consistency Perspective on Embodied World Models},
  author        = {Liu, Xuanyi and Wang, Haofeng and Li, Ruiqi and Yu, Danni and Wan, Rui and Zhang, Ruixu and others},
  journal       = {arXiv preprint arXiv:2609.26314},
  year          = {2026},
  eprint        = {2609.26314},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}

@article{li2026robotrustbench,
  title         = {{RoboTrustBench}: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation},
  author        = {Li, Huiqiong and Wang, Jiayu and Mei, Zhiting and Majumdar, Anirudha and Chen, Jingjing and Zhu, Bin},
  journal       = {arXiv preprint arXiv:2606.01600},
  year          = {2026},
  eprint        = {2606.01600},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {EMNLP 2026 Findings per arXiv comment}
}

@article{oh2026referencefree,
  title         = {Reference-Free Assessment of Physical Consistency in World Model-based Video Generation},
  author        = {Oh, Yun and Yun, Sukmin},
  journal       = {arXiv preprint arXiv:2606.22363},
  year          = {2026},
  eprint        = {2606.22363},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  note          = {2nd 3D-LLM/VLA Workshop, CVPR 2026, per arXiv comment}
}

@inproceedings{zhang2026worldinworld,
  title     = {{World-in-World}: World Models in a Closed-Loop World},
  author    = {Zhang, Jiahan and Jiang, Muqing and Dai, Nanru and Lu, Taiming and Uzunoglu, Arda and Zhang, Shunchi and Wei, Yana and Wang, Jiahao and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2510.18135},
  archivePrefix = {arXiv}
}

@article{qin2024worldsimbench,
  title         = {{WorldSimBench}: Towards Video Generation Models as World Simulators},
  author        = {Qin, Yiran and Shi, Zhelun and Yu, Jiwen and Wang, Xijun and Zhou, Enshen and Li, Lijun and Yin, Zhenfei and Liu, Xihui and others},
  journal       = {arXiv preprint arXiv:2410.18072},
  year          = {2024},
  eprint        = {2410.18072},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

% ===================== Group 4: generators =====================

@article{hacohen2024ltxvideo,
  title         = {{LTX-Video}: Realtime Video Latent Diffusion},
  author        = {HaCohen, Yoav and Chiprut, Nisan and Brazowski, Benny and Shalem, Daniel and Moshe, Dudu and Richardson, Eitan and Levin, Eran and Shiran, Guy and others},
  journal       = {arXiv preprint arXiv:2501.00103},
  year          = {2024},
  eprint        = {2501.00103},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{yang2025cogvideox,
  title     = {{CogVideoX}: Text-to-Video Diffusion Models with An Expert Transformer},
  author    = {Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2408.06072},
  archivePrefix = {arXiv}
}

@article{wan2025wan,
  title         = {Wan: Open and Advanced Large-Scale Video Generative Models},
  author        = {{Team Wan} and Wang, Ang and Ai, Baole and Wen, Bin and Mao, Chaojie and Xie, Chen-Wei and Chen, Di and Yu, Feiwu and others},
  journal       = {arXiv preprint arXiv:2503.20314},
  year          = {2025},
  eprint        = {2503.20314},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{blattmann2023svd,
  title         = {Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets},
  author        = {Blattmann, Andreas and Dockhorn, Tim and Kulal, Sumith and Mendelevitch, Daniel and Kilian, Maciej and Lorenz, Dominik and Levi, Yam and English, Zion and others},
  journal       = {arXiv preprint arXiv:2311.15127},
  year          = {2023},
  eprint        = {2311.15127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

% ===================== Group 5: reconstruction tools =====================

@inproceedings{schonberger2016sfm,
  title     = {Structure-from-Motion Revisited},
  author    = {Sch{\"o}nberger, Johannes L. and Frahm, Jan-Michael},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4104--4113},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.445}
}

@article{lowe2004sift,
  title   = {Distinctive Image Features from Scale-Invariant Keypoints},
  author  = {Lowe, David G.},
  journal = {International Journal of Computer Vision},
  volume  = {60},
  number  = {2},
  pages   = {91--110},
  year    = {2004},
  doi     = {10.1023/B:VISI.0000029664.99615.94}
}

@inproceedings{barath2019magsac,
  title     = {{MAGSAC}: Marginalizing Sample Consensus},
  author    = {Barath, Daniel and Noskova, Jana and Matas, Jiri},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10189--10197},
  year      = {2019},
  eprint    = {1803.07469},
  archivePrefix = {arXiv}
}

@inproceedings{barath2020magsacpp,
  title     = {{MAGSAC++}, a Fast, Reliable and Accurate Robust Estimator},
  author    = {Barath, Daniel and Noskova, Jana and Ivashechkin, Maksym and Matas, Jiri},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1301--1309},
  year      = {2020},
  eprint    = {1912.05909},
  archivePrefix = {arXiv}
}

@inproceedings{wang2024dust3r,
  title     = {{DUSt3R}: Geometric {3D} Vision Made Easy},
  author    = {Wang, Shuzhe and Leroy, Vincent and Cabon, Yohann and Chidlovskii, Boris and Revaud, Jerome},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {20697--20709},
  year      = {2024},
  eprint    = {2312.14132},
  archivePrefix = {arXiv}
}

@inproceedings{leroy2024mast3r,
  title     = {Grounding Image Matching in {3D} with {MASt3R}},
  author    = {Leroy, Vincent and Cabon, Yohann and Revaud, J{\'e}r{\^o}me},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {71--91},
  year      = {2024},
  eprint    = {2406.09756},
  archivePrefix = {arXiv}
}

@inproceedings{wang2025vggt,
  title     = {{VGGT}: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {5294--5306},
  year      = {2025},
  eprint    = {2503.11651},
  archivePrefix = {arXiv}
}

@inproceedings{zhang2025monst3r,
  title     = {{MonST3R}: A Simple Approach for Estimating Geometry in the Presence of Motion},
  author    = {Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Jampani, Varun and Darrell, Trevor and Cole, Forrester and Sun, Deqing and Yang, Ming-Hsuan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.03825},
  archivePrefix = {arXiv}
}

% ===================== Group 6: data =====================

@inproceedings{khazatsky2024droid,
  title     = {{DROID}: A Large-Scale In-The-Wild Robot Manipulation Dataset},
  author    = {Khazatsky, Alexander and Pertsch, Karl and Nair, Suraj and Balakrishna, Ashwin and Dasari, Sudeep and Karamcheti, Siddharth and Nasiriany, Soroush and Srirama, Mohan Kumar and others},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2024},
  eprint    = {2403.12945},
  archivePrefix = {arXiv}
}
```
