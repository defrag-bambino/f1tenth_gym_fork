#!/usr/bin/env python3
"""
Script to fine-tune a DreamerV3 agent on the real F1Tenth car after detecting out-of-distribution dynamics.

This script adapts the Isaac Sim fine-tuning approach for the real F1Tenth car:
1. Runs inference (eval mode) initially to establish baseline
2. Monitors for OOD detection using world model imagination
3. When OOD detected: immediately switches to TRAIN mode
4. During TRAIN mode: collects data AND trains jointly (WM + policy) online
5. After finetuning completes, switches back to EVAL mode

Key differences from simulation version:
- Uses F1TenthReal environment instead of Isaac Sim
- No robot modification (real car dynamics change naturally)
- Single environment (real car)
- Manual reset handling
- ROS2-based communication

Usage:
    python finetune_dreamerv3_real.py --checkpoint /path/to/checkpoint --max_speed 3.0
"""

import argparse
import pathlib
import sys
import traceback
import warnings

# Filter out PyTorch RNN memory warning
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory", category=UserWarning)

# Add paths BEFORE parsing args
dreamerv3_path = pathlib.Path(__file__).parent.parent.parent.parent / "dreamerv3"
if dreamerv3_path.exists():
    sys.path.insert(0, str(dreamerv3_path))
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune DreamerV3 agent on real F1Tenth car after detecting OOD dynamics.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")

# Real car parameters
parser.add_argument("--max_speed", type=float, default=3.0, help="Maximum speed for real car (m/s). Use conservative values for safety.")
parser.add_argument("--step_frequency", type=float, default=20.0, help="Step frequency for real car (Hz).")
parser.add_argument("--collision_threshold", type=float, default=0.1, help="Minimum distance to consider collision (m).")
parser.add_argument("--scan_beams", type=int, default=32, help="Number of LiDAR beams (subsampled from 1080).")
parser.add_argument("--scan_topic", type=str, default="/scan", help="ROS2 topic for LiDAR scan.")
parser.add_argument("--odom_topic", type=str, default="/odom", help="ROS2 topic for odometry.")
parser.add_argument("--drive_topic", type=str, default="/drive", help="ROS2 topic for drive commands.")
parser.add_argument("--automatic_reset", action="store_true", default=False, help="Use automatic reset with external tracking.")
parser.add_argument("--pose_topic", type=str, default="/vrpn_client_node/car/pose", help="ROS2 topic for external pose tracking.")

# Evaluation parameters
parser.add_argument("--baseline_steps", type=int, default=2000, help="Number of steps to establish baseline before OOD detection.")
parser.add_argument("--post_training_steps", type=int, default=5000, help="Number of steps after training completes.")
parser.add_argument("--imagination_horizon", type=int, default=15, help="Number of steps to imagine forward.")
parser.add_argument("--imagination_interval", type=int, default=50, help="Run imagination every N steps.")
parser.add_argument("--stabilization_steps", type=int, default=500, help="Steps before starting error collection.")
parser.add_argument("--error_window_size", type=int, default=50, help="Window size for averaging errors.")

# Fine-tuning parameters
parser.add_argument("--error_threshold", type=float, default=0.1, help="Error increase ratio to trigger fine-tuning.")
parser.add_argument("--reward_threshold", type=float, default=0.1, help="Reward decrease ratio to trigger fine-tuning.")
parser.add_argument("--finetune_steps", type=int, default=300, help="Number of fine-tuning steps (joint WM + policy).")
parser.add_argument("--train_ratio", type=float, default=16.0, help="Training updates per environment step during fine-tuning.")
parser.add_argument("--min_replay_for_training", type=int, default=100, help="Minimum transitions in replay before training can start.")
parser.add_argument("--reward_recovery_ratio", type=float, default=0.85, help="Stop finetuning when avg reward reaches this fraction of baseline reward.")
parser.add_argument("--target_error_reduction", type=float, default=0.5, help="Target error reduction factor.")

# Output
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results.")
parser.add_argument("--no_plots", action="store_true", default=False, help="Disable plotting.")

# Logging
parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging.")
parser.add_argument("--no_tensorboard", action="store_true", default=False, help="Disable TensorBoard logging.")

args_cli = parser.parse_args()


def _make_logger(logdir, args_cli):
    """Create logger with tensorboard support."""
    import elements
    
    step = elements.Counter()
    outputs = []
    
    # Always add terminal output
    outputs.append(elements.logger.TerminalOutput('finetune', 'Finetune'))
    
    # Add tensorboard if enabled
    if args_cli.tensorboard and not args_cli.no_tensorboard:
        outputs.append(elements.logger.TensorBoardOutput(logdir, fps=20))
        print(f"[INFO] TensorBoard logging enabled at {logdir}")
    
    # Add JSON logging
    outputs.append(elements.logger.JSONLOutput(logdir, 'finetune_metrics.jsonl'))
    
    multiplier = 1
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def main():
    """Fine-tune DreamerV3 agent on real F1Tenth car after detecting OOD dynamics."""
    import os
    import collections
    import copy
    import numpy as np
    import ruamel.yaml as yaml
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from functools import partial as bind
    
    # Import after setup
    import elements
    import embodied
    from embodied.core import streams
    from dreamerv3.agent import Agent
    from f1tenth_real import F1TenthReal, F1TenthRealWithAutomaticReset
    
    print("\n" + "="*70)
    print("DreamerV3 Fine-Tuning for Real F1Tenth Car")
    print("="*70)
    
    # Expand checkpoint path
    checkpoint_path = pathlib.Path(os.path.expanduser(args_cli.checkpoint))
    
    # Handle different checkpoint path formats:
    # 1. logdir/ckpt/TIMESTAMP/ -> logdir is parent.parent
    # 2. logdir/ckpt/ -> logdir is parent
    # 3. logdir/ -> logdir is checkpoint_path itself
    
    if checkpoint_path.name == 'ckpt':
        # User provided ckpt/ folder, find latest checkpoint inside
        logdir = checkpoint_path.parent
        checkpoint_folders = sorted([d for d in checkpoint_path.iterdir() if d.is_dir()])
        if checkpoint_folders:
            ckpt_dir = checkpoint_folders[-1]  # Latest checkpoint
            print(f"[INFO] Using latest checkpoint: {ckpt_dir.name}")
        else:
            print(f"[ERROR] No checkpoint folders found in: {checkpoint_path}")
            return
    elif checkpoint_path.parent.name == 'ckpt':
        # User provided logdir/ckpt/TIMESTAMP/
        logdir = checkpoint_path.parent.parent
        ckpt_dir = checkpoint_path
    elif (checkpoint_path / 'ckpt').exists():
        # User provided logdir/, check if ckpt/ exists inside
        logdir = checkpoint_path
        ckpt_path = checkpoint_path / 'ckpt'
        checkpoint_folders = sorted([d for d in ckpt_path.iterdir() if d.is_dir()])
        if checkpoint_folders:
            ckpt_dir = checkpoint_folders[-1]  # Latest checkpoint
            print(f"[INFO] Using latest checkpoint: {ckpt_dir.name}")
        else:
            print(f"[ERROR] No checkpoint folders found in: {ckpt_path}")
            return
    else:
        # Assume it's a checkpoint timestamp folder, try to find logdir
        logdir = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'ckpt' else checkpoint_path.parent
        ckpt_dir = checkpoint_path
    
    # Setup output directory
    if args_cli.output_dir:
        output_dir = pathlib.Path(args_cli.output_dir)
    else:
        output_dir = logdir / "finetune_real_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Logdir: {logdir}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Output: {output_dir}")
    
    # Load config from logdir (config.yaml is stored in the logdir, not in ckpt/)
    config_path = logdir / 'config.yaml'
    
    if not config_path.exists():
        print(f"[WARNING] Config file not found at {config_path}")
        print(f"[INFO] Attempting to load from DreamerV3 defaults...")
        # Fallback: load from DreamerV3 defaults (like train_dreamerv3.py does)
        import dreamerv3
        dreamerv3_path = pathlib.Path(dreamerv3.__file__).parent
        default_config_path = dreamerv3_path / "configs.yaml"
        if default_config_path.exists():
            configs = yaml.YAML(typ='safe').load(elements.Path(str(default_config_path)).read())
            config = elements.Config(configs['defaults'])
            # Try to infer model size from checkpoint if possible
            print(f"[INFO] Using DreamerV3 default config (size12m)")
            if 'size12m' in configs:
                config = config.update(configs['size12m'])
        else:
            print(f"[ERROR] Cannot find config file and DreamerV3 defaults not available")
            return
    else:
        print(f"[INFO] Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.YAML(typ='safe').load(f)
        config = elements.Config(config_dict)
    
    print(f"[INFO] Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.YAML(typ='safe').load(f)
    config = elements.Config(config_dict)
    
    # Parameters
    baseline_steps = args_cli.baseline_steps
    post_training_steps = args_cli.post_training_steps
    imagination_horizon = args_cli.imagination_horizon
    imagination_interval = args_cli.imagination_interval
    stabilization_steps = args_cli.stabilization_steps
    error_window_size = args_cli.error_window_size
    
    # Fine-tuning parameters
    error_threshold = args_cli.error_threshold
    reward_threshold = args_cli.reward_threshold
    finetune_steps = args_cli.finetune_steps
    train_ratio = args_cli.train_ratio
    min_replay_for_training = args_cli.min_replay_for_training
    reward_recovery_ratio = args_cli.reward_recovery_ratio
    target_error_reduction = args_cli.target_error_reduction
    
    print(f"\nConfiguration:")
    print(f"  Baseline steps: {baseline_steps}")
    print(f"  Post-training steps: {post_training_steps}")
    print(f"  Imagination horizon: {imagination_horizon}")
    print(f"  Stabilization steps: {stabilization_steps}")
    print(f"  Error threshold: {error_threshold}")
    print(f"  Reward threshold: {reward_threshold}")
    print(f"  Fine-tune steps: {finetune_steps}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Min replay for training: {min_replay_for_training}")
    print(f"  Reward recovery ratio: {reward_recovery_ratio}")
    print(f"  Max speed: {args_cli.max_speed} m/s")
    print(f"  Step frequency: {args_cli.step_frequency} Hz")
    print("="*70 + "\n")
    
    # Create real car environment
    print("[INFO] Creating real F1Tenth environment...")
    if args_cli.automatic_reset:
        env = F1TenthRealWithAutomaticReset(
            scan_beams=args_cli.scan_beams,
            max_speed=args_cli.max_speed,
            step_frequency=args_cli.step_frequency,
            collision_threshold=args_cli.collision_threshold,
            scan_topic=args_cli.scan_topic,
            odom_topic=args_cli.odom_topic,
            drive_topic=args_cli.drive_topic,
            pose_topic=args_cli.pose_topic,
        )
    else:
        env = F1TenthReal(
            scan_beams=args_cli.scan_beams,
            max_speed=args_cli.max_speed,
            step_frequency=args_cli.step_frequency,
            collision_threshold=args_cli.collision_threshold,
            scan_topic=args_cli.scan_topic,
            odom_topic=args_cli.odom_topic,
            drive_topic=args_cli.drive_topic,
        )
    
    # Apply wrappers
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    
    print(f"[INFO] Observation space: {env.obs_space}")
    print(f"[INFO] Action space: {env.act_space}")
    
    # Create agent
    print("[INFO] Creating agent...")
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(output_dir),
        seed=args_cli.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    ))
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint from {ckpt_dir}...")
    if not ckpt_dir.exists():
        print(f"[ERROR] Checkpoint directory not found at {ckpt_dir}")
        return
    
    # Check if ckpt_dir is a timestamped checkpoint folder or the ckpt/ folder itself
    agent_pkl = ckpt_dir / 'agent.pkl'
    if not agent_pkl.exists():
        # Maybe ckpt_dir is the ckpt/ folder, need to find latest timestamp inside
        checkpoint_folders = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
        if not checkpoint_folders:
            print(f"[ERROR] No checkpoint folders found in {ckpt_dir}")
            return
        latest_ckpt = checkpoint_folders[-1]
        print(f"[INFO] Using latest checkpoint: {latest_ckpt.name}")
    else:
        # ckpt_dir is already the timestamped checkpoint folder
        latest_ckpt = ckpt_dir
        print(f"[INFO] Loading checkpoint: {latest_ckpt.name}")
    
    # Initialize checkpoint without path, then load with path (like train_dreamerv3.py)
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(latest_ckpt, keys=['agent'])
    print(f"[INFO] Checkpoint loaded successfully")
    
    # Create logger for tensorboard
    print("[INFO] Setting up logger...")
    logger = _make_logger(output_dir, args_cli)
    step_counter = elements.Counter()
    
    # Log configuration
    logger.add({
        'config/baseline_steps': baseline_steps,
        'config/post_training_steps': post_training_steps,
        'config/imagination_horizon': imagination_horizon,
        'config/stabilization_steps': stabilization_steps,
        'config/error_threshold': error_threshold,
        'config/reward_threshold': reward_threshold,
        'config/finetune_steps': finetune_steps,
        'config/train_ratio': train_ratio,
        'config/reward_recovery_ratio': reward_recovery_ratio,
        'config/max_speed': args_cli.max_speed,
        'config/step_frequency': args_cli.step_frequency,
    }, prefix='finetune')
    logger.write()
    
    # Create replay buffer for fine-tuning
    print("[INFO] Creating replay buffer for fine-tuning...")
    replay_length = config.batch_length + config.replay_context
    finetune_replay = embodied.Replay(
        length=replay_length,
        capacity=100000,  # Enough for fine-tuning
        directory=str(output_dir / 'finetune_replay'),
        online=True,
    )
    
    # Initialize policy carry
    print("[INFO] Initializing policy...")
    carry = agent.init_policy(1)  # Single environment for real car
    
    # Initialize training state (for when we switch to training mode)
    train_state = [agent.init_train(config.batch_size)]
    
    # Get initial observations (wrapped environments need both 'action' and 'reset' keys)
    obs = env.step({
        'action': np.zeros(env.act_space['action'].shape, dtype=np.float32),
        'reset': True,
    })
    
    # Convert observation to vector format for compatibility
    # F1TenthReal uses dict format: {'scan', 'linear_vel_x', 'ang_vel_z', 'delta'}
    # We'll work with dict format directly
    print(f"[INFO] Initial observation keys: {obs.keys()}")
    
    # =========================================================================
    # Setup world model imagination (for OOD detection)
    # =========================================================================
    print("[INFO] Setting up imagination functions...")
    
    inner_agent = agent.model if hasattr(agent, 'model') else None
    
    if inner_agent is not None and hasattr(inner_agent, 'dyn'):
        dyn_module = inner_agent.dyn
        enc_module = inner_agent.enc
        dec_module = inner_agent.dec
        rew_module = inner_agent.rew
        feat2tensor_fn = inner_agent.feat2tensor
        print(f"[INFO] Accessed world model via agent.model.dyn/enc/dec/rew")
    else:
        print(f"[WARNING] Cannot access world model components")
        dyn_module = None
        dec_module = None
        rew_module = None
        feat2tensor_fn = None
    
    def imagine_with_world_model(dyn_carry, dec_carry, actions_sequence, horizon):
        """Perform TRUE imagination rollout using the world model."""
        if dyn_module is None:
            raise RuntimeError("World model not accessible")
        
        def _do_imagination():
            H = horizon
            B = dyn_carry['deter'].shape[0]
            _, imgfeat, imgact = dyn_module.imagine(
                dyn_carry, actions_sequence, H, training=False)
            reset = jnp.zeros((B, H), dtype=bool)
            _, _, recons = dec_module(dec_carry, imgfeat, reset, training=False)
            feat_tensor = feat2tensor_fn(imgfeat)
            rew_dist = rew_module(feat_tensor, bdims=2)
            imagined_rewards = rew_dist.pred()
            imagined_obs = {key: dist.pred() for key, dist in recons.items()}
            return imagined_obs, imagined_rewards
        
        pure_fn = nj.pure(_do_imagination)
        rng_seed = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        with jax.transfer_guard('allow'):
            _, (imagined_obs, imagined_rewards) = pure_fn(
                agent.params, seed=rng_seed, create=True, modify=True, ignore=True)
        
        return imagined_obs, imagined_rewards
    
    # Test imagination
    imagination_available = False
    if dyn_module is not None:
        print("[INFO] Testing world model imagination...")
        with jax.transfer_guard('allow'):
            try:
                test_dyn_carry = dyn_module.initial(1)
                test_dec_carry = dec_module.initial(1) if hasattr(dec_module, 'initial') else {}
                test_actions = {'action': jnp.zeros((1, 5, act_space['action'].shape[0]))}
                test_obs, test_rew = imagine_with_world_model(
                    test_dyn_carry, test_dec_carry, test_actions, 5)
                imagination_available = True
                print("[INFO] ✓ World model imagination is WORKING!")
            except Exception as e:
                print(f"[WARNING] World model imagination test failed: {e}")
                traceback.print_exc()
    
    if not imagination_available:
        print("[ERROR] World model imagination required for fine-tuning detection")
        return
    
    # Helper to safely convert JAX arrays to numpy
    def safe_to_numpy(x):
        if isinstance(x, (np.ndarray, float, int)):
            return np.array(x)
        x_host = jax.device_get(x)
        if hasattr(x_host, 'dtype') and str(x_host.dtype) == 'bfloat16':
            return np.array(x_host, dtype=np.float32)
        return np.array(x_host)
    
    # Helper to convert observation dict to vector (for storage/history)
    def obs_to_vector(obs_dict):
        """Convert observation dict to vector format."""
        parts = [
            obs_dict['scan'].flatten(),
            np.array([obs_dict['linear_vel_x']]),
            np.array([obs_dict['ang_vel_z']]),
            np.array([obs_dict['delta']]),
        ]
        return np.concatenate(parts)
    
    # =========================================================================
    # Create proper training stream (like train.py)
    # =========================================================================
    def make_train_stream():
        """Create a proper training stream from the replay buffer."""
        def sample_fn():
            return finetune_replay.sample(config.batch_size, mode='train')
        
        stream = streams.Stateless(sample_fn)
        stream = streams.Consec(
            stream,
            length=config.batch_length,
            consec=config.consec_train,
            prefix=config.replay_context,
            strict=True,
            contiguous=True,
        )
        return stream
    
    # =========================================================================
    # Storage
    # =========================================================================
    action_history = []
    obs_history = []  # Store dict format
    obs_vector_history = []  # Store vector format for imagination
    reward_history = []
    latent_history = []
    is_first_history = []
    
    prediction_errors_obs = []
    prediction_errors_rew = []
    avg_errors_timeline = []
    avg_rewards_timeline = []
    recent_errors = []
    recent_rewards = []
    first_post_stabilization_reward = None
    
    # Fine-tuning state
    baseline_error = None
    baseline_reward = None
    mode = 'eval'  # Start in eval mode
    finetune_triggered = False
    finetune_completed = False
    finetune_trigger_step = -1
    finetune_start_step = -1
    finetune_end_step = -1
    finetune_steps_done = 0
    ood_trigger_reason = None
    reward_recovery_reached = False
    finetune_finish_reason = None
    
    # Training metrics during fine-tuning
    finetune_train_metrics = []
    
    # Episode tracking
    episodes = collections.defaultdict(elements.Agg)
    episode_count = 0
    total_episodes = 0
    
    # Training ratio tracking
    batch_steps = config.batch_size * config.batch_length
    should_train = elements.when.Ratio(train_ratio / batch_steps)
    
    # Train stream (will be created when needed)
    stream_train = None
    
    # Skipped imaginations counter
    skipped_imaginations = 0
    
    # Track when to stop
    post_training_start_step = -1
    should_terminate = False
    
    print(f"\n[INFO] Starting evaluation on real car...")
    print(f"[INFO] Mode: {mode.upper()}")
    print(f"[INFO] Imagination will run every {imagination_interval} steps")
    print(f"[INFO] Will establish baseline for {baseline_steps} steps")
    print(f"[INFO] Will run {post_training_steps} steps after training completes")
    print(f"[WARNING] Real car operation - ensure safety measures are in place!")
    
    # =========================================================================
    # Main loop - runs until post-training period completes
    # =========================================================================
    step_idx = 0
    
    while not should_terminate:
        # Store current observation (both formats)
        current_obs_dict = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        obs_history.append(current_obs_dict)
        obs_vector_history.append(obs_to_vector(current_obs_dict))
        is_first_history.append(bool(obs['is_first']))
        
        # Store latent state for imagination
        enc_carry, dyn_carry, dec_carry, prevact = carry
        with jax.transfer_guard('allow'):
            def unpack_sharded(x):
                if isinstance(x, list):
                    if len(x) == 1:
                        return x[0]
                    else:
                        return jnp.concatenate(x, axis=0)
                return x
            
            stored_dyn = {k: np.array(jax.device_get(unpack_sharded(v))) for k, v in dyn_carry.items()}
            stored_dec = {k: np.array(jax.device_get(unpack_sharded(v))) for k, v in dec_carry.items()} if dec_carry else {}
            
            latent_history.append({
                'dyn_carry': stored_dyn,
                'dec_carry': stored_dec,
            })
        
        # Batch observation for DreamerV3 (expects batched inputs)
        obs_batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        
        # Get action from policy (mode depends on whether we're fine-tuning)
        policy_mode = 'train' if mode == 'train' else 'eval'
        carry, acts, policy_outs = agent.policy(carry, obs_batched, mode=policy_mode)
        
        # Unbatch action (remove batch dimension)
        acts_unbatched = {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v for k, v in acts.items()}
        acts_unbatched['reset'] = obs['is_last']
        
        action_vec = acts_unbatched.get('action', np.zeros(act_space['action'].shape))
        action_history.append(action_vec)
        
        # Step environment (use unbatched actions)
        obs_new = env.step(acts_unbatched)
        
        # Track reward
        reward = float(obs_new['reward'])
        reward_history.append(reward)
        
        # Add to replay buffer ONLY during TRAIN mode (post-OOD detection)
        if mode == 'train':
            # Unbatch policy_outs as well
            policy_outs_unbatched = {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v 
                                     for k, v in policy_outs.items() if not k.startswith('log/')}
            tran = {k: v for k, v in obs.items()}
            tran.update({k: v for k, v in acts_unbatched.items() if k != 'reset'})
            tran.update(policy_outs_unbatched)
            finetune_replay.add(tran, worker=0)
            
            # Episode tracking
            episode = episodes[0]
            if tran['is_first']:
                episode.reset()
            episode.add('score', tran['reward'], agg='sum')
            episode.add('length', 1, agg='sum')
            
            if tran['is_last']:
                result = episode.result()
                episode_count += 1
                total_episodes += 1
        
        # Move to new observations
        obs = obs_new
        
        # =====================================================================
        # Training (only in train mode, and only if we have minimum data)
        # =====================================================================
        if mode == 'train' and len(finetune_replay) >= max(config.batch_size * replay_length, min_replay_for_training):
            # Initialize stream if needed
            if stream_train is None:
                print("[TRAIN] Initializing training stream...")
                stream_train = iter(agent.stream(make_train_stream()))
            
            # Train based on ratio
            step_counter.increment(1)
            for _ in range(should_train(step_counter)):
                try:
                    batch = next(stream_train)
                    train_state[0], outs, mets = agent.train(train_state[0], batch)
                    finetune_steps_done += 1
                    
                    # Collect metrics
                    step_metrics = {}
                    for k, v in list(mets.items()):
                        try:
                            if hasattr(v, 'item'):
                                step_metrics[k] = float(v.item())
                            else:
                                step_metrics[k] = float(v)
                        except:
                            pass
                    finetune_train_metrics.append(step_metrics)
                    
                    # Log training metrics
                    if step_metrics:
                        logger.add(step_metrics, prefix='train')
                    
                    # Update replay priorities if needed
                    if 'replay' in outs:
                        finetune_replay.update(outs['replay'])
                    
                    # Log progress
                    if finetune_steps_done % 100 == 0:
                        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(step_metrics.items())[:3] if 'loss' in k])
                        print(f"[TRAIN] Step {finetune_steps_done}/{finetune_steps} - {loss_str}")
                    
                    # Check reward recovery
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                    if (baseline_reward is not None and baseline_reward > 0
                            and avg_reward >= baseline_reward * reward_recovery_ratio):
                        reward_recovery_reached = True
                        if finetune_finish_reason is None:
                            finetune_finish_reason = "reward recovery"
                        print(f"\n[TRAIN] Reward recovery threshold reached at step {step_idx} "
                              f"({avg_reward:.4f} ≥ {baseline_reward * reward_recovery_ratio:.4f}); stopping early.")
                        break
                    
                except StopIteration:
                    print("[TRAIN] Stream exhausted, recreating...")
                    stream_train = iter(agent.stream(make_train_stream()))
                except Exception as e:
                    print(f"[TRAIN] Error: {e}")
                    traceback.print_exc()
            
            # Check if fine-tuning is complete
            if not finetune_completed and (finetune_steps_done >= finetune_steps or reward_recovery_reached):
                print(f"\n{'='*60}")
                reason = finetune_finish_reason if finetune_finish_reason else "max steps"
                if finetune_finish_reason is None:
                    finetune_finish_reason = "max steps"
                print(f"[INFO] Fine-tuning COMPLETE after {finetune_steps_done} steps (reason: {reason})")
                print(f"{'='*60}")
                
                # Log training completion
                logger.add({
                    'training/completed': 1,
                    'training/total_steps': finetune_steps_done,
                    'training/finish_reason': 1 if reason == "reward recovery" else 0,
                }, prefix='finetune')
                
                mode = 'eval'
                finetune_completed = True
                finetune_end_step = step_idx
                post_training_start_step = step_idx
                
                # Reinitialize policy carry with updated parameters
                print("[INFO] Reinitializing policy with fine-tuned parameters...")
                carry = agent.init_policy(1)
                
                # Reset error tracking to monitor improvement
                recent_errors = []
                recent_rewards = []
                
                # Save fine-tuned checkpoint
                finetune_ckpt_dir = output_dir / 'ckpt'
                finetune_ckpt_dir.mkdir(exist_ok=True)
                cp_save = elements.Checkpoint(finetune_ckpt_dir)
                cp_save.agent = agent
                cp_save.save()
                print(f"[INFO] Fine-tuned checkpoint saved to: {finetune_ckpt_dir}")
                print(f"[INFO] Starting post-training evaluation for {post_training_steps} steps...")
        
        # =====================================================================
        # Imagination and error calculation (for OOD detection)
        # =====================================================================
        if step_idx >= stabilization_steps and step_idx % imagination_interval == 0:
            start_step = step_idx - imagination_horizon
            if start_step >= 0 and start_step + imagination_horizon <= len(obs_vector_history):
                # Check for episode boundary
                has_reset = any(is_first_history[start_step + 1 + i] for i in range(imagination_horizon)
                               if start_step + 1 + i < len(is_first_history))
                
                if not has_reset:
                    try:
                        # Get actual observations and rewards
                        actual_obs_dicts = obs_history[start_step + 1:start_step + 1 + imagination_horizon]
                        actual_rewards = np.array(reward_history[start_step:start_step + imagination_horizon])
                        
                        # Convert actual obs to vector format for comparison
                        actual_obs_vectors = np.array([obs_to_vector(o) for o in actual_obs_dicts])
                        
                        if len(latent_history) > start_step:
                            with jax.transfer_guard('allow'):
                                past_latent = latent_history[start_step]
                                past_dyn_carry = {k: jax.device_put(v[None, ...]) for k, v in past_latent['dyn_carry'].items()}
                                past_dec_carry = {k: jax.device_put(v[None, ...]) for k, v in past_latent['dec_carry'].items()}
                                
                                actions_np = np.array(action_history[start_step:start_step + imagination_horizon])
                                actions_jax = {'action': jax.device_put(actions_np[None, ...])}
                                
                                imagined_obs_dict, imagined_rewards = imagine_with_world_model(
                                    past_dyn_carry, past_dec_carry,
                                    actions_jax, imagination_horizon
                                )
                                
                                # Convert imagined obs to vector format
                                if 'scan' in imagined_obs_dict:
                                    # Reconstruct vector from dict
                                    imag_obs_vectors = []
                                    for i in range(imagination_horizon):
                                        imag_dict = {k: safe_to_numpy(v[0, i]) for k, v in imagined_obs_dict.items()}
                                        imag_obs_vectors.append(obs_to_vector(imag_dict))
                                    imag_obs = np.array(imag_obs_vectors)
                                else:
                                    # Fallback: concatenate all values
                                    imag_obs = np.concatenate(
                                        [safe_to_numpy(v[0]) for v in imagined_obs_dict.values()], axis=-1)
                                
                                imag_rew = safe_to_numpy(imagined_rewards[0])
                                
                                obs_error = np.mean(np.abs(imag_obs - actual_obs_vectors))
                                rew_error = np.mean(np.abs(imag_rew - actual_rewards))
                                
                                prediction_errors_obs.append(obs_error)
                                prediction_errors_rew.append(rew_error)
                                
                                # Log imagination metrics
                                logger.add({
                                    'imagination/obs_error': obs_error,
                                    'imagination/rew_error': rew_error,
                                }, prefix='finetune')
                                
                    except Exception as e:
                        if step_idx < stabilization_steps + 200:
                            print(f"  [WARNING] Imagination failed at step {step_idx}: {e}")
                            traceback.print_exc()
                else:
                    skipped_imaginations += 1
        
        # Track rolling averages
        if len(prediction_errors_obs) > 0:
            recent_errors.append(prediction_errors_obs[-1])
            if len(recent_errors) > error_window_size:
                recent_errors = recent_errors[-error_window_size:]
            
            recent_rewards.append(reward)
            if len(recent_rewards) > error_window_size:
                recent_rewards = recent_rewards[-error_window_size:]
            
            avg_error = np.mean(recent_errors)
            avg_reward = np.mean(recent_rewards)
            
            avg_errors_timeline.append(avg_error)
            avg_rewards_timeline.append(avg_reward)
            
            # Log rolling averages
            logger.add({
                'metrics/avg_prediction_error': avg_error,
                'metrics/avg_reward': avg_reward,
                'metrics/replay_size': len(finetune_replay),
            }, prefix='finetune')
            
            # Establish baseline before modification
            if first_post_stabilization_reward is None and step_idx >= stabilization_steps:
                first_post_stabilization_reward = avg_reward

            if step_idx < baseline_steps and baseline_error is None and len(recent_errors) >= error_window_size:
                baseline_error = avg_error
                baseline_reward = first_post_stabilization_reward if first_post_stabilization_reward is not None else avg_reward
                print(f"\n[BASELINE] Established at step {step_idx}:")
                print(f"  Baseline error: {baseline_error:.6f}")
                print(f"  Baseline reward: {baseline_reward:.6f}")
                
                # Log baseline
                logger.add({
                    'baseline/error': baseline_error,
                    'baseline/reward': baseline_reward,
                }, prefix='finetune')
            
            # =====================================================================
            # OOD Detection and Training Trigger (only if not already triggered)
            # =====================================================================
            if step_idx > baseline_steps and baseline_error is not None and not finetune_triggered and mode == 'eval':
                error_ratio = avg_error / baseline_error if baseline_error > 0 else float('inf')
                reward_ratio = avg_reward / baseline_reward if baseline_reward != 0 else 0
                
                error_trigger = error_ratio > (1 + error_threshold)
                reward_trigger = reward_ratio < (1 - reward_threshold)
                
                # Trigger training IMMEDIATELY when OOD is detected
                if error_trigger or reward_trigger:
                    trigger_reasons = []
                    if error_trigger:
                        trigger_reasons.append(f"Prediction Error ↑ {error_ratio:.2f}x")
                    if reward_trigger:
                        trigger_reasons.append(f"Reward ↓ {reward_ratio:.2f}x")
                    
                    ood_trigger_reason = ", ".join(trigger_reasons)
                    
                    print(f"\n{'='*70}")
                    print(f"[OOD DETECTED] Out-of-Distribution Dynamics at step {step_idx}!")
                    print(f"{'='*70}")
                    print(f"  Trigger: {ood_trigger_reason}")
                    print(f"  Baseline error: {baseline_error:.6f} → Current: {avg_error:.6f}")
                    print(f"  Baseline reward: {baseline_reward:.6f} → Current: {avg_reward:.6f}")
                    print(f"  Steps since baseline: {step_idx - baseline_steps}")
                    print(f"\n  → Switching to TRAIN mode (online learning)")
                    print(f"  → Will collect data AND train jointly for {finetune_steps} steps")
                    print(f"{'='*70}\n")
                    
                    # Log OOD detection event
                    logger.add({
                        'ood/detected': 1,
                        'ood/error_ratio': error_ratio,
                        'ood/reward_ratio': reward_ratio,
                        'ood/trigger_step': step_idx,
                        'ood/steps_since_baseline': step_idx - baseline_steps,
                    }, prefix='finetune')
                    
                    finetune_triggered = True
                    finetune_trigger_step = step_idx
                    finetune_start_step = step_idx
                    mode = 'train'
                    finetune_steps_done = 0
                    stream_train = None  # Will be recreated
        
        # Progress logging
        if step_idx % 500 == 0:
            avg_err = avg_errors_timeline[-1] if avg_errors_timeline else 0
            avg_rew = avg_rewards_timeline[-1] if avg_rewards_timeline else 0
            n_imag = len(prediction_errors_obs)
            replay_size = len(finetune_replay)
            
            if step_idx < stabilization_steps:
                status = "STABILIZING"
            elif mode == 'train':
                status = f"TRAIN ({finetune_steps_done}/{finetune_steps})"
            elif finetune_completed and post_training_start_step > 0:
                steps_in_post = step_idx - post_training_start_step
                status = f"EVAL (Post-Training {steps_in_post}/{post_training_steps})"
            elif finetune_completed:
                status = "EVAL (Fine-tuned)"
            elif step_idx > baseline_steps:
                status = "EVAL (Monitoring for OOD)"
            else:
                status = "EVAL (Baseline)"
            
            print(f"Step {step_idx}, Mode: {status}, "
                  f"Replay: {replay_size}, Avg Error: {avg_err:.6f}, Avg Reward: {avg_rew:.4f}")
            
            # Write logger every 500 steps
            logger.write()
        
        # Log mode status
        logger.add({
            'status/mode': 1 if mode == 'train' else 0,
            'status/step': step_idx,
            'status/finetune_steps_done': finetune_steps_done,
        }, prefix='finetune')
        
        # Increment step counter at end of loop
        step_idx += 1
        logger.step.increment()
        
        # Check if we should terminate after this step
        if post_training_start_step > 0 and step_idx >= post_training_start_step + post_training_steps:
            should_terminate = True
    
    print("\n[INFO] Evaluation complete. Processing results...")
    
    # =========================================================================
    # Save results
    # =========================================================================
    reward_history = np.array(reward_history)
    prediction_errors_obs = np.array(prediction_errors_obs)
    prediction_errors_rew = np.array(prediction_errors_rew)
    avg_errors_timeline = np.array(avg_errors_timeline)
    avg_rewards_timeline = np.array(avg_rewards_timeline)
    
    save_data = {
        "prediction_errors_obs": prediction_errors_obs,
        "prediction_errors_rew": prediction_errors_rew,
        "avg_errors_timeline": avg_errors_timeline,
        "avg_rewards_timeline": avg_rewards_timeline,
        "reward_history": reward_history,
        "baseline_error": baseline_error,
        "baseline_reward": baseline_reward,
        "reward_recovery_reached": reward_recovery_reached,
        "finetune_trigger_step": finetune_trigger_step,
        "finetune_start_step": finetune_start_step,
        "finetune_end_step": finetune_end_step,
        "finetune_completed": finetune_completed,
        "config": {
            "seed": args_cli.seed,
            "baseline_steps": baseline_steps,
            "post_training_steps": post_training_steps,
            "total_steps_executed": step_idx,
            "imagination_horizon": imagination_horizon,
            "error_threshold": error_threshold,
            "reward_threshold": reward_threshold,
            "finetune_steps": finetune_steps,
            "train_ratio": train_ratio,
            "min_replay_for_training": min_replay_for_training,
            "reward_recovery_ratio": reward_recovery_ratio,
            "ood_trigger_reason": ood_trigger_reason,
            "finetune_finish_reason": finetune_finish_reason,
        }
    }
    
    results_path = output_dir / "finetune_results.npz"
    np.savez(results_path, **save_data)
    print(f"[INFO] Results saved to: {results_path}")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*70)
    print("FINE-TUNING SUMMARY (Real F1Tenth Car)")
    print("="*70)
    print(f"Total steps executed: {step_idx}")
    print(f"Baseline steps: {baseline_steps}")
    print(f"Post-training steps: {post_training_steps}")
    print(f"Imagination runs: {len(prediction_errors_obs)}")
    print(f"Skipped imaginations (episode boundaries): {skipped_imaginations}")
    print(f"\nData Collection Strategy:")
    print(f"  Online learning during training: Yes")
    print(f"  Replay buffer final size: {len(finetune_replay)}")
    print(f"  (Contains ONLY post-OOD detection dynamics)")
    print(f"\nBaseline:")
    print(f"  Error: {baseline_error:.6f}" if baseline_error else "  Error: Not established")
    print(f"  Reward: {baseline_reward:.6f}" if baseline_reward else "  Reward: Not established")
    print(f"\nOOD Detection & Fine-tuning:")
    print(f"  OOD Detected: {'Yes' if finetune_triggered else 'No'}")
    if finetune_triggered:
        print(f"  OOD Trigger: {ood_trigger_reason}")
        print(f"  Detection step: {finetune_trigger_step}")
        print(f"  Steps from baseline to detection: {finetune_trigger_step - baseline_steps}")
        print(f"  Training start step: {finetune_start_step}")
        print(f"  Training end step: {finetune_end_step if finetune_end_step > 0 else 'N/A'}")
        print(f"  Training steps completed: {finetune_steps_done}")
        print(f"  Reward recovery ratio: {reward_recovery_ratio:.2f}")
        print(f"  Reward recovery reached: {'Yes' if reward_recovery_reached else 'No'}")
    print(f"  Fine-tuning Completed: {'Yes' if finetune_completed else 'No'}")
    print("="*70)
    
    # =========================================================================
    # Generate plots
    # =========================================================================
    if not args_cli.no_plots:
        try:
            import matplotlib.pyplot as plt
            
            print("\n[INFO] Generating plots...")
            
            # Plot 1: Timeline with events
            if len(avg_errors_timeline) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                
                x_values = np.linspace(stabilization_steps, step_idx - 1, len(avg_errors_timeline))
                
                ax1.plot(x_values, avg_errors_timeline, 'r-', linewidth=2, label='Avg Prediction Error')
                if baseline_error is not None:
                    ax1.axhline(y=baseline_error, color='blue', linestyle='--', 
                                alpha=0.7, label=f'Baseline ({baseline_error:.4f})')
                    ax1.axhline(y=baseline_error * (1 + error_threshold), color='orange', 
                                linestyle=':', alpha=0.7, label='Threshold')
                
                if baseline_steps:
                    ax1.axvline(x=baseline_steps, color='purple', linestyle='--', 
                                linewidth=2, label='Baseline Period End')
                
                if finetune_start_step > 0:
                    ax1.axvline(x=finetune_start_step, color='green', linestyle='-', 
                                alpha=0.8, linewidth=2, label='Fine-tuning Start')
                if finetune_end_step > 0:
                    ax1.axvline(x=finetune_end_step, color='green', linestyle='--', 
                                alpha=0.8, linewidth=2, label='Fine-tuning End')
                    ax1.axvspan(finetune_start_step, finetune_end_step, alpha=0.2, color='green')
                
                ax1.set_ylabel('Prediction Error')
                ax1.set_title('Prediction Error Timeline with Fine-tuning (Real Car)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Reward plot
                ax2.plot(x_values[:len(avg_rewards_timeline)], avg_rewards_timeline, 
                         'b-', linewidth=2, label='Avg Reward')
                if baseline_reward is not None:
                    ax2.axhline(y=baseline_reward, color='green', linestyle='--', 
                                alpha=0.7, label=f'Baseline ({baseline_reward:.4f})')
                
                if baseline_steps:
                    ax2.axvline(x=baseline_steps, color='purple', linestyle='--', linewidth=2)
                
                if finetune_start_step > 0 and finetune_end_step > 0:
                    ax2.axvspan(finetune_start_step, finetune_end_step, alpha=0.2, color='green')
                
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Reward')
                ax2.set_title('Reward Timeline')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "finetune_timeline.png", dpi=150)
                plt.close()
                print(f"  Saved: finetune_timeline.png")
            
            # Plot 2: Training losses during fine-tuning
            if finetune_train_metrics:
                first_entry = next((e for e in finetune_train_metrics if len(e) > 0), {})
                loss_keys = [k for k in first_entry.keys() if k.startswith('loss/')]
                
                if loss_keys:
                    n_losses = min(len(loss_keys), 6)
                    fig, axes = plt.subplots((n_losses + 1) // 2, 2, figsize=(14, 3.5 * ((n_losses + 1) // 2)))
                    if n_losses == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for i, key in enumerate(loss_keys[:n_losses]):
                        values = [l.get(key, 0) for l in finetune_train_metrics]
                        axes[i].plot(values, linewidth=1.5, alpha=0.8)
                        
                        # Add moving average
                        if len(values) > 10:
                            window = min(50, len(values) // 10)
                            ma = np.convolve(values, np.ones(window)/window, mode='valid')
                            axes[i].plot(range(window-1, len(values)), ma, 
                                       linewidth=2.5, color='orange', alpha=0.7, label='Moving Avg')
                        
                        axes[i].set_title(key, fontsize=11)
                        axes[i].set_xlabel('Training Step')
                        axes[i].set_ylabel('Loss')
                        axes[i].grid(True, alpha=0.3)
                        axes[i].legend(fontsize=8)
                    
                    for i in range(n_losses, len(axes)):
                        axes[i].axis('off')
                    
                    plt.suptitle('Fine-tuning Losses (Joint WM + Policy)', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(output_dir / "finetune_losses.png", dpi=150)
                    plt.close()
                    print(f"  Saved: finetune_losses.png")
            
            print(f"\n[INFO] All plots saved to: {output_dir}")
            
        except ImportError:
            print("[WARNING] matplotlib not available, skipping plots")
        except Exception as e:
            print(f"[WARNING] Error generating plots: {e}")
            traceback.print_exc()
    
    # Final logger write and cleanup
    print("\n[INFO] Finalizing logs...")
    logger.write()
    logger.close()
    
    # Cleanup
    env.close()
    print("\n[INFO] Fine-tuning analysis finished!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Fine-tuning analysis failed with exception:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}")