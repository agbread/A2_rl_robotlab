from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 500
    experiment_name = "a2_rough" 
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlMoeCtsActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCriticMoECTS"
    init_noise_std = 1.0
    expert_num = 8 # number of experts in the student model
    latent_dim = 32
    norm_type = 'l2norm' # normalization type for encoders: l2norm, simnorm
    teacher_encoder_hidden_dims = [512, 256]
    student_encoder_hidden_dims = [512, 256, 256]
    actor_hidden_dims=[512, 256, 128]
    critic_hidden_dims=[512, 256, 128]
    activation="elu"
    actor_obs_normalization = False
    critic_obs_normalization = False

@configclass
class RslRlMoeCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "MoECTS"
    value_loss_coef = 1.0
    load_balance_coef = 0.01  # coefficient for load balance loss
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    student_encoder_learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    betas = (0.9, 0.999)
    weight_decay = 0.0
    desired_kl = 0.01
    max_grad_norm = 1.0
    teacher_env_ratio = 0.75  # percentage of envs assigned to teacher

@configclass
class MoECTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "a2_moe_cts"
    class_name = "OnPolicyRunnerCTS"
    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 500
    policy = RslRlMoeCtsActorCriticCfg()
    algorithm = RslRlMoeCtsAlgorithmCfg()

# concat elu inspired by concat relu from https://arxiv.org/pdf/2303.07507
@configclass
class MoECTSCatELURunnerCfg(MoECTSRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.activation = 'cat_elu'


@configclass
class StairsMoECTSRunnerCfg(MoECTSRunnerCfg):
    """Conservative, weights-only fine-tuning settings for the A2 stair task."""

    experiment_name = "a2_stairs_cts"
    max_iterations = 2000
    save_interval = 100
    init_at_random_ep_len: bool = False

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.student_encoder_learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.symmetry_cfg = None


@configclass
class StairsSymMoECTSRunnerCfg(StairsMoECTSRunnerCfg):
    """Same task and PPO settings, with opt-in student action mirror consistency."""

    experiment_name = "a2_stairs_sym_cts"

    def __post_init__(self):
        super().__post_init__()
        # Initial experimental weight, not a tuned optimum. Log the unweighted
        # loss and validate gait/performance against StairsMoECTSRunnerCfg.
        self.algorithm.symmetry_cfg = {
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.05,
            "data_augmentation_func": "robot_lab.tasks.a2.mdp.symmetry:mirror_a2_observations",
            "ramp_updates": 200,
        }


@configclass
class GaitWarmRunnerCfg(StairsSymMoECTSRunnerCfg):
    experiment_name = "a2_gait_warm"
    max_iterations = 10000


@configclass
class GaitScratchRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "a2_gait_scratch"
    init_at_random_ep_len: bool = False

    def __post_init__(self):
        self.algorithm.symmetry_cfg = {
            "use_data_augmentation": False, "use_mirror_loss": True, "mirror_loss_coeff": .05,
            "data_augmentation_func": "robot_lab.tasks.a2.mdp.symmetry:mirror_a2_observations",
            "ramp_updates": 1000,
        }
