from rllab.algos.server_diffenvs import Server
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.server_env import ServerEnv
from rllab.envs.normalized_env import NormalizedEnv
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sys

def run_task(v):

    env = ServerEnv(agents_number, -10, 10)

    policy = GaussianMLPPolicy(
        env_spec=env.agents_envs[0].spec,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = Server(
        agents_number=agents_number,
        average_period=average_period,
        server_env=env,
        policy=policy,
        baseline=baseline,
        batch_size=400,
        max_path_length=100,
        n_itr=20,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    algo.train()


agents_numbers = [3, 8, 10, 12, 14]
average_periods = [2, 5, 7, 9]

for agents_number in agents_numbers:
    for average_period in average_periods:
        run_experiment_lite(
            run_task,
            exp_prefix="multi_agent_per_av_quant_diffenvs_test",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            mode="local",
            variant=dict(agents_number=agents_number, average_period=average_period)
            # plot=True,
            # terminate_machine=False,
        )