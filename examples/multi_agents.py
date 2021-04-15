from rllab.algos.server import Server
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sys

def run_task(v):

    print("_________________________________")
    print("#################################")
    print("_________________________________")
    print("_________________________________")
    print("#################################")
    print("###    agents_number : " + str(agents_number) +"    ####")
    print("###                          ####")
    print("### participation_rate : " + str(participation_rate) +" ####")
    print("###                          ####")
    print("###    average_period : " + str(average_period) +"   ####")
    print("###                          ####")
    print("### quantization_tuning : " + str(quantization_tuning) +" ####")
    print("#################################")
    print("_________________________________")
    print("_________________________________")
    print("#################################")
    print("_________________________________")

    env = normalize(CartpoleEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = Server(
        participation_rate=participation_rate,
        agents_number=agents_number,
        average_period=average_period,
        env=env,
        policy=policy,
        baseline=baseline,
        difference_params=False,
        quantize=False,
        quantization_tuning=quantization_tuning,
        batch_size=400,
        max_path_length=100,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    algo.train()

quantization_tunings = [0]
participation_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
agents_numbers = [1, 5, 10]
average_periods = [1]

for quantization_tuning in quantization_tunings:
    for participation_rate in participation_rates:
        for agents_number in agents_numbers:
            for average_period in average_periods:
                run_experiment_lite(
                    run_task,
                    exp_prefix="test_partrates_params_notquant",
                    # Number of parallel workers for sampling
                    n_parallel=1,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="last",
                    # Specifies the seed for the experiment. If this is not provided, a random seed
                    # will be used
                    mode="local",
                    variant=dict(quantization_tuning=quantization_tuning, participation_rate=participation_rate, agents_number=agents_number, average_period=average_period)
                    # plot=True,
                    # terminate_machine=False,
                )