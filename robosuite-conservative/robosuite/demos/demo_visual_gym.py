from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers.gym_visual_wrapper import GymVisualWrapper
from robosuite.wrappers.gym_visual_cat_wrapper import GymVisualCatWrapper
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Env args')
    parser.add_argument('--wrapper', type=str, help='Gym visual wrapper configs')
    args = parser.parse_args()

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == 'bimanual':
            options["robots"] = 'Baxter'
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Load default variants
    try:
        with open(args.wrapper) as f:
            variant = json.load(f)
    except FileNotFoundError:
        print("Error opening specified variant json at: {}. "
              "Please check filepath and try again.".format(variant))

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    variant["environment_kwargs"].pop("controller")

    # initialize the task
    env = suite.make(
        controller_configs=options["controller_configs"],
        has_renderer=False,
        **variant["environment_kwargs"],
    )

    # Notice how the environment is wrapped by the wrapper
    env = GymVisualCatWrapper(env, **variant["wrapper_kwargs"])

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
