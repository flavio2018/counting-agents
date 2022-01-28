#!/bin/bash

args=(
    --single_or_multi_agent 'single'
    --task 'classify'
    # Type of External tool: ['MoveAndWrite', 'WriteCoord', 'Abacus']
    --external_repr_tool 'WriteCoord'
    # Way the numerosity is presented: ['spatial', 'temporal']
    --observation 'spatial'
    # Training starts with maximimum of max_objects presented objects
    --max_objects 1
    # If curriculum_learning==True --> max_objects will be increased until max_max_objects
    --max_max_objects 9
    --curriculum_learning True
    # Run num_iterations, except agent masters all before
    --num_iterations 60000
    # obs_ext_shape determines the shape of the observation and the external tool
    --obs_ext_shape 10 1
    # exp_name will define the subfolder in which the results will be saved
    --collect_n_episodes_per_itr 64
    --exp_name static_relu
    --use_gpu False
)

python3 run_experiment.py "${args[@]}"
