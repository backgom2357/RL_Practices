class Config:

    render = False

    memory_size = 100000

    frame_height = 48
    frame_width = 48

    batch_size = 8

    learning_rate = 0.00025
    gradient_momentum = 0.95

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.9/1000000
    discount_factor = 0.99

    start_train = 50000
    train_frequency = 4

    steps = 1000000

    target_network_update_frequency = 10000

    render = True

    logdir = './logs/scalars/'
