
class Config():
    Stu_model_name = "vgg11" # resnet18、vgg11
    # Mentor_model_name = "ViT" # ViT,"resnet152
    dataset = "tiny"
    distribution_alpha = 0.5
    n_clients = 9
    data_diribution_balancedness_for_clents = False
    batch_size_for_clients = 32
    test_batch_size = 32
    communication_rounds = 50
    epochs_for_clients = 1
    SNR = 25
    SNR_MAX = 25
    SNR_MIN = 0
    Bandwidth_MAX = 10
    Power_MAX = 0.1
    eta = 1
    use_Rali = True
    use_RTN = False
    isc_lr = 1e-3
    channel_lr = 1e-3
    weight_delay = 1e-6
    device = "cuda"
    checkpoints_dir = "checkpoints"
    logs_dir = "logs"
    class_num = 200