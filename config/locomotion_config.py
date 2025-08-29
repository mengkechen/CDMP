import torch

from params_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 3       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = './weights/'
    dataset = 'TDMA-20'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'   
    horizon = 12      
    n_diffusion_steps = 200      
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)         
    returns_condition = True
    calc_energy=False
    dim=256       #128
    condition_dropout=0.25      
    condition_guidance_w = 1.6  
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.TDMADataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 6000    #1000
    hidden_dim = 256        #128
    ar_inv = True     #False
    train_CDMP_pen = False
    termination_penalty = 0       
    returns_scale = 1.0     

    ## training
    n_steps_per_epoch = 1000           
    loss_type = 'l2'
    n_train_steps = 100000     
    batch_size = 128        
    learning_rate = 1e-4      
    gradient_accumulate_every = 2    
    ema_decay = 0.995   
    log_freq = 1000         
    save_freq = 100000        
    sample_freq = 10000     
    n_saves = 1          
    save_parallel = False
    n_reference = 8      
    save_checkpoints = True     
