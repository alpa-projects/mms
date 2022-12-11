from collections import namedtuple

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    [
     "fixed_num_devices", "fixed_num_models", "fixed_slo_scale", # general
     "fixed_rate_scale", "fixed_cv_scale", # real trace only
     "num_devices_list", "num_models_list", "slo_scales", # general
     "rate_list", "cv_list", # synthetic trace only
     "rate_scales", "cv_scales", # real trace only
    ]
)

synthetic_suite = {
    "bert-1.3b": BenchmarkConfig(    
        fixed_num_devices = 32,
        fixed_num_models = 32,
        fixed_slo_scale = 5,
        fixed_rate_scale = None,
        fixed_cv_scale = None,
        num_devices_list = [2, 4, 8, 16, 24, 32, 40],
        num_models_list = [16, 32, 48, 64, 80, 96],
        slo_scales = [1, 1.5, 2, 2.5, 4, 5, 6, 7],
        rate_list = [16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8, 10, 12],
        rate_scales = [],
        cv_scales = [],
    ),
    "bert-2.6b": BenchmarkConfig(    
        fixed_num_devices = 32,
        fixed_num_models = 32,
        fixed_slo_scale = 5,
        fixed_rate_scale = None,
        fixed_cv_scale = None,
        num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88],
        num_models_list = [4, 8, 16, 24, 32, 40, 48, 56, 64],
        slo_scales = [1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        rate_list = [4, 8, 16, 32, 64, 96, 128],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [],
        cv_scales = [],
    ),
    "bert-6.7b": BenchmarkConfig(    
        fixed_num_devices = 32,
        fixed_num_models = 32,
        fixed_slo_scale = 5,
        fixed_rate_scale = None,
        fixed_cv_scale = None,
        num_devices_list = [24, 32, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184],
        num_models_list = [2, 4, 6, 8, 12, 16, 24, 32],
        slo_scales = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
        rate_list = [1, 2, 3, 4, 8, 12, 16, 20],
        cv_list = [0.2, 0.5, 1, 2, 3, 4],
        rate_scales = [],
        cv_scales = [],
    ),
}

azure_v1_suite = {
    "bert-1.3b": BenchmarkConfig(    
        fixed_num_devices = 16,
        fixed_num_models = 48,
        fixed_slo_scale = 5,
        fixed_rate_scale = 2e-3,
        fixed_cv_scale = 4,
        num_devices_list = [8, 16, 24, 32, 40, 48],
        num_models_list = [36, 48, 60, 72, 84, 96],
        slo_scales = [0.75, 1, 2, 3, 4, 5, 7.5, 10],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3],
        cv_scales = [1, 2, 3, 4, 5, 6, 8],
    ),
    "bert-2.6b": BenchmarkConfig(    
        fixed_num_devices = 32,
        fixed_num_models = 48,
        fixed_slo_scale = 5,
        fixed_rate_scale = 2e-3,
        fixed_cv_scale = 4,
        num_devices_list = [16, 24, 32, 40, 48, 56, 64],
        num_models_list = [32, 40, 48, 56, 72, 96, 128],
        slo_scales = [0.75, 1, 2, 3, 4, 5, 7.5, 10],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3],
        cv_scales = [1, 2, 3, 4, 5, 6, 8],
    ),
    "bert-6.7b": BenchmarkConfig(    
        fixed_num_devices = 64,
        fixed_num_models = 48,
        fixed_slo_scale = 5,
        fixed_rate_scale = 2e-3,
        fixed_cv_scale = 4,
        num_devices_list=[48, 56, 64, 72, 80, 96, 128],
        num_models_list=[40, 48, 64, 72, 84, 96, 108],
        slo_scales=[0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10],
        rate_list=[8, 16, 32, 64, 96, 128, 160, 192],
        cv_list=[1, 2, 4, 6, 8],
        rate_scales=[5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3],
        cv_scales=[1, 2, 3, 4, 5, 6, 8],
    ),
}

azure_v2_suite = {
    "bert-1.3b": BenchmarkConfig(
        fixed_num_devices = 16,
        fixed_num_models = 48,
        fixed_slo_scale = 5,
        fixed_rate_scale = 32,
        fixed_cv_scale = 1,
        num_devices_list = [4, 8, 12, 16, 20],
        num_models_list = [32, 40, 56, 64, 72, 80],
        slo_scales = [0.75, 1, 1.5, 2, 3, 4, 5, 7.5],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 4, 16, 32, 48, 72, 96, 128, 256],
        cv_scales = [1, 1.5, 2, 2.5, 3, 4, 5],
    ),
    "bert-2.6b": BenchmarkConfig(    
        fixed_num_devices = 32,
        fixed_num_models = 48,
        fixed_slo_scale = 5,
        fixed_rate_scale = 32,
        fixed_cv_scale = 1,
        num_devices_list =  [16, 24, 32, 40, 48, 56, 64],
        num_models_list = [32, 40, 56, 64, 72, 80],
        slo_scales = [0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 20],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 8, 16, 32, 48, 72, 96, 128],
        cv_scales = [1, 1.5, 2, 2.5, 3, 4, 5],
    ),
    "bert-6.7b": BenchmarkConfig(
        fixed_num_devices = 64,
        fixed_num_models = 48,
        fixed_slo_scale = 5,
        fixed_rate_scale = 32,
        fixed_cv_scale = 1,
        num_devices_list=[48, 56, 64, 72, 80, 96, 128],
        num_models_list=[40, 48, 56, 72, 84, 96, 108],
        slo_scales=[0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10],
        rate_list=[8, 16, 32, 64, 96, 128, 160, 192],
        cv_list=[1, 2, 4, 6, 8],
        rate_scales=[1, 8, 16, 32, 64, 96, 128],
        cv_scales=[1, 1.5, 2, 2.5, 3, 3.5, 4, 5],
    ),
}
