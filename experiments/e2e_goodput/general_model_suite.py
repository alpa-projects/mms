from collections import namedtuple

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    [
     "fixed_num_devices", "fixed_num_modelset", "fixed_slo_scale", # general
     "fixed_rate_scale", "fixed_cv_scale", # real trace only
     "num_devices_list", "num_modelset_list", "slo_scales",
     "rate_list", "cv_list", # synthetic trace only
     "rate_scales", "cv_scales", # real trace only
    ]
)

synthetic_suite = {
    "all_transformers": BenchmarkConfig(    
        fixed_num_devices = 24,
        fixed_num_modelset = 20,
        fixed_slo_scale = 5,
        fixed_rate_scale = None,
        fixed_cv_scale = None,
        num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120],
        num_modelset_list = [1, 4, 6, 8, 10, 12, 14, 16],
        slo_scales = [1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 8, 10, 12, 14, 16],
        rate_list = [8, 16, 24, 32, 48, 64, 80, 96],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [],
        cv_scales = [],
    ),
    "mixed": BenchmarkConfig(    
        fixed_num_devices = 32,
        fixed_num_modelset = 10,
        fixed_slo_scale = 5,
        fixed_rate_scale = None,
        fixed_cv_scale = None,
        num_devices_list = [8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200],
        num_modelset_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        slo_scales = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        rate_list = [8, 16, 24, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        rate_scales = [],
        cv_scales = [],
    ),
}

azure_v1_suite = {
    "all_transformers": BenchmarkConfig(    
        fixed_num_devices = 24,
        fixed_num_modelset = 12,
        fixed_slo_scale = 5,
        fixed_rate_scale = 2e-3,
        fixed_cv_scale = 4,
        num_devices_list = [8, 16, 24, 32, 40, 48],
        num_modelset_list =  [8, 12, 16, 20, 24, 28, 32],
        slo_scales = [0.75, 1, 2, 3, 4, 5, 7.5, 10],
        rate_list = [],
        cv_list = [],
        rate_scales = [5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3],
        cv_scales = [1, 2, 3, 4, 5, 6, 8],
    ),
    "mixed": BenchmarkConfig(    
        fixed_num_devices = 48,
        fixed_num_modelset = 12,
        fixed_slo_scale = 5,
        fixed_rate_scale = 2e-3,
        fixed_cv_scale = 4,
        num_devices_list = [32, 40, 48, 54, 64, 72, 96],
        num_modelset_list =  [8, 12, 16, 20, 24, 28, 32],
        slo_scales =[0.75, 1, 2, 3, 4, 5, 7.5, 10],
        rate_list = [],
        cv_list = [],
        rate_scales = [5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3],
        cv_scales = [1, 2, 3, 4, 5, 6, 8],
    ),
}

azure_v2_suite = {
    "all_transformers": BenchmarkConfig(    
        fixed_num_devices = 24,
        fixed_num_modelset = 12,
        fixed_slo_scale = 5,
        fixed_rate_scale = 32,
        fixed_cv_scale = 1,
        num_devices_list = [8, 16, 20, 24, 32, 40, 48],
        num_modelset_list =  [8, 12, 16, 20, 24],
        slo_scales = [0.75, 1, 1.25, 1.5, 2, 2.5, 5, 10],
        rate_list = [],
        cv_list = [],
        rate_scales = [1, 4, 8, 16, 32, 64, 128],
        cv_scales = [1, 2, 3, 4, 5, 6, 8],
    ),
    "mixed": BenchmarkConfig(    
        fixed_num_devices = 48,
        fixed_num_modelset = 12,
        fixed_slo_scale = 5,
        fixed_rate_scale = 32,
        fixed_cv_scale = 1,
        num_devices_list = [16, 24, 32, 40, 48, 54, 64],
        num_modelset_list =  [8, 12],
        slo_scales = [0.75, 1, 1.25, 1.5, 2, 2.5, 5],
        rate_list = [],
        cv_list = [],
        rate_scales = [1, 4, 8, 16, 32, 64],
        cv_scales = [1, 2, 3, 4, 6, 8],
    ),
}
