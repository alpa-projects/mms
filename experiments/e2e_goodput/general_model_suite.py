from collections import namedtuple

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    [
     "num_devices_list", "num_modelset_list", "slo_scales",
     "rate_list", "cv_list", # synthetic trace only
     "rate_scales", "cv_scales", # real trace only
    ]
)

synthetic_suite = {
    "all_transformers": BenchmarkConfig(    
        num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120],
        num_modelset_list = [1, 4, 8, 16, 24, 32],
        slo_scales = [1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 8, 10, 12, 14, 16],
        rate_list = [8, 16, 24, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 2, 4, 8, 16],
        cv_scales = [1, 2, 4, 8, 16],
    ),
    "mixed": BenchmarkConfig(    
        num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120],
        num_modelset_list = [1, 2, 4, 6, 8, 10, 12],
        slo_scales = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        rate_list = [8, 16, 24, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 1.5, 2, 2.5, 3, 3.5, 4],
        rate_scales = [1, 2, 4, 8, 16],
        cv_scales = [1, 2, 4, 8, 16],
    ),
}

azure_v1_suite = {
    "all_transformers": BenchmarkConfig(    
        num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
        num_modelset_list = [4, 8, 12, 16, 20, 24, 32],
        slo_scales = [1, 1.5, 2, 2.5, 5, 7.5, 10],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 2, 4, 8, 16],
        cv_scales = [1, 2, 4, 8, 16],
    ),
    "mixed": BenchmarkConfig(    
        num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
        num_modelset_list = [4, 6, 8, 12, 16, 20],
        slo_scales = [1, 1.5, 2, 2.5, 5, 7.5, 10],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 2, 4, 8, 16],
        cv_scales = [1, 2, 4, 8, 16],
    ),
}

azure_v2_suite = {
    "all_transformers": BenchmarkConfig(    
        num_devices_list = [4, 8, 12, 16, 20, 24, 32, 40],
        num_modelset_list =  [8, 12, 16, 20, 24, 28, 32],
        slo_scales = [0.75, 1, 1.25, 1.5, 2, 2.5, 5],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 2, 4, 8, 16, 32, 64],
        cv_scales = [1, 2, 3, 4, 6, 8],
    ),
    "mixed": BenchmarkConfig(    
        num_devices_list = [4, 8, 12, 16, 20, 24, 32, 40],
        num_modelset_list =  [8, 12, 16, 20, 24, 28, 32],
        slo_scales = [0.75, 1, 1.25, 1.5, 2, 2.5, 5],
        rate_list = [8, 16, 32, 64, 96, 128, 160, 192],
        cv_list = [1, 2, 4, 6, 8],
        rate_scales = [1, 2, 4, 8, 16, 32, 64],
        cv_scales = [1, 2, 3, 4, 6, 8],
    ),
}