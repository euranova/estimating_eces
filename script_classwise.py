from explicalib.experiments.setups.comparison_with_gt.classwise import *

n_classes = [2, 5, 7]
n_features=[2, 5, 7]
seeds_distribution = [i for i in range(5)]

compute(setup_name="classwise", 
        seeds_distribution=seeds_distribution, 
        n_classes=n_classes, 
        n_features=n_features)
