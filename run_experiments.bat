@REM #!/bin/bash

@REM ########################
@REM ### MAIN EXPERIMENTS ###
@REM ########################

@REM #################################################################
@REM ## IMAGE ONLY ##
python train.py --seed 0 --model image-only 
python train.py --seed 1 --model image-only
python train.py --seed 2 --model image-only
python train.py --seed 3 --model image-only
python train.py --seed 4 --model image-only
@REM ## END IMAGE ONLY ##
@REM #################################################################


@REM #################################################################
@REM ## FEATURE FUSION ##
python train.py --seed 0 --model feature-fusion
python train.py --seed 1 --model feature-fusion
python train.py --seed 2 --model feature-fusion
python train.py --seed 3 --model feature-fusion
python train.py --seed 4 --model feature-fusion
@REM ## END FEATURE FUSION ##
@REM #################################################################

@REM @REM #################################################################
@REM @REM ## LEARNED FEATURE FUSION (CONCAT) ##
python train.py --seed 0 --model learned-feature-fusion
python train.py --seed 1 --model learned-feature-fusion
python train.py --seed 2 --model learned-feature-fusion
python train.py --seed 3 --model learned-feature-fusion
python train.py --seed 4 --model learned-feature-fusion
@REM @REM ## END LEARNED FEATURE FUSION (CONCAT) ##
@REM @REM #################################################################

@REM #################################################################
@REM ## NON IMAGE ONLY ##
python train.py --seed 0 --model non-image-only
python train.py --seed 1 --model non-image-only
python train.py --seed 2 --model non-image-only
python train.py --seed 3 --model non-image-only
python train.py --seed 4 --model non-image-only
@REM ## END NON IMAGE ONLY ##
@REM #################################################################

@REM ############################
@REM ### END MAIN EXPERIMENTS ###
@REM ############################


@REM #############################
@REM ### FEATURE IMPORTANCE ON META FEATURES (ONLY FOR NON-IMAGE OR FEATURE FUSION)###
@REM #############################

@REM #################################################################
@REM ## FEATURE FUSION ##
python feature_imp.py --model feature-fusion --model_name feature-fusion_no-CW_aug_seed0
python feature_imp.py --model feature-fusion --model_name feature-fusion_no-CW_aug_seed1
python feature_imp.py --model feature-fusion --model_name feature-fusion_no-CW_aug_seed2
python feature_imp.py --model feature-fusion --model_name feature-fusion_no-CW_aug_seed3
python feature_imp.py --model feature-fusion --model_name feature-fusion_no-CW_aug_seed4
@REM ## END FEATURE FUSION ##
@REM #################################################################

@REM #################################################################
@REM ## LEARNED FEATURE FUSION (CONCAT) ##
python feature_imp.py --model learned-feature-fusion --model_name learned-feature-fusion-concat_no-CW_aug_seed0
python feature_imp.py --model learned-feature-fusion --model_name learned-feature-fusion-concat_no-CW_aug_seed1
python feature_imp.py --model learned-feature-fusion --model_name learned-feature-fusion-concat_no-CW_aug_seed2
python feature_imp.py --model learned-feature-fusion --model_name learned-feature-fusion-concat_no-CW_aug_seed3
python feature_imp.py --model learned-feature-fusion --model_name learned-feature-fusion-concat_no-CW_aug_seed4
@REM ## END LEARNED FEATURE FUSION (CONCAT) ##
@REM #################################################################

@REM #################################################################
@REM ## NON IMAGE ONLY ##
python feature_imp.py --model non-image-only --model_name non-image-only_no-CW_seed0
python feature_imp.py --model non-image-only --model_name non-image-only_no-CW_seed1
python feature_imp.py --model non-image-only --model_name non-image-only_no-CW_seed2
python feature_imp.py --model non-image-only --model_name non-image-only_no-CW_seed3
python feature_imp.py --model non-image-only --model_name non-image-only_no-CW_seed4
@REM ## END NON IMAGE ONLY ##
@REM #################################################################

@REM #############################
@REM ### END FEATURE IMPORTANCE ###
@REM #############################