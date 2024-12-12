script_dir=$(dirname $0)
source ${script_dir}/ansi_escape.sh

declare -A refer_dict
refer_dict=([EUVP515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr"
            [OceanEx]="/DataA/pwz/workshop/Datasets/ocean_ex/good"
            [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/reference"
            [LSUI]="/DataA/pwz/workshop/Datasets/LSUI/test/ref")

if [ $# -lt 5 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_v"
    echo -e "2) net"
    echo -e "3) name"
    echo -e "4) epochs: multiple epochs must be separated by commas"
    echo -e "5) load_prefix"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} ie ra LSUI_01 299 weights${ENDSTYLE}\""
    exit -1
fi
model_v=${1}
net=${2}
name=${3}
raw_epochs=${4}
epochs=(${raw_epochs//,/ })
load_prefix=${5}

num_epoch=${#epochs[@]}
if [ ${num_epoch} -gt 1 ]
then
    # fetch multiple epochs results for each test_set
    echo -e "reference eval of [${GREEN}${model_v}/${net}/${name}/${load_prefix}_{${raw_epochs}}${ENDSTYLE}]"
    for ds_name in ${!refer_dict[@]}
    do
        echo -e "${BOLD_BLUE}${ds_name}:${ENDSTYLE}"
        echo -e "=========================================="
        printf "${BOLD}%-8s %-8s %-8s %-8s${ENDSTYLE}\n" epoch psnr ssim mse
        echo -e "------------------------------------------"
        for epoch in ${epochs[*]}
        do
            target_file="results/${model_v}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}/ref_eval.csv"
            if [ -f "${target_file}" ]; then
                psnr=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
                ssim=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
                mse=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
                printf "%-8s %-8s %-8s %-8s\n" ${epoch} ${psnr} ${ssim} ${mse}
            fi
        done
        echo -e "==========================================\n"
    done
else
    epoch=${epochs[0]}
    echo -e "reference eval of [${GREEN}${model_v}/${net}/${name}/${load_prefix}_${epoch}${ENDSTYLE}]"
    echo -e "=================================================="
    printf "${BOLD}%-8s %-15s %-8s %-8s %-8s${ENDSTYLE}\n" epoch ds_name psnr ssim mse
    echo -e "--------------------------------------------------"
    for ds_name in ${!refer_dict[@]}
    do
        target_file="results/${model_v}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}/ref_eval.csv"
        if [ -f "${target_file}" ]; then
            psnr=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
            ssim=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
            mse=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
            printf "%-8s %-15s %-8s %-8s %-8s\n" ${epoch} ${ds_name} ${psnr} ${ssim} ${mse}
        fi
    done
    echo -e "==================================================\n"
fi
