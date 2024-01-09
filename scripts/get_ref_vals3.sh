GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
BOLD_GREEN="\e[1;32m"
ENDSTYLE="\e[0m"

declare -A refer_dict
refer_dict=([EUVP515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr"
            [OceanEx]="/DataA/pwz/workshop/Datasets/ocean_ex/good"
            [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/reference"
            [LSUI_Test]="/DataA/pwz/workshop/Datasets/LSUI/test/ref")

if [ $# -lt 1 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) result_dir"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} /path/to/results/\""
    exit -1
fi
res_dir=${1}

echo -e "reference eval of [${GREEN}${res_dir}${ENDSTYLE}]"
echo "================================================"
printf "${BOLD}%-15s %-8s %-8s %-8s${ENDSTYLE}\n" ds_name psnr ssim mse
echo "------------------------------------------------"
for ds_name in ${!refer_dict[@]}
do
    target_file="${res_dir}/${ds_name}/ref_eval.csv"
    if [ -f "${target_file}" ]; then
        psnr=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
        ssim=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
        mse=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
        printf "%-15s %-8s %-8s %-8s\n" ${ds_name} ${psnr} ${ssim} ${mse}
    fi
done
echo "================================================"