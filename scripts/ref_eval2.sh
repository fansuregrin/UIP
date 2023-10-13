GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
ENDSTYLE="\e[0m"

declare -A refer_dict
refer_dict=([EUVP515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr"
            [OceanEx]="/DataA/pwz/workshop/Datasets/ocean_ex/good"
            [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/reference"
            [LSUI]="/DataA/pwz/workshop/Datasets/LSUI/test/ref")

if [ $# -lt 4 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_v"
    echo -e "2) net"
    echo -e "3) name"
    echo -e "4) epoch"
    echo -e "for example: ${BOLD}bash ${0} ie ra LSUI_01 299${ENDSTYLE}"
    exit -1
fi
model_v=${1}
net=${2}
name=${3}
epoch=${4}

for ds_name in ${!refer_dict[@]}
do
    target_dir="results/${model_v}/${net}/${name}/${ds_name}/epoch_${epoch}"
    if [ -d ${target_dir} ]
    then
        python ./ref_eval.py \
            -inp "${target_dir}/single/predicted" \
            -ref "${refer_dict[${ds_name}]}" \
            -out "${target_dir}" \
            --resize
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

echo -e "reference eval of [${GREEN}${model_v}/${net}/${name}/epoch_${epoch}${ENDSTYLE}]"
echo "================================================"
printf "${BOLD}%-8s %-15s %-8s %-8s %-8s${ENDSTYLE}\n" epoch ds_name psnr ssim mse
echo "------------------------------------------------"
for ds_name in ${!refer_dict[@]}
do
    target_file="results/${model_v}/${net}/${name}/${ds_name}/epoch_${epoch}/ref_eval.csv"
    if [ -f "${target_file}" ]; then
        psnr=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
        ssim=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
        mse=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
        printf "%-8s %-15s %-8s %-8s %-8s\n" ${epoch} ${ds_name} ${psnr} ${ssim} ${mse}
    fi
done
echo "================================================"