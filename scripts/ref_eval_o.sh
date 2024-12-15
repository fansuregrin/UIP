script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

declare -A refer_dict
refer_dict=([EUVP515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr"
            [OceanEx]="/DataA/pwz/workshop/Datasets/ocean_ex/good"
            [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/reference"
            [LSUI_Test]="/DataA/pwz/workshop/Datasets/LSUI/test/ref")

if [ $# -lt 1 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) result_dir"
    echo -e "2) sub_path (optional)"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} /path/to/results/\""
    exit -1
fi
res_dir=${1}
sub_path=
if [ $# -gt 1 ]; then
    sub_path=${2}
fi

for ds_name in ${!refer_dict[@]}
do
    target_dir="${res_dir}/${ds_name}"
    if [ -d ${target_dir} ]
    then
        python ${proj_dir}/ref_eval_pd.py \
            -inp "${target_dir}/${sub_path}" \
            -ref "${refer_dict[${ds_name}]}" \
            -out "${target_dir}" \
            --resize
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

ds_names=$(echo "${!refer_dict[@]}")
python ${script_dir}/get_eval_o.py ${res_dir} \
    --eval_type ref \
    --ds_names ${ds_names} \
    --metric_names psnr ssim
# echo -e "reference eval of [${GREEN}${res_dir}${ENDSTYLE}]"
# echo "================================================"
# printf "${BOLD}%-15s %-8s %-8s %-8s${ENDSTYLE}\n" ds_name psnr ssim mse
# echo "------------------------------------------------"
# for ds_name in ${!refer_dict[@]}
# do
#     target_file="${res_dir}/${ds_name}/ref_eval.csv"
#     if [ -f "${target_file}" ]; then
#         psnr=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
#         ssim=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
#         mse=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
#         printf "%-15s %-8s %-8s %-8s\n" ${ds_name} ${psnr} ${ssim} ${mse}
#     fi
# done
# echo "================================================"