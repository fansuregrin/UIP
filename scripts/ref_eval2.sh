GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
BOLD_GREEN="\e[1;32m"
ENDSTYLE="\e[0m"

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

for ds_name in ${!refer_dict[@]}
do
    for epoch in ${epochs[*]}
    do
        target_dir="results/${model_v}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}"
        if [ -d ${target_dir} ]
        then
            python ./ref_eval_pd.py \
                -inp "${target_dir}/single/predicted" \
                -ref "${refer_dict[${ds_name}]}" \
                -out "${target_dir}" \
                --resize \
                # --width 224 --height 224 \
        else
            echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
        fi
    done
done

epochs_space_sep=$(echo ${raw_epochs} | tr ',' ' ')
script_dir=$(dirname $0)
python ${script_dir}/get_ref_vals.py \
    ${model_v} \
    ${net} \
    ${name} \
    ${epochs_space_sep} \
    ${load_prefix}