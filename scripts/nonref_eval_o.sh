script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

ds_names=(U45 RUIE_Color90 UPoor200 UW2023)

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

for ds_name in ${ds_names[@]}
do
    target_dir="${res_dir}/${ds_name}"
    if [ -d ${target_dir} ]
    then
        python ${proj_dir}/nonref_eval_pd.py \
            -inp "${target_dir}/${sub_path}" \
            -out "${target_dir}"
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

ds_names_=$(echo "${ds_names[@]}")
python ${script_dir}/get_eval_o.py ${res_dir} \
    --eval_type nonref \
    --ds_names ${ds_names_} \
    --metric_names uranker uciqe uiqm