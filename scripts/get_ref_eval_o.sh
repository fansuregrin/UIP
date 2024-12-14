script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

if [ $# -lt 1 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) result_dir"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} /path/to/results/\""
    exit -1
fi
res_dir=${1}

ds_names="LSUI_Test EUVP515 UIEB100 OceanEx"
python ${script_dir}/get_eval_o.py ${res_dir} \
    --eval_type ref \
    --ds_names ${ds_names} \
    --metric_names psnr ssim