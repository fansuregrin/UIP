script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

if [ $# -lt 5 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_v"
    echo -e "2) net"
    echo -e "3) name"
    echo -e "4) epochs"
    echo -e "5) load_prefix"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} ie ra LSUI_01 299 weights${ENDSTYLE}\""
    exit -1
fi
model_v=${1}
net=${2}
name=${3}
raw_epochs=${4}
load_prefix=${5}

epochs_space_sep=$(echo ${raw_epochs} | tr ',' ' ')
ds_names="U45 RUIE_Color90 UPoor200 UW2023"
python ${script_dir}/get_eval_m.py \
    ${model_v} \
    ${net} \
    ${name} \
    ${epochs_space_sep} \
    --eval_type nonref \
    --ds_names ${ds_names} \
    --metric_names uranker uciqe uiqm \
    --load_prefix ${load_prefix}