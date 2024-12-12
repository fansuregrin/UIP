script_dir=$(dirname $0)
source ${script_dir}/ansi_escape.sh

ds_names=(U45 RUIE_Color90 UPoor200 UW2023)

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

for ds_name in ${ds_names[@]}
do
    target_dir="results/${model_v}/${net}/${name}/${ds_name}"
    if [ -d ${target_dir} ]
    then
        python ./nonref_eval_pd.py \
            --model_v ${model_v} \
            --net ${net} \
            --name ${name} \
            --ds_name ${ds_name} \
            --epochs ${epochs_space_sep} \
            --load_prefix ${load_prefix}
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

python ${script_dir}/get_nonref_vals.py \
    ${model_v} \
    ${net} \
    ${name} \
    ${epochs_space_sep} \
    ${load_prefix}