script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

declare -A ds_dict
ds_dict=([U45]="configs/dataset/u45.yaml"
         [RUIE_Color90]="configs/dataset/ruie_color90.yaml"
         [UPoor200]="configs/dataset/upoor200.yaml"
         [UW2023]="configs/dataset/uw2023_256x256.yaml")

if [ $# -lt 5 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_name"
    echo -e "2) net_cfg"
    echo -e "3) name"
    echo -e "4) epochs"
    echo -e "5) load_prefix"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} ie configs/network/ra_9blocks_2down.yaml LSUI_01 299 weights${ENDSTYLE}\""
    exit -1
fi
model_name=${1}
net_cfg=${2}
name=${3}
raw_epochs=${4}
load_prefix=${5}

epochs=$(echo ${raw_epochs} | tr ',' ' ')

for ds_name in ${!ds_dict[@]};
do
    python ${proj_dir}/test.py \
    --model_name ${model_name} \
    --ds_cfg ${ds_dict[${ds_name}]} \
    --net_cfg ${net_cfg} \
    --name ${name} \
    --test_name ${ds_name} \
    --epochs ${epochs} \
    --load_prefix ${load_prefix}
done