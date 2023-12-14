GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
BOLD_GREEN="\e[1;32m"
ENDSTYLE="\e[0m"

ds_names=(U45 RUIE_Color90 UPoor200 UW2023)

if [ $# -lt 5 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_v"
    echo -e "2) net"
    echo -e "3) name"
    echo -e "4) epoch"
    echo -e "5) load_prefix"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} ie ra LSUI_01 299 weights${ENDSTYLE}\""
    exit -1
fi
model_v=${1}
net=${2}
name=${3}
epoch=${4}
load_prefix=${5}

for ds_name in ${ds_names[@]}
do
    target_dir="results/${model_v}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}"
    if [ -d ${target_dir} ]
    then
        python ./nonref_eval.py \
            --model_v ${model_v} \
            --net ${net} \
            --name ${name} \
            --ds_name ${ds_name} \
            --epochs ${epoch} \
            --load_prefix ${load_prefix}
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

echo -e "non-reference eval of [${GREEN}${model_v}/${net}/${name}/${load_prefix}_${epoch}${ENDSTYLE}]"
echo "=================================================================="
printf "${BOLD}%-8s %-15s %-8s %-8s %-8s %-8s %-8s${ENDSTYLE}\n" epoch ds_name niqe musiq uranker uciqe uiqm
echo "------------------------------------------------------------------"
for ds_name in ${ds_names[@]}
do
    target_file="results/${model_v}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}/noref_eval.csv"
    if [ -f "${target_file}" ]; then
        niqe=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
        musiq=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
        uranker=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
        uciqe=`tail "${target_file}" -n 1 | awk -F, '{print $5}'`
        uiqm=`tail "${target_file}" -n 1 | awk -F, '{print $6}'`
        printf "%-8s %-15s %-8s %-8s %-8s %-8s %-8s\n" ${epoch} ${ds_name} ${niqe} ${musiq} ${uranker} ${uciqe} ${uiqm}
    fi
done
echo "=================================================================="