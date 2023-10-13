GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
ENDSTYLE="\e[0m"

ds_names=(U45 RUIE_Color90 UPoor200 UW2023)

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

echo -e "non-reference eval of [${GREEN}${model_v}/${net}/${name}/epoch_${epoch}${ENDSTYLE}]"
echo "=================================================================="
printf "${BOLD}%-8s %-15s %-8s %-8s %-8s %-8s %-8s${ENDSTYLE}\n" epoch ds_name niqe musiq uranker uciqe uiqm
echo "------------------------------------------------------------------"
for ds_name in ${ds_names[@]}
do
    target_file="results/${model_v}/${net}/${name}/${ds_name}/epoch_${epoch}/noref_eval.csv"
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