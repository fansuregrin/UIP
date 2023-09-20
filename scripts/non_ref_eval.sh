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

for ds_name in ${ds_names[@]}
do
    target_dir="results/${model_v}/${net}/${name}/${ds_name}/epoch_${epoch}"
    if [ -d ${target_dir} ]
    then
        python ./nonref_eval.py \
            --model_v ${model_v} \
            --net ${net} \
            --name ${name} \
            --ds_name ${ds_name} \
            --epochs ${epoch}
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

echo -e "non-reference eval of [${GREEN}${model_v}/${net}/${name}/epoch_${epoch}${ENDSTYLE}]"
echo "================================="
echo -e "${BOLD}epoch\tds_name\tniqe\tmusiq\turanker\tuciqe\tuiqm${ENDSTYLE}"
echo "---------------------------------"
for ds_name in ${ds_names[@]}
do
    target_file="results/${model_v}/${net}/${name}/${ds_name}/epoch_${epoch}/noref_eval.csv"
    if [ -f "${target_file}" ]; then
        niqe=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
        musiq=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
        uranker=`tail "${target_file}" -n 1 | awk -F, '{print $4}'`
        uciqe=`tail "${target_file}" -n 1 | awk -F, '{print $5}'`
        uiqm=`tail "${target_file}" -n 1 | awk -F, '{print $6}'`
        echo -e "${epoch}\t${ds_name}\t${niqe}\t${musiq}\t${uranker}\t${uciqe}\t${uiqm}"
    fi
done
echo "================================="