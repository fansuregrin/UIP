script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh


################## parse arguments and options ###############################
short_opt=h,t
long_opt=help,load_prefix:,ds_names:,output_format,dry_run
options=$(getopt -a --options ${short_opt} --longoptions ${long_opt} -- "$@")
# echo ${options}

eval set -- "${options}"

help_str="
Usage:
 $0 model_name network name epochs [options]

Non-reference Evaluation.

Positional Arguments:
 model_name                     name of the model
 network                        name of the network
 name                           name of traing progress
 epochs                         epochs

Options:
 --load_prefix <load_prefix>    prefix of the filename of weight file
 --ds_names <ds_names>          test dataset names
 --output_format <format>...    format for saving evaluation data         
 -t, --dry_run                  just test not run actually
 -h, --help                     display this help
"

load_prefix=weights
output_format="csv pkl"
ds_names="U45 RUIE_Color90 UPoor200 UW2023"
dry_run=

while true; do
  case "$1" in
    --load_prefix)
      load_prefix="$2"
      shift 2
      ;;
    --ds_names)
      ds_names="$2"
      shift 2
      ;;
    --output_format)
      output_format="$2"
      shift 2
      ;;
    -t | --dry_run)
      dry_run="echo "
      shift 1
      ;;
    -h | --help)
      echo -e "${help_str}"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done

if [ $# -lt 4 ];then
    echo -e "${help_str}"
    exit 1
fi

model_name=${1}
net=${2}
name=${3}
epochs=${4}
################## parse arguments and options ###############################


ds_names_arr=(${ds_names})
epochs_arr=(${epochs})
for ds_name in ${ds_names_arr[@]}
do
    for epoch in ${epochs_arr[*]}
    do
        target_dir="${proj_dir}/results/${model_name}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}"
        if [ -d ${target_dir} ]
        then
            ${dry_run}python ${proj_dir}/nonref_eval_pd.py \
                -inp "${target_dir}/single/predicted" \
                -out "${target_dir}" \
                --output_format ${output_format}
        else
            echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
        fi
    done
done

${dry_run}python ${script_dir}/get_eval_m.py \
    ${model_name} \
    ${net} \
    ${name} \
    ${epochs} \
    --eval_type nonref \
    --ds_names ${ds_names} \
    --metric_names uranker uciqe uiqm \
    --load_prefix ${load_prefix}