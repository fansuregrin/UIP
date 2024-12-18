script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh


################## parse arguments and options ###############################
short_opt=h,t
long_opt=help,dry_run,load_prefix:,ds_names:,file_fmt:
options=$(getopt -a --options ${short_opt} --longoptions ${long_opt} -- "$@")
# echo ${options}

eval set -- "${options}"

help_str="
Usage:
 $0 model_name network name epochs [options]

Get Non-reference Evaluation Data.

Positional Arguments:
 model_name                     name of the model
 network                        name of the network
 name                           name of traing progress
 epochs                         epochs

Options:
 --load_prefix <load_prefix>    prefix of the filename of weight file                  
 --ds_names <ds_names>          test dataset names
 --file_fmt <file_fmt>          file format
 -t, --dry_run                  just test not run actually
 -h, --help                     display this help
"

load_prefix=weights
ds_names="U45 RUIE_Color90 UPoor200 UW2023"
file_fmt=pkl
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
    --file_fmt)
      file_fmt="$2"
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


${dry_run}python ${script_dir}/get_eval_m.py \
    ${model_name} \
    ${net} \
    ${name} \
    ${epochs} \
    --eval_type nonref \
    --file_fmt ${file_fmt} \
    --ds_names ${ds_names} \
    --metric_names uranker uciqe uiqm \
    --load_prefix ${load_prefix}